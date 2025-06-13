#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

sys.path.append("..")
from samsungtvws import SamsungTVWS  # type: ignore

# Optional deps
try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError:  # pragma: no cover
    # graceful no‑retry shim
    def retry(*a, **k):  # type: ignore
        def _wrap(f):  # noqa: D401
            return f

        return _wrap

    def stop_after_attempt(*a, **k):  # type: ignore
        return None

    def wait_exponential(*a, **k):  # type: ignore
        return None

    def retry_if_exception_type(*a, **k):  # type: ignore
        return None

import hashlib


_IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")


class RedactIPFilter(logging.Filter):
    """Mask IPv4 addresses unless DEBUG level or explicit opt‑out."""

    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        if self.enabled and record.levelno < logging.DEBUG:
            record.msg = _IP_RE.sub("<IP>", str(record.msg))
        return True


def configure_logging(level: str, *, redact_ip: bool) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.getLogger().addFilter(RedactIPFilter(enabled=redact_ip))


def load_state(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logging.warning("State file corrupt – starting fresh.")
        return []


def save_state(path: Path, data: List[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def sha1_of_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def is_hidden(p: Path) -> bool:
    return any(part.startswith(".") for part in p.parts)


def scan_images(root: Path, recursive: bool) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    iterator = root.rglob("*") if recursive else root.iterdir()
    return [
        p
        for p in iterator
        if p.suffix.lower() in exts and p.is_file() and not is_hidden(p) and not is_hidden(p.parent)
    ]


def within_quiet_hours(now: datetime, span: Tuple[int, int]) -> bool:
    start, end = span
    if start <= end:
        return start <= now.hour < end
    return now.hour >= start or now.hour < end


def extract_matte_from_name(p: Path, default: str) -> str:
    """
    Returns matte tag based on filename convention: <stem>#<matte>.<ext>
    e.g.  forest#shadow.jpg  → matte='shadow'
    """
    stem = p.stem
    if "#" in stem:
        tag = stem.split("#", 1)[1]
        return tag or default
    if "__" in stem:  # alternative delimiter
        tag = stem.split("__", 1)[1]
        return tag or default
    return default


def prepare_image(
    path: Path,
    *,
    resize: bool,
    max_w: int,
    max_h: int,
    quality: int,
) -> Tuple[bytes, str]:
    """Verify, optionally resize, and return (payload, sha1_hex)."""
    if Image is None or not resize:
        data = path.read_bytes()
        return data, sha1_of_bytes(data)

    with Image.open(path) as img:
        img.verify()
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_w, max_h), Image.LANCZOS)

        from io import BytesIO

        buf = BytesIO()
        if path.suffix.lower() == ".png":
            img.save(buf, format="PNG", optimize=True)
        else:
            img.save(buf, format="JPEG", quality=quality, optimize=True)
        payload = buf.getvalue()
    return payload, sha1_of_bytes(payload)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def tv_upload(tv: SamsungTVWS, payload: bytes, ext: str, matte: str) -> str:
    return tv.art().upload(payload, file_type=ext.upper(), matte=matte)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def tv_select(tv: SamsungTVWS, remote_name: str, *, show: bool) -> None:
    tv.art().select_image(remote_name, show=show)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="samsung-art",
        description="Manage Samsung Frame / Art‑mode images.",
    )
    p.add_argument("--tv-ip", default=os.getenv("TV_IP"), help="Samsung TV IP.")
    p.add_argument(
        "--state-file",
        type=Path,
        default=Path(os.getenv("UPLOAD_LIST", "~/.samsung_art_state.json")).expanduser(),
        help="State JSON tracking uploads.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    p.add_argument(
        "--no-redact-ip",
        action="store_true",
        help="Show raw IPs in logs (default: redacted).",
    )

    subs = p.add_subparsers(dest="cmd", required=True)

    # check
    subs.add_parser("check", help="Ping TV & show art‑mode capability.")

    # cleanup
    subs.add_parser("cleanup", help="Delete remote art no longer present locally.")

    # show‑random
    sp = subs.add_parser("show-random", help="Display a random existing upload.")
    sp.add_argument("--seed", type=int)
    sp.add_argument("--show", action="store_true")

    # show <file>
    sp = subs.add_parser("show", help="Show (and upload if needed) specific file(s).")
    sp.add_argument("files", nargs="+", type=Path)
    sp.add_argument("--show", action="store_true")

    # upload
    up = subs.add_parser("upload", help="Upload new art.")
    up.add_argument("--folder", type=Path, default=Path(os.getenv("ART_FOLDER", "./images")))
    up.add_argument("--stdin-paths", action="store_true", help="Read image paths from stdin.")
    up.add_argument("--upload-all", action="store_true")
    up.add_argument("--batch", type=int, metavar="N")
    up.add_argument("--resize", action="store_true")
    up.add_argument("--max-width", type=int, default=3840)
    up.add_argument("--max-height", type=int, default=2160)
    up.add_argument("--quality", type=int, default=90)
    up.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    up.add_argument("--seed", type=int)
    up.add_argument("--prefer-newest", action="store_true")
    up.add_argument("--no-recursive", dest="recursive", action="store_false")
    up.add_argument("--default-matte", default="none", help="Fallback matte type.")
    # scheduling
    up.add_argument("--quiet-hours", metavar="START-END")
    up.add_argument("--max-per-day", type=int)

    return p


def cmd_check(tv: SamsungTVWS) -> None:
    try:
        art_support = tv.art().supported()
        tv.rest_device_info()
        logging.info("TV reachable. Art‑mode %s.", "supported" if art_support else "NOT supported")
    except Exception as exc:  # noqa: BLE001
        logging.error("TV unreachable: %s", exc)
        sys.exit(1)


def cmd_cleanup(tv: SamsungTVWS, state: List[Dict[str, Any]], state_path: Path) -> None:
    """
    Feature 23 – delete remote images that no longer exist locally.
    Uses tv.art().delete(remote_name) if the API is available.
    """
    removed: List[str] = []
    remaining: List[Dict[str, Any]] = []
    for entry in state:
        local_path = Path(entry["file"])
        if local_path.exists():
            remaining.append(entry)
            continue
        remote = entry["remote_filename"]
        try:
            if hasattr(tv.art(), "delete"):
                tv.art().delete(remote)  # type: ignore[attr-defined]
            removed.append(remote)
            logging.info("Deleted orphaned remote image: %s", remote)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to delete %s: %s", remote, exc)
            remaining.append(entry)

    if removed:
        save_state(state_path, remaining)
    else:
        logging.info("Nothing to clean up.")


def cmd_show_random(tv: SamsungTVWS, state: List[Dict[str, Any]], *, seed: int | None, show: bool) -> None:
    if seed is not None:
        random.seed(seed)
    if not state:
        logging.error("State empty – nothing to show.")
        sys.exit(1)
    remote = random.choice(state)["remote_filename"]
    tv_select(tv, remote, show=show)
    logging.info("Displayed random image (%s).", remote)


def cmd_show_files(
    tv: SamsungTVWS,
    state: List[Dict[str, Any]],
    state_path: Path,
    *,
    files: List[Path],
    show_now: bool,
    default_matte: str,
) -> None:
    for file_path in files:
        file_path = file_path.expanduser().resolve()
        if not file_path.is_file():
            logging.error("File not found: %s", file_path)
            continue

        matte = extract_matte_from_name(file_path, default_matte)
        payload, sha_hex = prepare_image(file_path, resize=False, max_w=0, max_h=0, quality=90)
        entry = next((e for e in state if e.get("sha1") == sha_hex), None)
        if entry:
            remote = entry["remote_filename"]
            tv_select(tv, remote, show=show_now)
            logging.info("Selected existing art: %s", file_path.name)
            continue

        ext = "PNG" if file_path.suffix.lower() == ".png" else "JPEG"
        try:
            remote = tv_upload(tv, payload, ext, matte)
            state.append({"file": str(file_path), "sha1": sha_hex, "remote_filename": remote})
            save_state(state_path, state)
            logging.info("Uploaded: %s (matte=%s)", file_path.name, matte)
            tv_select(tv, remote, show=show_now)
        except Exception as exc:  # noqa: BLE001
            logging.error("Upload failed for %s: %s", file_path.name, exc)


def cmd_upload(
    tv: SamsungTVWS,
    args: argparse.Namespace,
    state: List[Dict[str, Any]],
    state_path: Path,
) -> None:
    now = datetime.now()

    # Quiet hours / daily limit (feature 19)
    if args.quiet_hours:
        try:
            start_s, end_s = args.quiet_hours.split("-")
            if within_quiet_hours(now, (int(start_s), int(end_s))):
                logging.info("Within quiet hours – exiting.")
                sys.exit()
        except ValueError:
            logging.error("--quiet-hours requires START-END (e.g. 18-23).")
            sys.exit()

    if args.max_per_day:
        today = date.today()
        todays = [e for e in state if e.get("uploaded_at") == today.isoformat()]
        if len(todays) >= args.max_per_day:
            logging.info("Daily limit reached (%d) – exiting.", args.max_per_day)
            sys.exit()

    # Collect paths
    if args.stdin_paths:
        paths = [Path(line.strip()) for line in sys.stdin if line.strip()]
    else:
        paths = scan_images(args.folder.expanduser().resolve(), recursive=args.recursive)

    if not paths:
        logging.info("No images found.")
        sys.exit()

    if args.seed:
        random.seed(args.seed)

    # Concurrent prep (hash, resize)
    prepared: List[Tuple[Path, bytes, str, str]] = []  # (path, payload, sha1, matte)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(
                prepare_image,
                p,
                resize=args.resize,
                max_w=args.max_width,
                max_h=args.max_height,
                quality=args.quality,
            ): p
            for p in paths
        }
        iterator: Iterable = tqdm(futs, desc="Prep", unit="img") if tqdm else futs
        for f in iterator:
            pth = futs[f]
            try:
                payload, sha_hex = f.result()
                matte = extract_matte_from_name(pth, args.default_matte)
                prepared.append((pth, payload, sha_hex, matte))
            except Exception as exc:  # noqa: BLE001
                logging.warning("Skipping %s (%s).", pth.name, exc)

    known = {e.get("sha1") for e in state}
    new_items = [(p, d, h, m) for p, d, h, m in prepared if h not in known]

    # Selection
    if args.upload_all:
        queue = new_items
    else:
        base_pool = new_items or prepared
        if args.prefer_newest:
            horizon = now - timedelta(days=30)
            recent = [item for item in base_pool if item[0].stat().st_mtime >= horizon.timestamp()]
            base_pool = recent or base_pool
        queue = [random.choice(base_pool)]

    if not queue:
        logging.info("Nothing to upload.")
        sys.exit()

    batch = args.batch or 1
    for chunk_start in range(0, len(queue), batch):
        block = queue[chunk_start : chunk_start + batch]
        if len(block) == 1 and batch == 1:
            _send_single(tv, block[0], state, state_path, show=not args.upload_all, now=now)
        else:
            _send_zip(tv, block, state, state_path, now=now)

    logging.info("Upload complete.")


def _send_single(
    tv: SamsungTVWS,
    item: Tuple[Path, bytes, str, str],
    state: List[Dict[str, Any]],
    state_path: Path,
    *,
    show: bool,
    now: datetime,
) -> None:
    path, payload, sha_hex, matte = item
    ext = "PNG" if path.suffix.lower() == ".png" else "JPEG"
    try:
        remote = tv_upload(tv, payload, ext, matte)
        state.append(
            {
                "file": str(path),
                "sha1": sha_hex,
                "remote_filename": remote,
                "uploaded_at": now.date().isoformat(),
            }
        )
        save_state(state_path, state)
        logging.info("Uploaded: %s (matte=%s)", path.name, matte)
        if show:
            tv_select(tv, remote, show=show)
    except Exception as exc:  # noqa: BLE001
        logging.error("Single upload failed (%s): %s", path.name, exc)


def _send_zip(
    tv: SamsungTVWS,
    items: Sequence[Tuple[Path, bytes, str, str]],
    state: List[Dict[str, Any]],
    state_path: Path,
    *,
    now: datetime,
) -> None:
    from io import BytesIO

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, payload, _, _ in items:
            zf.writestr(path.name, payload)
    zip_bytes = buf.getvalue()

    try:
        remote_names = tv.art().upload_archive(zip_bytes)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        logging.warning("Zip upload failed: %s – falling back.", exc)
        for itm in items:
            _send_single(tv, itm, state, state_path, show=False, now=now)
        return

    for (path, _, sha_hex, matte), remote in zip(items, remote_names, strict=True):
        state.append(
            {
                "file": str(path),
                "sha1": sha_hex,
                "remote_filename": remote,
                "uploaded_at": now.date().isoformat(),
            }
        )
        logging.info("Bulk‑uploaded: %s (matte=%s)", path.name, matte)
    save_state(state_path, state)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level, redact_ip=not args.no_redact_ip)

    if not args.tv_ip:
        logging.error("TV IP required (--tv-ip or TV_IP).")
        sys.exit(1)

    tv = SamsungTVWS(args.tv_ip)
    state_path: Path = args.state_file
    state: List[Dict[str, Any]] = load_state(state_path)

    try:
        cmd = args.cmd
        if cmd == "check":
            cmd_check(tv)
        elif cmd == "cleanup":
            cmd_cleanup(tv, state, state_path)
        elif cmd == "show-random":
            cmd_show_random(tv, state, seed=args.seed, show=args.show)
        elif cmd == "show":
            cmd_show_files(tv, state, state_path, files=args.files, show_now=args.show, default_matte="none")
        elif cmd == "upload":
            cmd_upload(tv, args, state, state_path)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")


if __name__ == "__main__":
    main()
