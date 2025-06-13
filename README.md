# Samsung Art-Mode Uploader

Manage image uploads for Samsung Frame TVs. This tool lets you upload, select and manage artwork in Art Mode straight from the command line.

## Features

- **Upload images** individually or in batches (including zipped archives).
- **Show artwork** directly on the TV or upload without showing.
- **Random display** of a previously uploaded image.
- **Cleanup** orphaned remote images that no longer exist locally.
- **Resizing and quality control** when preparing images.
- **State tracking** to avoid duplicate uploads and keep history.
- **Quiet hours** and **daily limits** to control when uploads occur.
- **Matte tagging** by filename (e.g. `forest#shadow.jpg`).
- **Concurrent processing** with configurable worker threads.

## Requirements

- Python 3.10 or newer
- A Samsung Frame TV on the same network
- Packages listed in `requirements.txt`

Install dependencies with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Invoke the tool with `python StartArt.py` or install it as an executable script. The `--tv-ip` argument (or `TV_IP` environment variable) must point to your TV’s IP address.

```bash
python StartArt.py --tv-ip 192.168.0.9 <command> [options]
```

### Commands

- `check` – verify connectivity and Art Mode support.
- `upload` – upload new images.
- `show` – show one or more files (uploads if needed).
- `show-random` – display a random uploaded image.
- `cleanup` – delete remote art that is no longer present locally.

Run `python StartArt.py <command> --help` for all available options.

### Common options

- `--state-file` – path to the JSON file tracking uploads (default: `~/.samsung_art_state.json` or `$UPLOAD_LIST`).
- `--log-level` – set logging verbosity (`DEBUG`, `INFO`, etc.).
- `--no-redact-ip` – show IP addresses in logs.

#### Upload options

- `--folder` – directory containing images (default: `./images` or `$ART_FOLDER`).
- `--stdin-paths` – read image paths from standard input.
- `--upload-all` – send all new images; otherwise one is chosen at random.
- `--batch N` – upload files in groups of `N` using a zip archive when possible.
- `--resize` – resize images to `--max-width`/`--max-height` with quality `--quality`.
- `--workers N` – number of threads for image preparation.
- `--seed` – random seed for deterministic selection.
- `--prefer-newest` – pick from files modified in the last 30 days if available.
- `--no-recursive` – disable recursive directory scan.
- `--default-matte` – fallback matte style when filename has no matte tag.
- `--quiet-hours START-END` – skip uploads during these hours (24‑h clock).
- `--max-per-day N` – limit number of uploads per day.

### Examples

Upload one random image from `~/Pictures/FrameArt`:

```bash
python StartArt.py --tv-ip 192.168.0.9 upload --folder ~/Pictures/FrameArt
```

Upload all PNG/JPEG files recursively, resizing if needed:

```bash
python StartArt.py --tv-ip 192.168.0.9 upload --folder ./images \
  --upload-all --resize --max-width 3840 --max-height 2160
```

Display a specific local file (uploads if missing):

```bash
python StartArt.py --tv-ip 192.168.0.9 show path/to/image.jpg --show
```

Delete orphaned remote art:

```bash
python StartArt.py --tv-ip 192.168.0.9 cleanup
```

## Environment Variables

- `TV_IP` – default TV IP address.
- `UPLOAD_LIST` – path to the state JSON file.
- `ART_FOLDER` – default folder for uploads.

## State File

The state file stores metadata for every uploaded image:

```json
[
  {
    "file": "/path/to/image.jpg",
    "sha1": "...",
    "remote_filename": "uuid",
    "uploaded_at": "YYYY-MM-DD"
  }
]
```

Keeping this file allows the script to avoid re‑uploading existing images and to clean up remote art when files are removed locally.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
