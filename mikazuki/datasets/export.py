from __future__ import annotations

import io
import queue
import threading
import zipfile
from pathlib import Path
from urllib.parse import quote

from fastapi import HTTPException
from fastapi.responses import FileResponse, StreamingResponse


def resolve_dataset_file(dataset_dir: Path, rel: str) -> Path:
    cleaned = (rel or "").strip().replace("\\", "/")
    if not cleaned:
        raise HTTPException(status_code=400, detail="path is required")
    target = (dataset_dir / cleaned).resolve()
    try:
        target.relative_to(dataset_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="path escapes dataset") from exc
    if not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return target


def file_download_response(dataset_dir: Path, rel: str) -> FileResponse:
    target = resolve_dataset_file(dataset_dir, rel)
    return FileResponse(str(target), filename=target.name)


def _is_visible(path: Path, dataset_dir: Path) -> bool:
    return not any(part.startswith(".") for part in path.relative_to(dataset_dir).parts)


def iter_exportable_files(dataset_dir: Path):
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file() and _is_visible(path, dataset_dir):
            yield path


class _QueueWriter(io.RawIOBase):
    def __init__(self):
        self.chunks: queue.Queue[bytes | None] = queue.Queue()

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:
        self.chunks.put(bytes(data))
        return len(data)


def stream_dataset_zip(dataset_dir: Path) -> StreamingResponse:
    writer = _QueueWriter()

    def run() -> None:
        try:
            with zipfile.ZipFile(writer, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in iter_exportable_files(dataset_dir):
                    archive.write(path, path.relative_to(dataset_dir).as_posix())
        finally:
            writer.chunks.put(None)

    threading.Thread(target=run, daemon=True).start()

    def generate():
        while True:
            chunk = writer.chunks.get()
            if chunk is None:
                break
            yield chunk

    filename = quote(f"{dataset_dir.name}.zip")
    return StreamingResponse(
        generate(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename*=utf-8''{filename}"},
    )
