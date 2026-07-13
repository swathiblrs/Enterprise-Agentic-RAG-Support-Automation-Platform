import base64
from pathlib import Path
from typing import Dict


DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def save_uploaded_document(filename: str, content_base64: str) -> Dict:
    safe_name = Path(filename).name
    extension = Path(safe_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file type. Use .md, .txt, or .pdf.")

    try:
        content = base64.b64decode(content_base64)
    except Exception as error:
        raise ValueError("Invalid base64 document content.") from error

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / safe_name
    destination.write_bytes(content)

    return {
        "filename": safe_name,
        "path": str(destination),
        "bytes_written": len(content),
        "extension": extension,
    }
