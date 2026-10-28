import base64
from pathlib import Path
from typing import Dict


DATA_DIR = Path("data")
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def save_uploaded_document(
    filename: str,
    content_base64: str,
    domain: str,
    source_type: str,
) -> Dict:
    safe_name = Path(filename).name
    extension = Path(safe_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file type. Use .md, .txt, or .pdf.")

    try:
        content = base64.b64decode(content_base64)
    except Exception as error:
        raise ValueError("Invalid base64 document content.") from error

    safe_domain = sanitize_metadata_value(domain)
    safe_source_type = sanitize_metadata_value(source_type)
    upload_dir = DATA_DIR / safe_domain / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / safe_name
    destination.write_bytes(content)

    return {
        "filename": safe_name,
        "path": str(destination),
        "bytes_written": len(content),
        "extension": extension,
        "domain": safe_domain,
        "source_type": safe_source_type,
    }


def sanitize_metadata_value(value: str) -> str:
    cleaned = value.strip().lower().replace(" ", "_").replace("-", "_")
    return "".join(character for character in cleaned if character.isalnum() or character == "_") or "general"
