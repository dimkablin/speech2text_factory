"""Save image module."""
from fastapi import UploadFile
from pathlib import Path
import os
import shutil

def save_upload_file(upload_file: UploadFile) -> list:
    try:
        extension = upload_file.filename[-4:]
        file_path = Path("./src/ai_models/ocr_models/buff/" + \
            f"{str(hash(upload_file))[:10]}" + extension)
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return [os.fspath(file_path)]
    finally:
        upload_file.file.close()
