import os
import uuid
import io
from typing import Tuple
from PIL import Image
from fastapi import UploadFile, HTTPException, status

from backend.app.config import settings


class ImageService:
    def __init__(self, upload_dir: str = settings.UPLOAD_DIR):
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)

    async def validate_and_save(self, file: UploadFile) -> Tuple[Image.Image, str, bytes]:
        """
        Validates content type, file size and image integrity.
        Saves the image to upload_dir and returns (PIL Image, saved_path, raw_bytes).
        """
        # 1. Content-Type check
        if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "INVALID_IMAGE_FORMAT",
                    "message": f"Only JPEG and PNG images are supported. Received: {file.content_type}"
                }
            )

        # 2. Read bytes and check file size
        raw_bytes = await file.read()
        if len(raw_bytes) > settings.MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "code": "FILE_TOO_LARGE",
                    "message": f"File size exceeds maximum limit of {settings.MAX_FILE_SIZE_BYTES // (1024*1024)} MB."
                }
            )

        if len(raw_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "CORRUPTED_IMAGE",
                    "message": "Uploaded file is empty (0 bytes)."
                }
            )

        # 3. Image integrity check
        try:
            # First pass: verify integrity
            with Image.open(io.BytesIO(raw_bytes)) as test_img:
                test_img.verify()
            
            # Second pass: open for actual use and convert to RGB
            img = Image.open(io.BytesIO(raw_bytes))
            img = img.convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "CORRUPTED_IMAGE",
                    "message": f"Could not decode image: {str(e)}"
                }
            )

        # 4. Save to disk
        ext = ".png" if file.content_type == "image/png" else ".jpg"
        unique_name = f"{uuid.uuid4()}{ext}"
        save_path = os.path.join(self.upload_dir, unique_name)

        try:
            with open(save_path, "wb") as f:
                f.write(raw_bytes)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "code": "STORAGE_ERROR",
                    "message": f"Failed to save image to disk: {str(e)}"
                }
            )

        return img, save_path, raw_bytes
