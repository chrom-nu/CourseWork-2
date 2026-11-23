import os
import sys
import io
import pytest
from PIL import Image

# Ensure workspace root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Set test environment variables
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_api.db"
os.environ["SYNC_DATABASE_URL"] = "sqlite:///./test_api.db"
os.environ["UPLOAD_DIR"] = "backend/test_uploads"
os.environ["DEVICE"] = "cpu"

from fastapi.testclient import TestClient
from backend.app.main import app, model_service


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    # Ensure stub model is allowed if best_model.pt is not present (for CI)
    if not model_service.is_loaded:
        model_service.load_model(allow_stub=True)
    yield
    # Cleanup test db and uploads
    for f in ["test_api.db", "test.db"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (256, 256), color=(60, 140, 70))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def valid_png_bytes() -> bytes:
    img = Image.new("RGB", (256, 256), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def corrupted_image_bytes() -> bytes:
    return b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00corrupted_garbage_bytes_123456789"


@pytest.fixture
def text_file_bytes() -> bytes:
    return b"Hello world, this is a plain text file, not an image."
