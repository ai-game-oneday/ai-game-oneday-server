import os, torch
from pathlib import Path

dll_dir = Path(torch.__file__).parent / "lib"  # …\site-packages\torch\lib
os.add_dll_directory(dll_dir)

import base64
from io import BytesIO
from PIL import Image
from rembg import remove
import onnxruntime as ort


def remove_background(b64_string: str) -> str:
    img_bytes = base64.b64decode(b64_string)
    result_bytes = remove(img_bytes)
    return base64.b64encode(result_bytes).decode("utf-8")
