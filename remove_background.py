import os
import torch
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
from rembg import remove
import onnxruntime as ort

# Windows에서만 DLL 디렉토리 추가 (Linux에서는 무시)
if os.name == "nt":  # Windows인 경우에만
    dll_dir = Path(torch.__file__).parent / "lib"
    os.add_dll_directory(dll_dir)
else:
    print(
        "[remove_background] Linux environment detected - skipping DLL directory setup"
    )


def remove_background(b64_string: str) -> str:
    """
    Base64 인코딩된 이미지의 배경을 제거하고 다시 Base64로 반환

    Args:
        b64_string: Base64 인코딩된 이미지 문자열

    Returns:
        배경이 제거된 Base64 인코딩된 이미지 문자열
    """
    try:
        img_bytes = base64.b64decode(b64_string)
        result_bytes = remove(img_bytes)
        return base64.b64encode(result_bytes).decode("utf-8")
    except Exception as e:
        print(f"[remove_background] Error: {e}")
        # 에러 발생시 원본 이미지 반환
        return b64_string
