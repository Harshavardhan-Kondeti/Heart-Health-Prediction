import io
from typing import Optional
import numpy as np
import pandas as pd
from fastapi import UploadFile
from PIL import Image


async def load_ecg_from_upload(file: UploadFile) -> np.ndarray:
    filename = (file.filename or "").lower()
    data = await file.read()
    return await load_ecg_from_bytes(data, filename)


async def load_ecg_from_bytes(data: bytes, filename: str) -> np.ndarray:
    filename = (filename or "").lower()
    if filename.endswith(".csv"):
        buf = io.BytesIO(data)
        df = pd.read_csv(buf)
        if df.shape[1] > 1:
            arr = df.iloc[:, 0].to_numpy()
        else:
            arr = df.iloc[:, 0].to_numpy()
        return np.asarray(arr, dtype=np.float32)

    if filename.endswith(".npy"):
        buf = io.BytesIO(data)
        arr = np.load(buf, allow_pickle=False)
        if arr.ndim == 2 and arr.shape[1] > 1:
            arr = arr[:, 0]
        return np.asarray(arr, dtype=np.float32)

    # Try generic text load as fallback
    try:
        buf = io.BytesIO(data)
        arr = np.loadtxt(buf, dtype=np.float32, delimiter=",")
        if arr.ndim == 2 and arr.shape[1] > 1:
            arr = arr[:, 0]
        return np.asarray(arr, dtype=np.float32)
    except Exception:
        return np.asarray([], dtype=np.float32)


async def load_image_from_upload(file: UploadFile, image_size: Optional[tuple[int, int]] = None) -> np.ndarray:
    filename = (file.filename or "").lower()
    data = await file.read()
    return await load_image_from_bytes(data, filename, image_size=image_size)


async def load_image_from_bytes(data: bytes, filename: str, image_size: Optional[tuple[int, int]] = None) -> np.ndarray:
    filename = (filename or "").lower()
    if not (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
        return np.asarray([], dtype=np.float32)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if image_size is not None:
        img = img.resize(image_size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    return arr


async def load_input_from_upload(file: UploadFile, image_size: Optional[tuple[int, int]] = None) -> np.ndarray:
    name = (file.filename or "").lower()
    if any(name.endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
        return await load_image_from_upload(file, image_size=image_size)
    return await load_ecg_from_upload(file)


async def load_input_from_bytes(data: bytes, filename: str, image_size: Optional[tuple[int, int]] = None) -> np.ndarray:
    name = (filename or "").lower()
    if any(name.endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
        return await load_image_from_bytes(data, name, image_size=image_size)
    return await load_ecg_from_bytes(data, name)
