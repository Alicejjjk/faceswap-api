from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import uuid
import os
from faceswap_engine import do_faceswap

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok", "message": "FaceSwap API is running"}

@app.post("/swap")
async def swap_faces(
    source_image: UploadFile = File(...),
    target_image: UploadFile = File(...)
):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    source_path = f"{temp_dir}/{uuid.uuid4()}_source.jpg"
    target_path = f"{temp_dir}/{uuid.uuid4()}_target.jpg"
    output_path = f"{temp_dir}/{uuid.uuid4()}_output.jpg"

    with open(source_path, "wb") as f:
        f.write(await source_image.read())

    with open(target_path, "wb") as f:
        f.write(await target_image.read())

    # 调用换脸处理函数
    do_faceswap(source_path, target_path, output_path)

    return FileResponse(output_path, media_type="image/jpeg")
