from fastapi import FastAPI, UploadFile, File
from faceswap_engine import swap_face

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok", "message": "FaceSwap API running"}

@app.post("/swap")
async def swap(
    source_image: UploadFile = File(...),
    target_image: UploadFile = File(...)
):
    source_bytes = await source_image.read()
    target_bytes = await target_image.read()

    result = swap_face(source_bytes, target_bytes)
    return result
