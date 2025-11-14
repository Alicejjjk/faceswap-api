from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faceswap_engine import face_swap

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "FaceSwap API Running"}

@app.post("/swap")
async def swap(source_image: UploadFile = File(...), target_image: UploadFile = File(...)):
    # 读取图片数据
    source_bytes = await source_image.read()
    target_bytes = await target_image.read()

    result = face_swap(source_bytes, target_bytes)

    if isinstance(result, dict) and "error" in result:
        return result

    return {"status": "ok", "note": "Faces swapped successfully", "swapped_image": result}
