from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse  # 导入 JSONResponse
from faceswap_engine import face_swap
import base64

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

    # 调用换脸功能
    result = face_swap(source_bytes, target_bytes)

    # 如果返回结果包含错误信息
    if isinstance(result, dict) and "error" in result:
        return JSONResponse(content=result, status_code=400)

    # 将返回的图像转换为 base64 编码
    if isinstance(result, bytes):  # 确保返回的结果是字节流（即图片）
        result_base64 = base64.b64encode(result).decode('utf-8')
        return JSONResponse(content={
            "status": "ok",
            "note": "Faces swapped successfully",
            "result_base64": result_base64
        })
    else:
        return JSONResponse(content={"error": "Invalid result format"}, status_code=500)
