from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from faceswap_engine import swap_face

app = FastAPI()

@app.get("/")
def root():
    return {"status": "running", "message": "FaceSwap API is OK"}

@app.post("/swap")
async def swap_api(
    source: UploadFile = File(...),
    target: UploadFile = File(...)
):
    try:
        source_bytes = await source.read()
        target_bytes = await target.read()

        result = swap_face(source_bytes, target_bytes)

        return JSONResponse({
            "success": True,
            "message": "Face swap completed (placeholder)",
            "result_info": result
        })

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})