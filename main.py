from fastapi import FastAPI, File, UploadFile
from faceswap_engine import swap_face

app = FastAPI()

@app.post("/swap")
async def swap(source: UploadFile = File(...), target: UploadFile = File(...)):
    source_bytes = await source.read()
    target_bytes = await target.read()

    result = swap_face(source_bytes, target_bytes)
    return result
