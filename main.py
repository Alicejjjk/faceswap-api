from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faceswap_engine import swap_face

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
    source_bytes = await source_image.read()
    target_bytes = await target_image.read()

    result = swap_face(source_bytes, target_bytes)

    return result
