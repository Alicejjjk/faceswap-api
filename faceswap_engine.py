import numpy as np
from PIL import Image
import io
import base64
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo


# 初始化模型（程序启动时加载一次即可）
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 轻量换脸模型（自动下载）
swapper = model_zoo.get_model('inswapper_128.onnx', providers=['CPUExecutionProvider'])


def swap_face(source_bytes, target_bytes):
    """
    真·换脸实现（InsightFace）
    """

    # --- 读取图片 ---
    source_img = Image.open(io.BytesIO(source_bytes)).convert("RGB")
    target_img = Image.open(io.BytesIO(target_bytes)).convert("RGB")

    source_np = np.array(source_img)
    target_np = np.array(target_img)

    # --- 人脸检测 ---
    source_faces = app.get(source_np)
    target_faces = app.get(target_np)

    if len(source_faces) == 0:
        return {"error": "未检测到源图的人脸"}

    if len(target_faces) == 0:
        return {"error": "未检测到目标图的人脸"}

    source_face = source_faces[0]
    target_face = target_faces[0]

    # --- 换脸 ---
    swapped_np = swapper.get(target_np, target_face, source_face)

    # --- 输出 JPEG ---
    output_img = Image.fromarray(swapped_np)
    buf = io.BytesIO()
    output_img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    # --- Base64 ---
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    return {
        "result_base64": img_base64,
        "note": "success"
    }
