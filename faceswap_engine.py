import numpy as np
from PIL import Image
import io
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo


# 初始化人脸检测器 + 换脸器（只在启动时加载一次）
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

swapper = model_zoo.get_model('inswapper_128.onnx', providers=['CPUExecutionProvider'])


def swap_face(source_bytes, target_bytes):
    """
    真实换脸逻辑
    """

    # --- 读取图片 ---
    source_img = Image.open(io.BytesIO(source_bytes)).convert("RGB")
    target_img = Image.open(io.BytesIO(target_bytes)).convert("RGB")

    source_np = np.array(source_img)
    target_np = np.array(target_img)

    # --- 检测人脸 ---
    source_faces = app.get(source_np)
    target_faces = app.get(target_np)

    if len(source_faces) == 0:
        return {"error": "未检测到 source 图像的人脸"}

    if len(target_faces) == 0:
        return {"error": "未检测到 target 图像的人脸"}

    source_face = source_faces[0]
    target_face = target_faces[0]

    # --- 人脸替换 ---
    swapped = swapper.get(target_np, target_face, source_face)

    # --- 输出 JPEG ---
    output = Image.fromarray(swapped)
    buf = io.BytesIO()
    output.save(buf, format="JPEG")

    return {
        "result_base64": buf.getvalue().hex(),  # 或改 base64
        "note": "success"
    }
