import numpy as np
from PIL import Image
import io
import base64
import cv2
from facefusion import FaceFusion

# 初始化轻量级人脸换脸模型（facefusion）
model = FaceFusion()

def swap_face(source_bytes, target_bytes):
    try:
        # 读取图片
        source_img = Image.open(io.BytesIO(source_bytes)).convert("RGB")
        target_img = Image.open(io.BytesIO(target_bytes)).convert("RGB")

        # 转为 numpy 数组
        source_np = np.array(source_img)
        target_np = np.array(target_img)

        # 进行人脸替换
        result_img = model.swap(source_np, target_np)

        # 输出为图片格式
        output_img = Image.fromarray(result_img)
        buf = io.BytesIO()
        output_img.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        # 将图片转换为 Base64 格式
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        return {
            "result_base64": img_base64,
            "note": "success"
        }

    except Exception as e:
        return {"error": str(e)}
