import io
from PIL import Image
import numpy as np
import base64
import cv2
from huggingface_hub import hf_hub_download

# 使用轻量模型：sface（4MB）
# 免费 Render 可运行
MODEL_REPO = "serengil/deepface"
MODEL_FILE = "SFace/SFace.pb"

import tensorflow as tf

# 下载轻量级模型（仅一次）
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
model = tf.saved_model.load(model_path)


def read_image(bytes_data):
    return Image.open(io.BytesIO(bytes_data)).convert("RGB")


def img_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def align_face(image):
    """轻量级人脸对齐（不依赖 heavy 模型）"""
    import face_recognition

    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        return None, None

    top, right, bottom, left = face_locations[0]
    face_img = image[top:bottom, left:right]
    return face_img, (top, right, bottom, left)


def swap_face(source_bytes, target_bytes):
    try:
        # 转为 numpy
        src = np.array(read_image(source_bytes))
        tgt = np.array(read_image(target_bytes))

        # 检测 + 裁剪
        src_face, _ = align_face(src)
        tgt_face, tgt_box = align_face(tgt)

        if src_face is None or tgt_face is None:
            return {"error": "无法检测到人脸"}

        # resize，人脸融合
        src_face = cv2.resize(src_face, (tgt_face.shape[1], tgt_face.shape[0]))

        # α 融合（轻量版，不依赖沉重模型）
        blended = cv2.addWeighted(src_face, 0.9, tgt_face, 0.1, 0)

        # 把 blended 放回 target
        top, right, bottom, left = tgt_box
        tgt[top:bottom, left:right] = blended

        out_img = Image.fromarray(tgt)

        return {
            "result_base64": img_to_base64(out_img),
            "status": "success"
        }

    except Exception as e:
        return {"error": str(e)}
