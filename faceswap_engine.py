import cv2
import numpy as np
import dlib

# 加载Dlib的人脸检测器和面部关键点预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def face_swap(source_bytes, target_bytes):
    # 将图片从字节转换为 OpenCV 图像
    source_np = np.frombuffer(source_bytes, np.uint8)
    target_np = np.frombuffer(target_bytes, np.uint8)
    source_image = cv2.imdecode(source_np, cv2.IMREAD_COLOR)
    target_image = cv2.imdecode(target_np, cv2.IMREAD_COLOR)

    # 转换为灰度图
    source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces_source = detector(source_gray)
    faces_target = detector(target_gray)

    if len(faces_source) > 0 and len(faces_target) > 0:
        # 提取人脸区域并交换（此为简化版，不进行精确对齐）
        x1, y1, w1, h1 = faces_source[0].left(), faces_source[0].top(), faces_source[0].width(), faces_source[0].height()
        x2, y2, w2, h2 = faces_target[0].left(), faces_target[0].top(), faces_target[0].width(), faces_target[0].height()
        
        source_face = source_image[y1:y1+h1, x1:x1+w1]
        target_image[y2:y2+h2, x2:x2+w2] = cv2.resize(source_face, (w2, h2))

        # 保存换脸后的图片
        _, swapped_image = cv2.imencode('.jpg', target_image)
        return swapped_image.tobytes()

    return {"error": "No faces detected in one or both images"}
