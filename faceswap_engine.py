import cv2
import numpy as np

# 使用 OpenCV 的 haarcascade_frontalface_default.xml 来替代 dlib
def face_swap(source_bytes, target_bytes):
    # 读取图片
    source_np = np.frombuffer(source_bytes, np.uint8)
    target_np = np.frombuffer(target_bytes, np.uint8)
    source_image = cv2.imdecode(source_np, cv2.IMREAD_COLOR)
    target_image = cv2.imdecode(target_np, cv2.IMREAD_COLOR)

    # 转为灰度图
    source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    # 使用 OpenCV 的 Haar 级联分类器检测人脸
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces_source = face_cascade.detectMultiScale(source_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces_target = face_cascade.detectMultiScale(target_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 如果检测到人脸
    if len(faces_source) > 0 and len(faces_target) > 0:
        x1, y1, w1, h1 = faces_source[0]
        x2, y2, w2, h2 = faces_target[0]

        source_face = source_image[y1:y1+h1, x1:x1+w1]
        target_image[y2:y2+h2, x2:x2+w2] = cv2.resize(source_face, (w2, h2))

        # 保存换脸后的图片
        _, swapped_image = cv2.imencode('.jpg', target_image)
        return swapped_image.tobytes()

    return {"error": "No faces detected in one or both images"}
