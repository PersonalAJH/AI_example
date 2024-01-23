import cv2
import numpy as np

# 모델과 설정 파일 경로
model = './pose_iter_160000.caffemodel'
config = './pose_deploy_linevec_faster_4_stages.prototxt'

# 네트워크 로드
net = cv2.dnn.readNetFromCaffe(config, model)

# 이미지 로드
image = cv2.imread('./img.png')
(h, w) = image.shape[:2]

# 입력 이미지 전처리
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

# 네트워크에 blob 입력
net.setInput(blob)

# 감지 결과 받기
detections = net.forward()

# 감지된 객체 순회
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.2:  # confidence threshold
        idx = int(detections[0, 0, i, 1])
        
        # 사람 클래스의 인덱스는 일반적으로 15입니다 (COCO dataset 기준)
        if idx == 15:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # 바운딩 박스 그리기
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# 결과 이미지 표시
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
