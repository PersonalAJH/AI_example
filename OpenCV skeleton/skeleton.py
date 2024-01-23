import cv2
import numpy as np
import matplotlib.pyplot as plt

# 모델과 설정 파일 경로
protoFile = "./pose_deploy_linevec.prototxt"
weightsFile = "./pose_iter_160000.caffemodel"


BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]


# 네트워크 로드
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# 이미지 로드
image = cv2.imread('./img.png')
height, width = image.shape[:2]

# 입력 이미지 전처리
# 이미지 전처리에서는 여러가지가 있는데
# 1. 정규화(Normalize) : 이미지를 0~1 사이로 스케일링합니다. 1.0/255 로 나누어 정규화 한다.
# 2. 크기조정(Resizing) : 사진의 스케일을 DNN 모델에 맞는 크기로 재조정
# 3. 평균 뺴기(Mean Substraction) : 평균 뺴기를 위한 값. 이는 픽셀 값에서 특정 평균값을 뺴서 데이터를 중심화
# 4. 채널교환(Channel Swapping) : swapRB=False는 Red와 Blue 채널을 바꾸지 않겠다는 것을 의미합니다. 일부 이미지는 Red와 Blue 채널이 반대로 되어 있을 수 있다.
# 5. 자르기 (Cropping) : crop=False는 이미지를 자르지 않겠다는 것을 의미합니다.
inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (width, height), (0, 0, 0), swapRB=False, crop=False)

# 네트워크에 blob 입력
net.setInput(inpBlob)

# 감지 결과 받기
output = net.forward()

H = output.shape[2]
W = output.shape[3]

# 포인트를 저장할 빈 리스트
points = []

for i in range(18):  # 18개의 키포인트
    # 키포인트의 신뢰도 맵
    probMap = output[0, i, :, :]

    # 최대값 위치 찾기
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    
    # 원래 이미지의 크기에 맞게 조정
    x = (width * point[0]) / W
    y = (height * point[1]) / H

    if prob > 0.1:  # 신뢰도 임계값
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        points.append((int(x), int(y)))
    else:
        points.append(None)

# 스켈레톤을 형성하기 위한 키포인트 쌍
# POSE_PAIRS = [
#     [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
#     [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
#     [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]
# ]

# 스켈레톤 그리기
# for pair in POSE_PAIRS:
#     partA = pair[0]
#     partB = pair[1]

#     if points[partA] and points[partB]:
#         cv2.line(image, points[partA], points[partB], (0, 255, 0), 3)

for pair in POSE_PAIRS:
    partA = pair[0]             # Head
    partA = BODY_PARTS[partA]   # 0
    partB = pair[1]             # Neck
    partB = BODY_PARTS[partB]   # 1
    
    #print(partA," 와 ", partB, " 연결\n")
    if points[partA] and points[partB]:
        cv2.line(image, points[partA], points[partB], (0, 255, 0), 2)


image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 이미지 시각화
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.axis("off")
plt.show()
