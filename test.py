import os
import cv2
from ultralytics import YOLO

# ==============================
# 1. 경로 설정
# ==============================

# 현재 프로젝트 경로
BASE_DIR = os.getcwd()

# 학습된 모델 경로
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# 테스트 이미지 폴더
IMAGE_DIR = os.path.join(BASE_DIR, "test_images")

# 결과 저장 폴더
OUTPUT_DIR = os.path.join(BASE_DIR, "result_images")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# 2. 모델 로드
# ==============================

model = YOLO(MODEL_PATH)

# ==============================
# 3. 이미지 추론
# ==============================

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"총 {len(image_files)}개 이미지 추론 시작")

for image_name in image_files:

    image_path = os.path.join(IMAGE_DIR, image_name)

    # 이미지 읽기
    image = cv2.imread(image_path)

    if image is None:
        print(f"이미지 로드 실패: {image_name}")
        continue

    # ==============================
    # YOLO 추론
    # ==============================

    results = model(image)

    # bbox 그려진 결과 이미지
    annotated_image = results[0].plot()

    # ==============================
    # 결과 저장
    # ==============================

    output_path = os.path.join(
        OUTPUT_DIR,
        image_name
    )

    cv2.imwrite(output_path, annotated_image)

    print(f"저장 완료: {output_path}")

print("모든 이미지 추론 완료!")