import os
import cv2
from ultralytics import YOLO

calories = {
    # 국물류 — 국물 절반 정도 섭취 가정
    "라면": 510,        # 면+건더기+국물 일부
    "짬뽕": 700,        # 면+해물/야채+국물 일부
    "김치찌개": 330,     # 건더기+국물 일부 (기름 일부 섭취)
    "된장찌개": 265,     # 건더기+국물 일부

    # 비국물류 — 소스 포함
    "탕수육": 540,       # 튀김 480 + 소스 약 60
    "제육볶음": 600,     # 양념이 본체에 포함
    "햄버거": 590,       # 패티번 550 + 케첩/마요 약 40
    "피자": 700,         # 660 + 디핑소스 약 40
    "스파게티": 670,     # 소스가 본체에 포함
    "비빔밥": 590,       # 560 + 고추장 약 30
    "삼겹살": 560,       # 고기 500 + 쌈장/기름장 약 60
    "돈까스" : 670,      #경양식으로 가정시 600 + 소스 약 70

    # 단독 기준
    "달걀 프라이": 90,    # 소금·후추는 사실상 0kcal
    "소시지": 250,
    "짜장": 700,         # 춘장 소스가 본체에 포함
    "밥": 300,
}

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

    # 검출된 음식의 클래스명과 칼로리를 출력하고 bbox 라벨에도 표시
    result = results[0]
    calorie_labels = {}

    for class_id in result.boxes.cls.int().tolist():
        class_name = result.names[class_id]
        calorie = calories.get(class_name)
        calorie_text = f"{calorie} kcal" if calorie is not None else "칼로리 정보 없음"

        calorie_labels[class_id] = f"{class_name} ({calorie_text})"
        print(f"{image_name} - {class_name}: {calorie_text}")

    result.names = {
        class_id: calorie_labels.get(class_id, class_name)
        for class_id, class_name in result.names.copy().items()
    }

    # bbox 그려진 결과 이미지
    annotated_image = result.plot()

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
