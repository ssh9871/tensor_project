# ==============================
# 2. 라이브러리 import
# ==============================

import os
import shutil
import random
from PIL import Image

# ==============================
# 3. UECFOOD 경로 설정
# ==============================

# 현재 실행중인 파일 기준 경로
BASE_DIR = os.getcwd()

# UECFOOD100 폴더
UEC_ROOT = os.path.join(BASE_DIR, "UECFOOD100")

# 생성될 YOLO 데이터셋 폴더
YOLO_ROOT = os.path.join(BASE_DIR, "food_dataset")

# ==============================
# 4. 사용할 클래스 정의
# ==============================

selected_classes = {
    1: "밥",
    11: "비빔밥",
    17: "햄버거",
    18: "피자",
    27: "스파게티",
    38: "소시지",
    56: "돈까스",
    68: "달걀 프라이"
}

# YOLO용 새로운 클래스 번호 매핑
class_mapping = {
    1: 0,
    11: 1,
    17: 2,
    18: 3,
    27: 4,
    38: 5,
    56: 6,
    68: 7
}

# ==============================
# 5. 폴더 생성
# ==============================

for split in ["train", "valid"]:

    os.makedirs(
        os.path.join(YOLO_ROOT, split, "images"),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(YOLO_ROOT, split, "labels"),
        exist_ok=True
    )

# ==============================
# 6. bbox → YOLO format 변환 함수
# ==============================

def convert_bbox(size, box):

    w, h = size

    xmin, ymin, xmax, ymax = box

    x_center = ((xmin + xmax) / 2) / w
    y_center = ((ymin + ymax) / 2) / h

    bbox_width = (xmax - xmin) / w
    bbox_height = (ymax - ymin) / h

    return (
        x_center,
        y_center,
        bbox_width,
        bbox_height
    )

# ==============================
# 7. 데이터 변환
# ==============================

for original_class_id, class_name in selected_classes.items():

    folder_path = os.path.join(
        UEC_ROOT,
        str(original_class_id)
    )

    bb_info_path = os.path.join(
        folder_path,
        "bb_info.txt"
    )

    if not os.path.exists(bb_info_path):

        print(f"bb_info.txt 없음: {bb_info_path}")
        continue

    with open(bb_info_path, "r") as f:
        lines = f.readlines()

    # 첫 줄(header) 제거
    data_lines = lines[1:]

    random.shuffle(data_lines)

    split_index = int(len(data_lines) * 0.8)

    train_lines = data_lines[:split_index]
    valid_lines = data_lines[split_index:]

    for split_name, split_lines in [
        ("train", train_lines),
        ("valid", valid_lines)
    ]:

        for line in split_lines:

            parts = line.strip().split()

            if len(parts) < 5:
                continue

            image_name = parts[0]

            # jpg 확장자 자동 추가
            if not image_name.endswith(".jpg"):
                image_name += ".jpg"

            xmin = int(parts[1])
            ymin = int(parts[2])
            xmax = int(parts[3])
            ymax = int(parts[4])

            image_path = os.path.join(
                folder_path,
                image_name
            )

            if not os.path.exists(image_path):

                print(f"이미지 없음: {image_path}")
                continue

            # 이미지 열기
            img = Image.open(image_path)

            w, h = img.size

            # bbox 변환
            bbox = convert_bbox(
                (w, h),
                (xmin, ymin, xmax, ymax)
            )

            # 새 클래스 번호
            yolo_class_id = class_mapping[
                original_class_id
            ]

            # 새 파일 이름
            new_name = f"{class_name}_{image_name}"

            # 이미지 저장 경로
            dst_image_path = os.path.join(
                YOLO_ROOT,
                split_name,
                "images",
                new_name
            )

            # 이미지 복사
            shutil.copy(image_path, dst_image_path)

            # label 저장 경로
            label_path = os.path.join(
                YOLO_ROOT,
                split_name,
                "labels",
                new_name.replace(".jpg", ".txt")
            )

            # label txt 저장
            with open(label_path, "w") as label_file:

                label_file.write(
                    f"{yolo_class_id} "
                    f"{bbox[0]} "
                    f"{bbox[1]} "
                    f"{bbox[2]} "
                    f"{bbox[3]}"
                )

print("YOLO 데이터셋 생성 완료!")

# ==============================
# 8. data.yaml 생성
# ==============================

yaml_text = f"""
train: {os.path.join(YOLO_ROOT, 'train', 'images')}
val: {os.path.join(YOLO_ROOT, 'valid', 'images')}

nc: 8

names:
  0: 밥
  1: 비빔밥
  2: 햄버거
  3: 피자
  4: 스파게티
  5: 소시지
  6: 돈까스
  7: 달걀 프라이
"""

yaml_path = os.path.join(YOLO_ROOT, "data.yaml")

with open(yaml_path, "w") as f:
    f.write(yaml_text)

print("data.yaml 생성 완료!")

# ==============================
# 9. YOLOv8 학습
# ==============================

from ultralytics import YOLO
from multiprocessing import freeze_support

if __name__ == "__main__":

    freeze_support()

    model = YOLO("yolov8n.pt")

    model.train(
        data=yaml_path,
        epochs=30,
        imgsz=640,
        batch=16,
        workers=0
    )

    print("학습 완료!")