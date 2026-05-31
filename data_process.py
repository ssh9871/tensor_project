# ==============================
# 1. 라이브러리 import
# ==============================

import os
import shutil
import random
from PIL import Image

# ==============================
# 2. UECFOOD 경로 설정
# ==============================

# 현재 실행중인 파일 기준 경로
BASE_DIR = os.getcwd()

# UECFOOD100 폴더
UEC_ROOT = os.path.join(BASE_DIR, "UECFOOD100")

# 한국 음식 이미지 폴더
KOREAN_ROOT = os.path.join(BASE_DIR, "korean_food_images")

# 생성될 YOLO 데이터셋 폴더
YOLO_ROOT = os.path.join(BASE_DIR, "food_dataset")

# 직접 bbox를 지정한 음식 이미지 폴더
SELF_BBOX_ROOT = os.path.join(BASE_DIR, "self_bbox")

# ==============================
# 3. 사용할 클래스 정의
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

# korean_food_images의 하위 폴더명을 YOLO 클래스에 추가
korean_class_names = []

if os.path.exists(KOREAN_ROOT):
    korean_class_names = sorted([
        folder_name
        for folder_name in os.listdir(KOREAN_ROOT)
        if os.path.isdir(os.path.join(KOREAN_ROOT, folder_name))
    ])
else:
    print(f"korean_food_images 폴더 없음: {KOREAN_ROOT}")

korean_class_mapping = {
    class_name: len(selected_classes) + index
    for index, class_name in enumerate(korean_class_names)
}

self_bbox_classes = {
    "blackbean_noodle": "짜장",
    "stired_pork": "제육볶음"
}

self_bbox_class_mapping = {
    folder_name: len(selected_classes) + len(korean_class_names) + index
    for index, folder_name in enumerate(self_bbox_classes)
}

all_class_names = (
    list(selected_classes.values())
    + korean_class_names
    + list(self_bbox_classes.values())
)

# ==============================
# 4. 폴더 생성
# ==============================

for split in ["train", "valid"]:

    for folder_name in ["images", "labels"]:

        folder_path = os.path.join(
            YOLO_ROOT,
            split,
            folder_name
        )

        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

        os.makedirs(
            folder_path,
            exist_ok=True
        )

# ==============================
# 5. bbox → YOLO format 변환 함수
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


def parse_crop_area_line(line):

    line = line.strip()

    if not line or "=" not in line:
        return None

    image_stem, crop_value = line.split("=", 1)
    crop_tokens = crop_value.strip().split()

    if not image_stem or not crop_tokens:
        return None

    coords = crop_tokens[0].split(",")

    if len(coords) != 4:
        return None

    try:
        xmin, ymin, xmax, ymax = [
            int(coord)
            for coord in coords
        ]
    except ValueError:
        return None

    return image_stem, (xmin, ymin, xmax, ymax)


def build_image_lookup(folder_path):

    image_lookup = {}
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    for file_name in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file_name)

        if not os.path.isfile(file_path):
            continue

        stem, ext = os.path.splitext(file_name)

        if ext.lower() in image_extensions:
            image_lookup[stem] = file_name

    return image_lookup


def clamp_bbox(box, size):

    w, h = size
    xmin, ymin, xmax, ymax = box

    xmin = max(0, min(xmin, w))
    xmax = max(0, min(xmax, w))
    ymin = max(0, min(ymin, h))
    ymax = max(0, min(ymax, h))

    if xmax <= xmin or ymax <= ymin:
        return None

    return xmin, ymin, xmax, ymax

# ==============================
# 6. 데이터 변환
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
                os.path.splitext(new_name)[0] + ".txt"
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

for class_name in korean_class_names:

    folder_path = os.path.join(
        KOREAN_ROOT,
        class_name
    )

    crop_area_path = os.path.join(
        folder_path,
        "crop_area.properties"
    )

    if not os.path.exists(crop_area_path):

        print(f"crop_area.properties 없음: {crop_area_path}")
        continue

    with open(crop_area_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    image_lookup = build_image_lookup(folder_path)
    data_items = []

    for line in lines:

        parsed = parse_crop_area_line(line)

        if parsed is None:
            continue

        image_stem, box = parsed
        image_name = image_lookup.get(image_stem)

        if image_name is None:
            print(f"이미지 없음: {os.path.join(folder_path, image_stem)}")
            continue

        data_items.append((image_name, box))

    random.shuffle(data_items)

    split_index = int(len(data_items) * 0.8)

    train_items = data_items[:split_index]
    valid_items = data_items[split_index:]

    for split_name, split_items in [
        ("train", train_items),
        ("valid", valid_items)
    ]:

        for image_name, box in split_items:

            image_path = os.path.join(
                folder_path,
                image_name
            )

            img = Image.open(image_path)

            w, h = img.size

            clamped_box = clamp_bbox(
                box,
                (w, h)
            )

            if clamped_box is None:
                continue

            bbox = convert_bbox(
                (w, h),
                clamped_box
            )

            yolo_class_id = korean_class_mapping[
                class_name
            ]

            new_name = f"{class_name}_{image_name}"

            dst_image_path = os.path.join(
                YOLO_ROOT,
                split_name,
                "images",
                new_name
            )

            shutil.copy(image_path, dst_image_path)

            label_path = os.path.join(
                YOLO_ROOT,
                split_name,
                "labels",
                os.path.splitext(new_name)[0] + ".txt"
            )

            with open(label_path, "w") as label_file:

                label_file.write(
                    f"{yolo_class_id} "
                    f"{bbox[0]} "
                    f"{bbox[1]} "
                    f"{bbox[2]} "
                    f"{bbox[3]}"
                )

for folder_name, class_name in self_bbox_classes.items():

    folder_path = os.path.join(
        SELF_BBOX_ROOT,
        folder_name
    )

    images_path = os.path.join(
        folder_path,
        "images"
    )

    labels_path = os.path.join(
        folder_path,
        "labels"
    )

    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        print(f"self_bbox 폴더 없음: {folder_path}")
        continue

    data_items = []

    for image_name in os.listdir(images_path):

        image_path = os.path.join(
            images_path,
            image_name
        )

        if not os.path.isfile(image_path):
            continue

        image_stem, image_ext = os.path.splitext(image_name)

        if image_ext.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        label_path = os.path.join(
            labels_path,
            image_stem + ".txt"
        )

        if not os.path.exists(label_path):
            print(f"라벨 없음: {label_path}")
            continue

        data_items.append((image_name, label_path))

    random.shuffle(data_items)

    split_index = int(len(data_items) * 0.8)

    train_items = data_items[:split_index]
    valid_items = data_items[split_index:]

    for split_name, split_items in [
        ("train", train_items),
        ("valid", valid_items)
    ]:

        for image_name, source_label_path in split_items:

            image_path = os.path.join(
                images_path,
                image_name
            )

            new_name = f"{class_name}_{image_name}"

            dst_image_path = os.path.join(
                YOLO_ROOT,
                split_name,
                "images",
                new_name
            )

            shutil.copy(image_path, dst_image_path)

            label_path = os.path.join(
                YOLO_ROOT,
                split_name,
                "labels",
                os.path.splitext(new_name)[0] + ".txt"
            )

            yolo_class_id = self_bbox_class_mapping[
                folder_name
            ]

            with open(source_label_path, "r") as source_label_file:
                source_labels = source_label_file.readlines()

            with open(label_path, "w") as label_file:

                for source_label in source_labels:

                    parts = source_label.strip().split()

                    if len(parts) != 5:
                        continue

                    label_file.write(
                        f"{yolo_class_id} "
                        f"{' '.join(parts[1:])}\n"
                    )

print("YOLO 데이터셋 생성 완료!")
