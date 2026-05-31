import os
from multiprocessing import freeze_support

from ultralytics import YOLO


BASE_DIR = os.getcwd()
KOREAN_ROOT = os.path.join(BASE_DIR, "korean_food_images")
YOLO_ROOT = os.path.join(BASE_DIR, "food_dataset")

selected_class_names = [
    "밥",
    "비빔밥",
    "햄버거",
    "피자",
    "스파게티",
    "소시지",
    "돈까스",
    "달걀 프라이",
]


def get_all_class_names():

    korean_class_names = []

    if os.path.exists(KOREAN_ROOT):
        korean_class_names = sorted([
            folder_name
            for folder_name in os.listdir(KOREAN_ROOT)
            if os.path.isdir(os.path.join(KOREAN_ROOT, folder_name))
        ])
    else:
        print(f"korean_food_images 폴더 없음: {KOREAN_ROOT}")

    return selected_class_names + korean_class_names


def create_data_yaml():

    all_class_names = get_all_class_names()

    yaml_text = f"""
train: {os.path.join(YOLO_ROOT, 'train', 'images')}
val: {os.path.join(YOLO_ROOT, 'valid', 'images')}

nc: {len(all_class_names)}

names:
{os.linesep.join([
    f"  {index}: {class_name}"
    for index, class_name in enumerate(all_class_names)
])}
"""

    yaml_path = os.path.join(YOLO_ROOT, "data.yaml")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print("data.yaml 생성 완료!")

    return yaml_path


if __name__ == "__main__":

    freeze_support()

    yaml_path = create_data_yaml()

    model = YOLO("yolov8n.pt")

    model.train(
        data=yaml_path,
        epochs=30,
        imgsz=640,
        batch=16,
        workers=0
    )

    print("학습 완료!")
