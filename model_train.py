import argparse
import os
from multiprocessing import freeze_support


BASE_DIR = os.getcwd()
KOREAN_ROOT = os.path.join(os.path.join(BASE_DIR, "data"), "korean_food_images")
YOLO_ROOT = os.path.join(BASE_DIR, "food_dataset")

SELECTED_CLASS_NAMES = [
    "밥",
    "비빔밥",
    "햄버거",
    "피자",
    "스파게티",
    "소시지",
    "돈까스",
    "달걀 프라이",
]

SELF_BBOX_CLASS_NAMES = [
    "짜장",
    "제육볶음",
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

    return SELECTED_CLASS_NAMES + korean_class_names + SELF_BBOX_CLASS_NAMES


def create_data_yaml():
    all_class_names = get_all_class_names()
    names_text = "\n".join(
        f"  {index}: {class_name}"
        for index, class_name in enumerate(all_class_names)
    )

    yaml_text = f"""train: {os.path.join(YOLO_ROOT, 'train', 'images')}
val: {os.path.join(YOLO_ROOT, 'valid', 'images')}

nc: {len(all_class_names)}

names:
{names_text}
"""

    yaml_path = os.path.join(YOLO_ROOT, "data.yaml")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print("data.yaml 생성 완료!")
    return yaml_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the YOLO food detection model with tuned defaults."
    )
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--name", default="food_tuned")
    return parser.parse_args()


def train_model(args):
    from ultralytics import YOLO

    yaml_path = create_data_yaml()
    model = YOLO(args.model)

    model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        name=args.name,
        pretrained=True,
        seed=42,
        deterministic=True,
        cos_lr=True,
        close_mosaic=15,
        optimizer="AdamW",
        lr0=0.003,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        hsv_h=0.015,
        hsv_s=0.65,
        hsv_v=0.35,
        degrees=5.0,
        translate=0.08,
        scale=0.45,
        shear=2.0,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.1,
        val=True,
        plots=True,
    )

    print("학습 완료!")


if __name__ == "__main__":
    freeze_support()
    train_model(parse_args())
