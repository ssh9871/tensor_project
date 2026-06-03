import argparse
from pathlib import Path


BASE_DIR = Path.cwd()
DEFAULT_MODEL_PATH = BASE_DIR / "best.pt"
DEFAULT_DATA_YAML = BASE_DIR / "food_dataset" / "data.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test best.pt on food_dataset valid images."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--data", default=str(DEFAULT_DATA_YAML))
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--name", default="best_pt_valid_test")
    return parser.parse_args()


def validate_paths(model_path, data_yaml):
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일 없음: {model_path}")

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml 없음: {data_yaml}")

    valid_images = BASE_DIR / "food_dataset" / "valid" / "images"
    valid_labels = BASE_DIR / "food_dataset" / "valid" / "labels"

    if not valid_images.exists():
        raise FileNotFoundError(f"valid 이미지 폴더 없음: {valid_images}")

    if not valid_labels.exists():
        raise FileNotFoundError(f"valid 라벨 폴더 없음: {valid_labels}")


def format_metric(value):
    return "N/A" if value is None else f"{value:.4f}"


def get_box_metric(metrics, name):
    box_metrics = getattr(metrics, "box", None)
    return getattr(box_metrics, name, None)


def print_performance_summary(metrics):
    box_metrics = getattr(metrics, "box", None)

    print("\n====== best.pt 성능 테스트 결과 ======")
    print(f"Precision:     {format_metric(get_box_metric(metrics, 'mp'))}")
    print(f"Recall:        {format_metric(get_box_metric(metrics, 'mr'))}")
    print(f"mAP@50:        {format_metric(get_box_metric(metrics, 'map50'))}")
    print(f"mAP@75:        {format_metric(get_box_metric(metrics, 'map75'))}")
    print(f"mAP@50-95:     {format_metric(get_box_metric(metrics, 'map'))}")

    speed = getattr(metrics, "speed", None)
    if isinstance(speed, dict):
        print(f"Inference(ms): {format_metric(speed.get('inference'))}")

    class_maps = getattr(box_metrics, "maps", None)
    names = getattr(metrics, "names", {})

    if class_maps is not None:
        print("\n------ 클래스별 mAP@50-95 ------")
        for class_id, class_map in enumerate(class_maps):
            class_name = names.get(class_id, str(class_id))
            print(f"{class_id:>2} {class_name:<12} {class_map:.4f}")

    print("====================================\n")


def main():
    args = parse_args()
    model_path = Path(args.model)
    data_yaml = Path(args.data)

    validate_paths(model_path, data_yaml)

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        plots=True,
        name=args.name,
    )

    print_performance_summary(metrics)


if __name__ == "__main__":
    main()
