import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from train_object_classifier import extract_hog_feature, make_hog_descriptor


def classify_image(image_path: Path, model_path: Path, metadata_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    classes = metadata["classes"]
    image_size = int(metadata["image_size"])

    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    hog = make_hog_descriptor(image_size=image_size)
    feature = extract_hog_feature(resized, hog).reshape(1, -1).astype(np.float32)

    svm = cv2.ml.SVM_load(str(model_path))
    _, prediction = svm.predict(feature)
    class_idx = int(prediction[0, 0])

    print(f"Predicted class: {classes[class_idx]}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify image with trained HOG+SVM model")
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/object_classifier/object_hog_svm.yml"),
        help="Path to trained OpenCV SVM model",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("runs/object_classifier/metadata.json"),
        help="Path to metadata.json generated during training",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    classify_image(args.image, args.model, args.metadata)


if __name__ == "__main__":
    main()
