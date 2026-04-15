# Object Image Classifier (Synthetic Data)

This folder contains a simple image classifier pipeline for objects from `resources/assets/objects`.

## What it does

1. Reads one template image per class from `map_Kd` in `*/textured_box.mtl` (for example `backpack.jpg`).
2. Generates synthetic 2D images from heavy augmentations of that template.
3. Falls back to mesh rendering from `*/textured_box.obj` if template is missing.
4. Trains a `HOG + Linear SVM` classifier in OpenCV.
5. Saves the model and metadata for inference.

## Train

Run from project root (`roboschool_competition`):

```bash
python scripts/vision/train_object_classifier.py \
  --objects_root resources/assets/objects \
  --output_dir runs/object_classifier \
  --samples_per_class 450 \
  --image_size 128
```

Outputs:

- `runs/object_classifier/synthetic_dataset/train/...`
- `runs/object_classifier/synthetic_dataset/val/...`
- `runs/object_classifier/object_hog_svm.yml`
- `runs/object_classifier/metadata.json`

## Inference

```bash
python scripts/vision/classify_image.py \
  --image path/to/image.png \
  --model runs/object_classifier/object_hog_svm.yml \
  --metadata runs/object_classifier/metadata.json
```

## Notes

- Current pipeline prefers texture images (`map_Kd`) when available because this is usually closer to camera appearance.
- If you later collect camera frames from the simulator, you can reuse the same classifier code and retrain on mixed synthetic+real images.
