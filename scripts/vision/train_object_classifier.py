import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray


def parse_obj_mesh(obj_path: Path) -> Mesh:
    vertices: List[List[float]] = []
    triangles: List[List[int]] = []

    with obj_path.open("r", encoding="utf-8") as file:
        for raw in file:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                # Faces can be triangles/quads and each token can be "v", "v/vt", "v/vt/vn".
                face_indices = []
                for item in parts[1:]:
                    index_token = item.split("/")[0]
                    face_indices.append(int(index_token) - 1)  # OBJ is 1-indexed.

                # Fan triangulation for faces with >3 vertices.
                for i in range(1, len(face_indices) - 1):
                    triangles.append([face_indices[0], face_indices[i], face_indices[i + 1]])

    if not vertices or not triangles:
        raise ValueError(f"OBJ mesh at {obj_path} has no vertices/faces")

    return Mesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(triangles, dtype=np.int32),
    )


def rotation_matrix_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    r_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    r_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    r_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return r_z @ r_y @ r_x


def render_mesh_image(mesh: Mesh, image_size: int, rng: random.Random) -> np.ndarray:
    img = np.full((image_size, image_size, 3), rng.randint(20, 80), dtype=np.uint8)
    noise = rng.randint(0, 30)
    if noise > 0:
        img = np.clip(
            img.astype(np.int16) + np.random.randint(-noise, noise + 1, img.shape, dtype=np.int16),
            0,
            255,
        ).astype(np.uint8)

    rx = rng.uniform(-math.pi, math.pi)
    ry = rng.uniform(-math.pi, math.pi)
    rz = rng.uniform(-math.pi, math.pi)
    rot = rotation_matrix_xyz(rx, ry, rz)

    vertices = mesh.vertices @ rot.T
    center = vertices.mean(axis=0, keepdims=True)
    vertices = vertices - center

    max_extent = np.max(np.linalg.norm(vertices, axis=1))
    scale = rng.uniform(0.35, 0.6) * (image_size / max(max_extent, 1e-6))
    vertices = vertices * scale

    tx = rng.uniform(-0.12, 0.12) * image_size
    ty = rng.uniform(-0.12, 0.12) * image_size
    vertices[:, 0] += image_size / 2 + tx
    vertices[:, 1] += image_size / 2 + ty

    light_dir = np.array([0.3, -0.4, 0.8], dtype=np.float32)
    light_dir = light_dir / np.linalg.norm(light_dir)

    face_depths: List[Tuple[float, int]] = []
    for idx, tri in enumerate(mesh.faces):
        depth = float(vertices[tri, 2].mean())
        face_depths.append((depth, idx))
    face_depths.sort(key=lambda item: item[0])  # painter's algorithm (back-to-front)

    for _, face_idx in face_depths:
        tri = mesh.faces[face_idx]
        pts3d = vertices[tri]
        pts2d = pts3d[:, :2].astype(np.int32)

        if np.any(pts2d < -image_size) or np.any(pts2d > 2 * image_size):
            continue

        v0 = pts3d[1] - pts3d[0]
        v1 = pts3d[2] - pts3d[0]
        normal = np.cross(v0, v1)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal = normal / norm
        shade = float(np.clip(np.dot(normal, light_dir), -1.0, 1.0))
        shade = 0.35 + 0.65 * (shade + 1.0) * 0.5

        base_color = np.array(
            [rng.randint(80, 230), rng.randint(80, 230), rng.randint(80, 230)],
            dtype=np.float32,
        )
        color = np.clip(base_color * shade, 0, 255).astype(np.uint8).tolist()
        cv2.fillConvexPoly(img, pts2d, color=color)
        cv2.polylines(img, [pts2d], isClosed=True, color=(20, 20, 20), thickness=1)

    # Small blur and color jitter increase robustness.
    if rng.random() < 0.6:
        kernel = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (kernel, kernel), sigmaX=0.0)

    alpha = rng.uniform(0.8, 1.2)
    beta = rng.uniform(-20, 20)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img


def make_hog_descriptor(image_size: int) -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(
        (image_size, image_size),
        (32, 32),
        (16, 16),
        (16, 16),
        9,
    )


def extract_hog_feature(image_bgr: np.ndarray, hog: cv2.HOGDescriptor) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    feature = hog.compute(gray)
    return feature.flatten().astype(np.float32)


def parse_texture_path_from_mtl(mtl_path: Path) -> Optional[Path]:
    if not mtl_path.exists():
        return None
    with mtl_path.open("r", encoding="utf-8") as file:
        for raw in file:
            line = raw.strip()
            if line.lower().startswith("map_kd "):
                texture_name = line.split(maxsplit=1)[1].strip()
                texture_path = mtl_path.parent / texture_name
                if texture_path.exists():
                    return texture_path
    return None


def discover_object_meshes(objects_root: Path) -> Dict[str, Path]:
    meshes: Dict[str, Path] = {}
    for class_dir in sorted([p for p in objects_root.iterdir() if p.is_dir()]):
        candidate = class_dir / "textured_box.obj"
        if candidate.exists():
            meshes[class_dir.name] = candidate
    if not meshes:
        raise RuntimeError(f"No textured_box.obj files were found in {objects_root}")
    return meshes


def discover_class_templates(objects_root: Path) -> Dict[str, Path]:
    templates: Dict[str, Path] = {}
    for class_dir in sorted([p for p in objects_root.iterdir() if p.is_dir()]):
        texture_path = parse_texture_path_from_mtl(class_dir / "textured_box.mtl")
        if texture_path is not None:
            templates[class_dir.name] = texture_path
    return templates


def split_indices(total: int, train_ratio: float, rng: random.Random) -> Tuple[List[int], List[int]]:
    indices = list(range(total))
    rng.shuffle(indices)
    split = max(1, int(total * train_ratio))
    split = min(split, total - 1)
    return indices[:split], indices[split:]


def render_template_image(template_bgr: np.ndarray, image_size: int, rng: random.Random) -> np.ndarray:
    canvas = np.full((image_size, image_size, 3), rng.randint(25, 95), dtype=np.uint8)
    noise = rng.randint(0, 18)
    if noise > 0:
        canvas = np.clip(
            canvas.astype(np.int16) + np.random.randint(-noise, noise + 1, canvas.shape, dtype=np.int16),
            0,
            255,
        ).astype(np.uint8)

    h, w = template_bgr.shape[:2]
    crop_scale = rng.uniform(0.92, 1.0)
    ch = max(4, int(h * crop_scale))
    cw = max(4, int(w * crop_scale))
    y0 = rng.randint(0, max(0, h - ch))
    x0 = rng.randint(0, max(0, w - cw))
    cropped = template_bgr[y0:y0 + ch, x0:x0 + cw]

    target_long = int(image_size * rng.uniform(0.62, 0.86))
    aspect = cropped.shape[1] / max(1, cropped.shape[0])
    if aspect >= 1.0:
        rw = target_long
        rh = max(8, int(target_long / aspect))
    else:
        rh = target_long
        rw = max(8, int(target_long * aspect))
    resized = cv2.resize(cropped, (rw, rh), interpolation=cv2.INTER_AREA)

    mask_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, base_mask = cv2.threshold(mask_gray, 245, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    angle = rng.uniform(-25.0, 25.0)
    scale = rng.uniform(0.9, 1.08)
    center = (rw / 2.0, rh / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(resized, rot_mat, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    rotated_mask = cv2.warpAffine(base_mask, rot_mat, (rw, rh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    x = rng.randint(0, max(0, image_size - rw))
    y = rng.randint(0, max(0, image_size - rh))

    roi = canvas[y:y + rh, x:x + rw]
    alpha = (rotated_mask.astype(np.float32) / 255.0)[..., None]
    blended = (alpha * rotated.astype(np.float32) + (1.0 - alpha) * roi.astype(np.float32)).astype(np.uint8)
    canvas[y:y + rh, x:x + rw] = blended

    if rng.random() < 0.55:
        blur_k = rng.choice([3, 5])
        canvas = cv2.GaussianBlur(canvas, (blur_k, blur_k), sigmaX=0.0)

    alpha_gain = rng.uniform(0.85, 1.2)
    beta_gain = rng.uniform(-22, 22)
    canvas = cv2.convertScaleAbs(canvas, alpha=alpha_gain, beta=beta_gain)
    return canvas


def train_classifier(
    objects_root: Path,
    output_dir: Path,
    image_size: int,
    samples_per_class: int,
    train_ratio: float,
    seed: int,
) -> None:
    rng = random.Random(seed)
    np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "synthetic_dataset"
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    class_to_mesh = discover_object_meshes(objects_root)
    class_to_template = discover_class_templates(objects_root)
    class_names = sorted(class_to_mesh.keys())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    meshes = {name: parse_obj_mesh(path) for name, path in class_to_mesh.items()}
    templates: Dict[str, np.ndarray] = {}
    for name, path in class_to_template.items():
        image = cv2.imread(str(path))
        if image is not None:
            templates[name] = image

    hog = make_hog_descriptor(image_size=image_size)
    train_features: List[np.ndarray] = []
    train_labels: List[int] = []
    val_features: List[np.ndarray] = []
    val_labels: List[int] = []

    for class_name in class_names:
        class_train_dir = train_dir / class_name
        class_val_dir = val_dir / class_name
        class_train_dir.mkdir(parents=True, exist_ok=True)
        class_val_dir.mkdir(parents=True, exist_ok=True)

        images: List[np.ndarray] = []
        for _ in range(samples_per_class):
            if class_name in templates:
                images.append(render_template_image(templates[class_name], image_size=image_size, rng=rng))
            else:
                images.append(render_mesh_image(meshes[class_name], image_size=image_size, rng=rng))

        train_ids, val_ids = split_indices(len(images), train_ratio=train_ratio, rng=rng)
        class_idx = class_to_idx[class_name]

        for i in train_ids:
            path = class_train_dir / f"{class_name}_{i:04d}.png"
            cv2.imwrite(str(path), images[i])
            train_features.append(extract_hog_feature(images[i], hog))
            train_labels.append(class_idx)

        for i in val_ids:
            path = class_val_dir / f"{class_name}_{i:04d}.png"
            cv2.imwrite(str(path), images[i])
            val_features.append(extract_hog_feature(images[i], hog))
            val_labels.append(class_idx)

    x_train = np.asarray(train_features, dtype=np.float32)
    y_train = np.asarray(train_labels, dtype=np.int32)
    x_val = np.asarray(val_features, dtype=np.float32)
    y_val = np.asarray(val_labels, dtype=np.int32)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(1.5)
    svm.train(x_train, cv2.ml.ROW_SAMPLE, y_train)

    _, train_pred = svm.predict(x_train)
    _, val_pred = svm.predict(x_val)
    train_pred = train_pred.flatten().astype(np.int32)
    val_pred = val_pred.flatten().astype(np.int32)

    train_acc = float(np.mean(train_pred == y_train))
    val_acc = float(np.mean(val_pred == y_val))

    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int32)
    for gt, pred in zip(y_val, val_pred):
        confusion[gt, pred] += 1

    model_path = output_dir / "object_hog_svm.yml"
    svm.save(str(model_path))

    metadata = {
        "classes": class_names,
        "class_to_idx": class_to_idx,
        "image_size": image_size,
        "samples_per_class": samples_per_class,
        "train_ratio": train_ratio,
        "seed": seed,
        "model_path": str(model_path),
        "dataset_dir": str(dataset_dir),
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "val_confusion_matrix": confusion.tolist(),
        "template_classes": sorted(templates.keys()),
        "mesh_fallback_classes": sorted([name for name in class_names if name not in templates]),
    }
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Training finished.")
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(y_train)}, Val samples: {len(y_val)}")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Val accuracy:   {val_acc:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {meta_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train synthetic object classifier from OBJ meshes")
    parser.add_argument(
        "--objects_root",
        type=Path,
        default=Path("resources/assets/objects"),
        help="Directory with object class folders that contain textured_box.obj",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runs/object_classifier"),
        help="Where to save synthetic dataset and trained model",
    )
    parser.add_argument("--image_size", type=int, default=128, help="Synthetic image size")
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=450,
        help="Number of synthetic images generated for each class",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train_classifier(
        objects_root=args.objects_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        samples_per_class=args.samples_per_class,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
