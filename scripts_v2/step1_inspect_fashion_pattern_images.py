#!/usr/bin/env python3
"""Step1: inspect yainage90/fashion-pattern-images (gated), export label stats and previews."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import ClassLabel, Features, load_dataset
from PIL import Image, ImageDraw
from tqdm import tqdm

TARGET_STYLE11 = [
    "chequered",
    "dotted",
    "grid",
    "honeycombed",
    "knitted",
    "paisley",
    "polka-dotted",
    "striped",
    "waffled",
    "woven",
    "zigzagged",
]

IMAGE_CANDIDATE_KEYS = ["image", "img", "pixel_values", "picture", "photo"]
LABEL_CANDIDATE_KEYS = ["pattern", "label", "labels", "category", "class", "style"]


class Step1Error(RuntimeError):
    """Readable error for user-facing failures."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a gated HF dataset, print/serialize label counts, and save per-label previews."
        )
    )
    parser.add_argument("--dataset", default="yainage90/fashion-pattern-images")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out_dir", default="outputs_v2")
    parser.add_argument("--max_preview_per_label", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text).strip())
    slug = slug.strip("._-")
    return slug or "unknown_label"


def pretty_features(features: Features) -> str:
    return json.dumps({k: str(v) for k, v in features.items()}, ensure_ascii=False, indent=2)


def pick_image_field(features: Features) -> Optional[str]:
    keys = list(features.keys())
    for key in keys:
        if "Image" in str(features[key]):
            return key
    for c in IMAGE_CANDIDATE_KEYS:
        if c in keys:
            return c
    return None


def pick_label_field(features: Features, image_field: str) -> Optional[str]:
    keys = [k for k in features.keys() if k != image_field]
    for key in keys:
        if isinstance(features[key], ClassLabel):
            return key
    for c in LABEL_CANDIDATE_KEYS:
        if c in keys:
            return c
    for key in keys:
        fstr = str(features[key]).lower()
        if "classlabel" in fstr or "label" in key.lower() or "pattern" in key.lower():
            return key
    return keys[0] if keys else None


def load_dataset_with_auth(dataset: str, split: str):
    try:
        return load_dataset(dataset, split=split, token=True)
    except TypeError:
        return load_dataset(dataset, split=split, use_auth_token=True)
    except Exception:
        try:
            return load_dataset(dataset, split=split, use_auth_token=True)
        except Exception as exc:
            raise Step1Error(build_auth_error_message(dataset, exc)) from exc


def build_auth_error_message(dataset: str, exc: Exception) -> str:
    detail = str(exc)
    lower = detail.lower()
    if any(x in lower for x in ["403", "forbidden", "gated", "unauthorized", "401", "cannot be accessed", "doesn't exist"]):
        return (
            f"Failed to access dataset '{dataset}' due to authentication/gated access.\n"
            "Please complete these steps first:\n"
            "1) Open the dataset page and click Agree/Access: "
            f"https://huggingface.co/datasets/{dataset}\n"
            "2) Login locally with a READ token:\n"
            "   - python -m huggingface_hub login\n"
            "   - or hf auth login\n"
            "3) Re-run this script.\n"
            f"Original error: {detail}"
        )
    return (
        f"Failed to load dataset '{dataset}'.\n"
        "If this is a gated dataset, make sure you accepted access conditions on the webpage "
        "and logged in with a Hugging Face Read token.\n"
        f"Original error: {detail}"
    )


def resolve_label_info(ds, label_field: str) -> Tuple[Dict[int, str], List[int], List[str]]:
    raw_values = ds[label_field]
    feature = ds.features[label_field]

    if isinstance(feature, ClassLabel):
        ids = [int(v) for v in raw_values]
        names = [str(feature.int2str(i)) for i in ids]
        id_to_name = {int(i): str(name) for i, name in enumerate(feature.names)}
        return id_to_name, ids, names

    unique_values = sorted(set(raw_values), key=lambda x: str(x))
    value_to_id = {v: i for i, v in enumerate(unique_values)}
    ids = [value_to_id[v] for v in raw_values]
    id_to_name = {i: str(v) for v, i in value_to_id.items()}
    names = [id_to_name[i] for i in ids]
    return id_to_name, ids, names


def ensure_rgb(img_obj: Any) -> Image.Image:
    if isinstance(img_obj, Image.Image):
        return img_obj.convert("RGB")
    if isinstance(img_obj, dict) and "bytes" in img_obj:
        from io import BytesIO

        return Image.open(BytesIO(img_obj["bytes"])).convert("RGB")
    if isinstance(img_obj, str):
        return Image.open(img_obj).convert("RGB")
    raise Step1Error(f"Unsupported image object type for preview export: {type(img_obj)}")


def make_grid(
    sampled_paths_by_label: Dict[str, List[Path]],
    out_path: Path,
    thumb_size: Tuple[int, int] = (128, 128),
) -> None:
    labels = list(sampled_paths_by_label.keys())
    if not labels:
        raise Step1Error("No sampled images were collected; cannot build preview grid.")

    max_cols = max(len(v) for v in sampled_paths_by_label.values())
    label_col_w = 220
    cell_w, cell_h = thumb_size
    margin = 8
    row_h = cell_h + margin * 2
    header_h = 36

    canvas_w = label_col_w + max_cols * (cell_w + margin * 2)
    canvas_h = header_h + len(labels) * row_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)

    draw.text((10, 10), "Preview grid by label (rows)", fill=(20, 20, 20))

    for row_idx, label in enumerate(labels):
        y0 = header_h + row_idx * row_h
        draw.text((10, y0 + cell_h // 2 - 8), label, fill=(10, 10, 10))
        for col_idx, img_path in enumerate(sampled_paths_by_label[label]):
            x0 = label_col_w + col_idx * (cell_w + margin * 2) + margin
            y1 = y0 + margin
            try:
                tile = Image.open(img_path).convert("RGB")
                tile.thumbnail((cell_w, cell_h))
                tile_canvas = Image.new("RGB", (cell_w, cell_h), (230, 230, 230))
                tx = (cell_w - tile.width) // 2
                ty = (cell_h - tile.height) // 2
                tile_canvas.paste(tile, (tx, ty))
                canvas.paste(tile_canvas, (x0, y1))
            except Exception:
                pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    preview_root = out_dir / "preview_fashion_patterns"
    preview_root.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset_with_auth(args.dataset, args.split)
    except Step1Error as exc:
        print(str(exc), file=sys.stderr)
        return 2

    features = ds.features
    image_field = pick_image_field(features)
    if not image_field:
        print("Could not auto-detect image field. Dataset features:", file=sys.stderr)
        print(pretty_features(features), file=sys.stderr)
        return 3

    label_field = pick_label_field(features, image_field=image_field)
    if not label_field:
        print("Could not auto-detect pattern/label field. Dataset features:", file=sys.stderr)
        print(pretty_features(features), file=sys.stderr)
        return 4

    id_to_name, label_ids, label_names = resolve_label_info(ds, label_field)
    count_by_id = Counter(label_ids)
    count_by_name = Counter(label_names)

    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Size: {len(ds)}")
    print(f"Detected image field: {image_field}")
    print(f"Detected label field: {label_field}")
    print("Label counts (id -> name -> count):")
    for lid in sorted(id_to_name.keys()):
        name = id_to_name[lid]
        print(f"  {lid:>2} -> {name:<24} -> {count_by_id.get(lid, 0)}")

    stats = {
        "dataset": args.dataset,
        "split": args.split,
        "size": len(ds),
        "image_field": image_field,
        "label_field": label_field,
        "id_to_name": {str(k): v for k, v in sorted(id_to_name.items(), key=lambda kv: kv[0])},
        "count_by_id": {str(k): int(v) for k, v in sorted(count_by_id.items(), key=lambda kv: kv[0])},
        "count_by_name": {k: int(v) for k, v in sorted(count_by_name.items(), key=lambda kv: kv[0])},
    }
    write_json(out_dir / "fashion_pattern_images_label_stats.json", stats)

    # Sample indices for each label.
    indices_by_name: Dict[str, List[int]] = defaultdict(list)
    for idx, n in enumerate(label_names):
        indices_by_name[n].append(idx)

    sampled_paths_by_label: Dict[str, List[Path]] = {}
    for label_name in sorted(indices_by_name.keys()):
        idxs = indices_by_name[label_name][:]
        random.shuffle(idxs)
        take = idxs[: max(1, args.max_preview_per_label)]
        label_dir = preview_root / safe_slug(label_name)
        label_dir.mkdir(parents=True, exist_ok=True)

        paths: List[Path] = []
        for i, idx in enumerate(tqdm(take, desc=f"Exporting {label_name}", leave=False), start=1):
            row = ds[int(idx)]
            image = ensure_rgb(row[image_field])
            out_img_path = label_dir / f"sample_{i:04d}.png"
            image.save(out_img_path)
            paths.append(out_img_path)
        sampled_paths_by_label[label_name] = paths

    make_grid(sampled_paths_by_label, preview_root / "grid_by_label.png")

    mapping_template = {
        "source_dataset": args.dataset,
        "source_labels": {str(k): v for k, v in sorted(id_to_name.items(), key=lambda kv: kv[0])},
        "target_style11": TARGET_STYLE11,
        "mapping": {name: "" for _, name in sorted(id_to_name.items(), key=lambda kv: kv[0])},
        "notes": "Fill mapping values with one of target_style11 or leave empty to drop.",
    }
    write_json(out_dir / "pattern19_to_style11_template.json", mapping_template)

    print("\nDone. Generated files:")
    print(f"- {out_dir / 'fashion_pattern_images_label_stats.json'}")
    print(f"- {out_dir / 'pattern19_to_style11_template.json'}")
    print(f"- {preview_root / 'grid_by_label.png'}")
    print(f"- {preview_root / '<label_name>/sample_XXXX.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
