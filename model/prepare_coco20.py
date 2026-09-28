#!/usr/bin/env python3
"""Prepare the pinned official COCO subset20 for RGB letterbox calibration.

Downloads live under build/, never in the source tree. Source hashes are fixed
in coco20.sources.json. These 20 images must not be used as held-out evaluation.
"""

import argparse
import hashlib
import json
import math
import pathlib
import urllib.request


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=root.parent / "build/model_prepare/coco20")
    args = parser.parse_args()
    import cv2
    import numpy as np

    cv2.setNumThreads(1)
    sources_path = root / "coco20.sources.json"
    sources = json.loads(sources_path.read_text())
    output = args.output_dir.resolve()
    original, prepared = output / "original", output / "letterbox640"
    original.mkdir(parents=True, exist_ok=True)
    prepared.mkdir(parents=True, exist_ok=True)
    records = []
    for item in sources["images"]:
        path = original / item["file"]
        if not path.exists():
            temporary = path.with_suffix(".download")
            with urllib.request.urlopen(item["url"], timeout=120) as response:
                temporary.write_bytes(response.read())
            if sha256(temporary) != item["sha256"]:
                raise RuntimeError("Download SHA256 mismatch: " + item["file"])
            temporary.replace(path)
        if sha256(path) != item["sha256"]:
            raise RuntimeError("Source SHA256 mismatch: " + item["file"])
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Cannot decode " + str(path))
        height, width = image.shape[:2]
        scale = min(640.0 / width, 640.0 / height)
        # Match std::round on positive dimensions in src/yolo_decode.cpp.
        resized_width = max(1, min(640, int(math.floor(width * scale + 0.5))))
        resized_height = max(1, min(640, int(math.floor(height * scale + 0.5))))
        left, top = (640 - resized_width) // 2, (640 - resized_height) // 2
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[top:top + resized_height, left:left + resized_width] = cv2.resize(
            image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        target = prepared / (path.stem + ".png")
        # imwrite consumes BGR and writes an ordinary RGB image file. Toolkit's
        # quant_img_RGB2BGR=False loader reads RGB; there is no double swap.
        if not cv2.imwrite(str(target), canvas):
            raise RuntimeError("Cannot save " + str(target))
        records.append({
            "path": str(target), "sha256": sha256(target),
            "source_file": item["file"], "source_sha256": item["sha256"],
            "source_url": item["url"], "original_size_wh": [width, height],
            "resized_size_wh": [resized_width, resized_height],
            "left_top": [left, top],
        })
    dataset = output / "dataset.txt"
    dataset.write_text("".join(item["path"] + "\n" for item in records))
    manifest = {
        "schema_version": 1,
        "purpose": "INT8 calibration only, excluded from held-out evaluation",
        "source_manifest": str(sources_path),
        "source_manifest_sha256": sha256(sources_path),
        "dataset": str(dataset), "dataset_sha256": sha256(dataset),
        "image_count": len(records),
        "preprocessing": {
            "shape_hwc": [640, 640, 3], "pixel_dtype": "uint8",
            "file_color_space": "RGB", "toolkit_quant_img_RGB2BGR": False,
            "resize": "OpenCV INTER_LINEAR; positive dimensions rounded half up",
            "padding": "114 on all channels; left/top floor of remaining/2",
            "normalization": "none in images; RKNN mean=0 std=255",
            "opencv_version": cv2.__version__,
        },
        "images": records,
    }
    manifest_path = output / "dataset.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Dataset:", dataset)
    print("Manifest:", manifest_path)
    print("Images:", len(records))


if __name__ == "__main__":
    main()
