#!/usr/bin/env python3
"""Convert the pinned Rockchip YOLOv8n ONNX to an RK3588 RKNN.

FP16 remains the default. INT8 explicitly requires a hashed dataset manifest
and 640x640 prepared calibration images; it never overwrites the baseline model.
"""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import urllib.request


MODEL_ZOO_COMMIT = "bad6c7334531becaf90a561988519b7bec34d0ab"
TOOLKIT_COMMIT = "42aa1d426c0a9e0869b6374edba009f7208a1926"
ONNX_URL = "https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov8/yolov8n.onnx"
ONNX_SHA256 = "0c8716701f471067932b797eeb67c8e5db47c693c2557c881d7679ec12e21bc5"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(dataset, manifest_path):
    import cv2

    dataset, manifest_path = dataset.resolve(), manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported calibration manifest schema")
    if sha256(dataset) != manifest["dataset_sha256"]:
        raise RuntimeError("Calibration dataset list SHA256 mismatch")
    records = manifest["images"]
    lines = [line.strip() for line in dataset.read_text().splitlines() if line.strip()]
    paths = [(dataset.parent / line).resolve() for line in lines]
    expected = [pathlib.Path(item["path"]).resolve() for item in records]
    if not records or len(records) != manifest["image_count"] or paths != expected:
        raise RuntimeError("Calibration list differs from manifest")
    if len(set(paths)) != len(paths):
        raise RuntimeError("Duplicate calibration images")
    prep = manifest["preprocessing"]
    if prep["shape_hwc"] != [640, 640, 3] or prep["toolkit_quant_img_RGB2BGR"] is not False:
        raise RuntimeError("Calibration must be pre-letterboxed 640x640 RGB")
    for path, item in zip(paths, records):
        if sha256(path) != item["sha256"]:
            raise RuntimeError("Calibration image SHA256 mismatch: " + str(path))
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape != (640, 640, 3) or image.dtype.name != "uint8":
            raise RuntimeError("Calibration image must be 640x640 uint8 RGB: " + str(path))
    return manifest


def main():
    root = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=pathlib.Path, default=root / "onnx/yolov8n.onnx")
    parser.add_argument("--output", type=pathlib.Path,
                        help="Default: RK3588/yolov8n.rknn, or yolov8n_int8.rknn with --int8")
    parser.add_argument("--download", action="store_true", help="Download the pinned ONNX if it is absent")
    parser.add_argument("--int8", action="store_true", help="Explicitly enable W8A8 calibration")
    parser.add_argument("--dataset", type=pathlib.Path, help="List of prepared RGB 640x640 calibration images")
    parser.add_argument("--dataset-manifest", type=pathlib.Path, help="Hashed calibration metadata from prepare_coco20.py")
    args = parser.parse_args()
    if args.int8 != bool(args.dataset and args.dataset_manifest):
        parser.error("--int8 requires both --dataset and --dataset-manifest")
    if not args.int8 and (args.dataset or args.dataset_manifest):
        parser.error("Dataset options require --int8")
    if args.output is None:
        args.output = root / ("RK3588/yolov8n_int8.rknn" if args.int8 else "RK3588/yolov8n.rknn")
    if args.int8 and args.output.resolve() == (root / "RK3588/yolov8n.rknn").resolve():
        parser.error("INT8 output must not overwrite the FP16 baseline")
    calibration = validate_dataset(args.dataset, args.dataset_manifest) if args.int8 else None

    if not args.onnx.exists():
        if not args.download:
            parser.error("ONNX is absent; supply it or explicitly pass --download")
        args.onnx.parent.mkdir(parents=True, exist_ok=True)
        temporary_download = args.onnx.with_suffix(".onnx.download")
        with urllib.request.urlopen(ONNX_URL, timeout=120) as source, temporary_download.open("wb") as target:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
        if sha256(temporary_download) != ONNX_SHA256:
            raise RuntimeError("Official download differs from the recorded ONNX hash; refusing conversion")
        temporary_download.replace(args.onnx)
    if sha256(args.onnx) != ONNX_SHA256:
        raise RuntimeError("ONNX SHA256 mismatch; this recipe only accepts the recorded Rockchip YOLOv8n")

    toolkit_version = importlib.metadata.version("rknn-toolkit2")
    if toolkit_version != "2.3.2":
        raise RuntimeError("This conversion recipe requires rknn-toolkit2==2.3.2, found " + toolkit_version)
    import onnx
    from rknn.api import RKNN

    graph = onnx.load(str(args.onnx))
    onnx.checker.check_model(graph)

    def tensor_info(value):
        tensor = value.type.tensor_type
        return {
            "name": value.name,
            "shape": [dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param
                      for dimension in tensor.shape.dim],
            "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = args.output.with_suffix(".temporary.rknn")
    rknn = RKNN(verbose=False)
    try:
        result = rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                             target_platform="rk3588", quant_img_RGB2BGR=False,
                             quantized_dtype="w8a8", quantized_algorithm="normal",
                             quantized_method="channel")
        if result not in (None, 0):
            raise RuntimeError("RKNN config failed: {}".format(result))
        if rknn.load_onnx(model=str(args.onnx)) != 0:
            raise RuntimeError("RKNN load_onnx failed")
        build_args = {"do_quantization": args.int8}
        if args.int8:
            build_args["dataset"] = str(args.dataset.resolve())
        if rknn.build(**build_args) != 0:
            raise RuntimeError("RKNN build failed")
        if rknn.export_rknn(str(temporary_model)) != 0:
            raise RuntimeError("RKNN export failed")
    finally:
        rknn.release()
    temporary_model.replace(args.output)
    manifest = {
        "model": "YOLOv8n COCO, Rockchip optimized split-head export",
        "model_zoo_commit": MODEL_ZOO_COMMIT,
        "model_zoo_source": "https://github.com/airockchip/rknn_model_zoo/tree/" + MODEL_ZOO_COMMIT + "/examples/yolov8",
        "model_upstream": "https://github.com/airockchip/ultralytics_yolov8",
        "onnx_url": ONNX_URL,
        "onnx_sha256": ONNX_SHA256,
        "onnx_inputs": [tensor_info(value) for value in graph.graph.input],
        "onnx_outputs": [tensor_info(value) for value in graph.graph.output],
        "onnx_opsets": [{"domain": item.domain, "version": item.version} for item in graph.opset_import],
        "toolkit_version": toolkit_version,
        "toolkit_source_commit": TOOLKIT_COMMIT,
        "target_platform": "rk3588",
        "do_quantization": args.int8,
        "quantization_dataset": calibration,
        "quantization": {"dtype": "w8a8", "algorithm": "normal", "method": "channel",
                         "quant_img_RGB2BGR": False} if args.int8 else None,
        "mean_values": [[0, 0, 0]],
        "std_values": [[255, 255, 255]],
        "input_pixels": "RGB uint8 0..255; normalizing by 255 is embedded in RKNN",
        "decoder_layout": "split9: (DFL box logits, sigmoid class scores, class-score sum) at strides 8,16,32",
        "class_count": 80,
        "bbox_distribution_bins": 16,
        "rknn_file": args.output.name,
        "rknn_size_bytes": args.output.stat().st_size,
        "rknn_sha256": sha256(args.output),
        "conversion_host": {"machine": platform.machine(), "python": platform.python_version(),
                            "platform": platform.platform()},
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("RKNN:", args.output, manifest["rknn_sha256"])
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
