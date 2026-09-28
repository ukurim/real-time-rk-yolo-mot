#!/usr/bin/env python3
"""Summarize post-NMS reference-model agreement; never reports accuracy or mAP.

The only dependency is Python 3.8+. Matching is exact, one-to-one, and per class:
maximize number of IoU-qualified pairs, then maximize their sum of IoUs.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iou(box_a, box_b):
    intersection = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])) * max(
        0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    union = ((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]) +
             (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]) - intersection)
    return intersection / union if union > 0 else 0.0


def min_cost_assignment(cost):
    """Rectangular Hungarian assignment, rows <= columns, deterministic ties."""
    if not cost:
        return []
    rows, columns = len(cost), len(cost[0])
    if rows > columns or any(len(row) != columns for row in cost):
        raise ValueError("Assignment matrix must be rectangular with rows <= columns")
    u, v = [0.0] * (rows + 1), [0.0] * (columns + 1)
    p, way = [0] * (columns + 1), [0] * (columns + 1)
    for row in range(1, rows + 1):
        p[0], column = row, 0
        minimum, used = [math.inf] * (columns + 1), [False] * (columns + 1)
        while True:
            used[column] = True
            active_row = p[column]
            delta, next_column = math.inf, 0
            for candidate_column in range(1, columns + 1):
                if used[candidate_column]:
                    continue
                reduced = cost[active_row - 1][candidate_column - 1] - u[active_row] - v[candidate_column]
                if reduced < minimum[candidate_column]:
                    minimum[candidate_column], way[candidate_column] = reduced, column
                if minimum[candidate_column] < delta:
                    delta, next_column = minimum[candidate_column], candidate_column
            for candidate_column in range(columns + 1):
                if used[candidate_column]:
                    u[p[candidate_column]] += delta
                    v[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if p[column] == 0:
                break
        while True:
            previous_column = way[column]
            p[column] = p[previous_column]
            column = previous_column
            if column == 0:
                break
    result = [-1] * rows
    for column in range(1, columns + 1):
        if p[column]:
            result[p[column] - 1] = column - 1
    return result


def qualified_pairs(overlaps, overlap_threshold):
    """Maximize cardinality first, total IoU second; unmatched rows cost zero."""
    if not overlaps or not overlaps[0]:
        return []
    rows, columns = len(overlaps), len(overlaps[0])
    # Losing one valid pair can never be compensated by all other IoU gains.
    cardinality_bonus = min(rows, columns) + 1.0
    prohibited_cost = cardinality_bonus * (rows + columns + 1)
    cost = [[-(cardinality_bonus + overlap) if overlap >= overlap_threshold else prohibited_cost
             for overlap in row] + [0.0] * rows for row in overlaps]
    assignments = min_cost_assignment(cost)
    return [(row, column) for row, column in enumerate(assignments)
            if column < columns and overlaps[row][column] >= overlap_threshold]


def compare_frame(frame, threshold, overlap_threshold):
    baseline = [(index, det) for index, det in enumerate(frame["baseline"]) if det["score"] >= threshold]
    candidate = [(index, det) for index, det in enumerate(frame["candidate"]) if det["score"] >= threshold]
    matches = []
    classes = sorted({det["class_id"] for _, det in baseline + candidate})
    for class_id in classes:
        class_baseline = [(index, det) for index, det in baseline if det["class_id"] == class_id]
        class_candidate = [(index, det) for index, det in candidate if det["class_id"] == class_id]
        overlaps = [[iou(left["bbox_xyxy"], right["bbox_xyxy"])
                     for _, right in class_candidate] for _, left in class_baseline]
        for left, right in qualified_pairs(overlaps, overlap_threshold):
            baseline_index, baseline_det = class_baseline[left]
            candidate_index, candidate_det = class_candidate[right]
            baseline_box, candidate_box = baseline_det["bbox_xyxy"], candidate_det["bbox_xyxy"]
            center_delta_x = (candidate_box[0] + candidate_box[2] - baseline_box[0] - baseline_box[2]) / 2
            center_delta_y = (candidate_box[1] + candidate_box[3] - baseline_box[1] - baseline_box[3]) / 2
            matches.append({"class_id": class_id, "baseline_index": baseline_index,
                            "candidate_index": candidate_index, "iou": overlaps[left][right],
                            "baseline_score": baseline_det["score"], "candidate_score": candidate_det["score"],
                            "score_delta_candidate_minus_baseline": candidate_det["score"] - baseline_det["score"],
                            "center_delta_xy_pixels": [center_delta_x, center_delta_y],
                            "center_distance_pixels": math.hypot(center_delta_x, center_delta_y)})
    matched_baseline = {match["baseline_index"] for match in matches}
    matched_candidate = {match["candidate_index"] for match in matches}
    return {"frame_id": frame["frame_id"], "confidence_threshold": threshold,
            "baseline_count": len(baseline), "candidate_count": len(candidate), "matches": matches,
            "unmatched_baseline_indices": [index for index, _ in baseline if index not in matched_baseline],
            "unmatched_candidate_indices": [index for index, _ in candidate if index not in matched_candidate]}


def percentile(sorted_values, fraction):
    position = fraction * (len(sorted_values) - 1)
    low = math.floor(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def stats(values):
    if not values:
        return None
    values = sorted(values)
    return {"count": len(values), "mean": statistics.fmean(values), "min": values[0],
            "p05": percentile(values, .05), "p50": percentile(values, .50),
            "p95": percentile(values, .95), "p99": percentile(values, .99), "max": values[-1]}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def count_summary(baseline, candidate, matched):
    return {"baseline_count": baseline, "candidate_count": candidate, "matched_pairs": matched,
            "unmatched_baseline": baseline - matched, "unmatched_candidate": candidate - matched,
            "matched_fraction_of_baseline": ratio(matched, baseline),
            "matched_fraction_of_candidate": ratio(matched, candidate)}


def summarize(frames, comparisons, threshold):
    pairs = [match for comparison in comparisons for match in comparison["matches"]]
    result = {"confidence_threshold": threshold, **count_summary(
        sum(item["baseline_count"] for item in comparisons),
        sum(item["candidate_count"] for item in comparisons), len(pairs)),
        "matched_iou": stats([match["iou"] for match in pairs]),
        "matched_score_delta_candidate_minus_baseline": stats(
            [match["score_delta_candidate_minus_baseline"] for match in pairs]),
        "matched_absolute_score_delta": stats([abs(match["score_delta_candidate_minus_baseline"]) for match in pairs]),
        "matched_center_distance_pixels": stats([match["center_distance_pixels"] for match in pairs])}
    by_class = defaultdict(lambda: [0, 0, 0])
    for frame in frames:
        for column, name in enumerate(("baseline", "candidate")):
            for detection in frame[name]:
                if detection["score"] >= threshold:
                    by_class[detection["class_id"]][column] += 1
    for pair in pairs:
        by_class[pair["class_id"]][2] += 1
    result["per_class"] = {str(class_id): count_summary(*counts) for class_id, counts in sorted(by_class.items())}
    return result


def load_records(path):
    run, summary, frames = None, None, []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON: " + value)))
            record_type = record.get("record_type")
            if summary is not None:
                raise ValueError("Record after summary at line {}".format(line_number))
            if record_type == "run":
                if run is not None or frames or record.get("schema_version") != 1:
                    raise ValueError("Expected one schema_version=1 run record first")
                run = record
            elif record_type == "frame":
                if run is None:
                    raise ValueError("Frame before run metadata")
                expected_id = run["sampling"]["start_frame"] + len(frames) * run["sampling"]["stride"]
                if record["sample_index"] != len(frames) or record["frame_id"] != expected_id:
                    raise ValueError("Sample index/frame_id inconsistent with selection metadata")
                if record["timestamp_domain"] != "stream_pts":
                    raise ValueError("Evaluation video requires source PTS")
                if frames and record["image_timestamp_ns"] <= frames[-1]["image_timestamp_ns"]:
                    raise ValueError("Source image timestamps are not strictly increasing")
                if min(record["image_width"], record["image_height"]) <= 0:
                    raise ValueError("Invalid image dimensions")
                for name in ("baseline", "candidate"):
                    for detection in record[name]:
                        box, score = detection["bbox_xyxy"], detection["score"]
                        if (not isinstance(detection["class_id"], int) or
                                not 0 <= detection["class_id"] < run["detector"]["class_count"] or
                                len(box) != 4 or not all(math.isfinite(value) for value in box) or
                                not math.isfinite(score) or not 0 <= score <= 1 or box[2] <= box[0] or box[3] <= box[1]):
                            raise ValueError("Invalid detection at frame {}".format(record["frame_id"]))
                        if score + 1e-7 < run["detector"]["confidence_threshold"]:
                            raise ValueError("Detection below recorded generation threshold")
                frames.append(record)
            elif record_type == "summary":
                summary = record
            else:
                raise ValueError("Unknown record_type at line {}".format(line_number))
    if run is None or summary is None or not frames:
        raise ValueError("Comparison needs run metadata, nonempty frames and a completed summary")
    if summary.get("interrupted") or summary["selected_frames"] != len(frames):
        raise ValueError("Comparison was interrupted or selected-frame count differs")
    if summary["requested_samples"] != run["sampling"]["requested_samples"]:
        raise ValueError("Requested sample counts differ")
    return run, frames, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL from compare_detectors")
    parser.add_argument("--output", type=Path, help="Summary JSON, default INPUT.summary.json")
    parser.add_argument("--details", type=Path, help="Optional JSONL with per-frame matched/unmatched indices")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[.1, .25, .5])
    parser.add_argument("--match-iou", type=float, default=.5)
    args = parser.parse_args()
    if (any(not math.isfinite(value) or not 0 < value <= 1 for value in args.thresholds) or
            not math.isfinite(args.match_iou) or not 0 < args.match_iou <= 1):
        parser.error("Confidence/IoU thresholds must be finite in (0,1]")
    output = args.output or args.input.with_suffix(".summary.json")
    paths = [args.input.resolve(), output.resolve()]
    if args.details:
        paths.append(args.details.resolve())
    if len(paths) != len(set(paths)):
        parser.error("Input, summary and details must have different paths")
    run, frames, collection_summary = load_records(args.input)
    if min(args.thresholds) + 1e-7 < run["detector"]["confidence_threshold"]:
        parser.error("Cannot analyze below the confidence threshold used to generate detections")
    summary = {"schema_version": 1, "evaluation": "reference_model_agreement_not_ground_truth",
               "warning": "FP16 agreement is not detection accuracy, precision/recall against labels, or mAP. "
                          "Both models can make the same error. Calibration/evaluation overlap must be audited separately.",
               "input_jsonl": {"path": str(args.input.resolve()), "sha256": sha256(args.input)},
               "run": run, "collection_summary": collection_summary,
               "matching": {"method": "same_class_maximum_cardinality_then_maximum_sum_iou",
                            "iou_threshold": args.match_iou, "score_filter": "score >= threshold, both models",
                            "boxes": "post_NMS_xyxy_original_distorted_image_pixels",
                            "index_domain": "zero_based_detection_index_in_original_frame_record",
                            "calibration_overlap": "not_checked_by_tool"},
               "sample_count": len(frames), "threshold_results": []}
    details = []
    for threshold in sorted(set(args.thresholds)):
        comparisons = [compare_frame(frame, threshold, args.match_iou) for frame in frames]
        summary["threshold_results"].append(summarize(frames, comparisons, threshold))
        if args.details:
            details.extend(comparisons)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.details:
        with args.details.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps({"record_type": "protocol", "input_jsonl": summary["input_jsonl"],
                                     "matching": summary["matching"]}, allow_nan=False) + "\n")
            for detail in details:
                stream.write(json.dumps({"record_type": "comparison", **detail}, allow_nan=False) + "\n")
    print("Reference-model agreement only; {} sampled frames; not mAP/ground-truth accuracy.".format(len(frames)))
    print("score  baseline  candidate  matched  unmatched_ref  unmatched_cand  IoU_p50  |score_delta|_p95")
    for row in summary["threshold_results"]:
        print("{:.2f} {:9d} {:10d} {:8d} {:14d} {:15d} {:>8} {:>18}".format(
            row["confidence_threshold"], row["baseline_count"], row["candidate_count"], row["matched_pairs"],
            row["unmatched_baseline"], row["unmatched_candidate"],
            "{:.4f}".format(row["matched_iou"]["p50"]) if row["matched_iou"] else "n/a",
            "{:.4f}".format(row["matched_absolute_score_delta"]["p95"]) if row["matched_absolute_score_delta"] else "n/a"))
    print("Summary: {}".format(output))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError) as error:
        print("analyze_detector_comparison: {}".format(error), file=sys.stderr)
        sys.exit(1)
