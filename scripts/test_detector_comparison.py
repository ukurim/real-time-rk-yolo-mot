#!/usr/bin/env python3
"""CPU-only checks for model-agreement matching/reporting semantics."""
import json
from pathlib import Path
import tempfile
import unittest

from analyze_detector_comparison import compare_frame, iou, load_records, qualified_pairs, summarize


class AgreementTests(unittest.TestCase):
    def test_iou_geometry(self):
        self.assertEqual(iou([0, 0, 10, 10], [10, 0, 20, 10]), 0)
        self.assertAlmostEqual(iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)

    def test_global_cardinality_precedes_largest_local_overlap(self):
        # A greedy 0.99 match would consume the only eligible candidate for row1.
        self.assertEqual(qualified_pairs([[.99, .6], [.6, .49]], .5), [(0, 1), (1, 0)])

    def test_best_total_iou_at_same_cardinality(self):
        self.assertEqual(qualified_pairs([[.95, .7], [.75, .8]], .5), [(0, 0), (1, 1)])

    def test_rectangular_and_no_matches(self):
        self.assertEqual(qualified_pairs([[.1], [.9], [.5]], .5), [(1, 0)])
        self.assertEqual(qualified_pairs([[.49]], .5), [])
        self.assertEqual(qualified_pairs([[]], .5), [])
        self.assertEqual(qualified_pairs([], .5), [])
        self.assertEqual(qualified_pairs([[.5]], .5), [(0, 0)])

    def test_classes_threshold_and_original_indices(self):
        frame = {"frame_id": 7,
                 "baseline": [{"class_id": 1, "score": .2, "bbox_xyxy": [0, 0, 10, 10]},
                              {"class_id": 2, "score": .8, "bbox_xyxy": [0, 0, 10, 10]}],
                 "candidate": [{"class_id": 1, "score": .9, "bbox_xyxy": [0, 0, 10, 10]},
                               {"class_id": 2, "score": .7, "bbox_xyxy": [1, 0, 11, 10]}]}
        result = compare_frame(frame, .25, .5)
        self.assertEqual(result["baseline_count"], 1)
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["unmatched_baseline_indices"], [])
        self.assertEqual(result["unmatched_candidate_indices"], [0])
        match = result["matches"][0]
        self.assertEqual((match["baseline_index"], match["candidate_index"]), (1, 1))
        self.assertAlmostEqual(match["score_delta_candidate_minus_baseline"], -.1)
        self.assertEqual(match["center_distance_pixels"], 1)
        report = summarize([frame], [result], .25)
        self.assertEqual(report["matched_fraction_of_baseline"], 1)
        self.assertEqual(report["matched_fraction_of_candidate"], .5)
        self.assertEqual(report["per_class"]["1"]["unmatched_candidate"], 1)

    def test_empty_frame_has_no_fabricated_agreement(self):
        frame = {"frame_id": 0, "baseline": [], "candidate": []}
        report = summarize([frame], [compare_frame(frame, .1, .5)], .1)
        self.assertIsNone(report["matched_fraction_of_baseline"])
        self.assertIsNone(report["matched_iou"])

    def test_reject_truncated_interrupted_or_wrong_sequence(self):
        metadata = {"record_type": "run", "schema_version": 1,
                    "sampling": {"start_frame": 5, "stride": 10, "requested_samples": 1},
                    "detector": {"class_count": 80, "confidence_threshold": .1}}
        frame = {"record_type": "frame", "sample_index": 0, "frame_id": 5,
                 "timestamp_domain": "stream_pts", "image_timestamp_ns": 500,
                 "image_width": 1920, "image_height": 1080, "baseline": [], "candidate": []}
        summary = {"record_type": "summary", "selected_frames": 1,
                   "requested_samples": 1, "interrupted": False}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"

            def write(records):
                path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

            write([metadata, frame, summary])
            self.assertEqual(len(load_records(path)[1]), 1)
            write([metadata, frame])
            with self.assertRaisesRegex(ValueError, "completed summary"):
                load_records(path)
            write([metadata, frame, {**summary, "interrupted": True}])
            with self.assertRaisesRegex(ValueError, "interrupted"):
                load_records(path)
            write([metadata, {**frame, "frame_id": 6}, summary])
            with self.assertRaisesRegex(ValueError, "selection metadata"):
                load_records(path)


if __name__ == "__main__":
    unittest.main()
