#!/usr/bin/env python3
"""Validate actual application output, including ordering and LOS conventions."""
import argparse
import json
import math


def check(path, expected_frames=None):
    count = targets = predicted = 0
    previous_id = previous_time = -1
    domain = None
    with open(path) as source:
        for line in source:
            row = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            assert row['frame_id'] > previous_id
            assert row['image_timestamp_ns'] > previous_time
            assert domain is None or row['image_timestamp_domain'] == domain
            previous_id, previous_time, domain = row['frame_id'], row['image_timestamp_ns'], row['image_timestamp_domain']
            assert row['publish_started_monotonic_ns'] >= row['result_ready_monotonic_ns'] >= row['received_monotonic_ns']
            for value in row['latency_ms'].values():
                assert value is None or (math.isfinite(value) and value >= 0)
            ids = set()
            for target in row['targets']:
                assert target['track_id'] not in ids
                ids.add(target['track_id'])
                bbox = target['bbox_xyxy']
                assert all(math.isfinite(v) for v in bbox)
                assert bbox[2] > bbox[0] and bbox[3] > bbox[1]
                assert 0 <= target['score'] <= 1
                assert target['seconds_since_update'] >= 0
                if target['detection_updated']:
                    assert target['observation'] == 'detection_update'
                    assert target['seconds_since_update'] == 0
                else:
                    assert target['observation'] == 'prediction'
                    assert target['seconds_since_update'] > 0
                    predicted += 1
                los = target['los']
                assert target['los_valid'] == (los is not None)
                if not row['calibrated']:
                    assert los is None
                if los:
                    x, y, z = los['unit_vector_camera']
                    assert all(math.isfinite(v) for v in (x, y, z)) and z > 0
                    assert abs(x*x + y*y + z*z - 1) < 1e-8
                    assert abs(los['azimuth_rad'] - math.atan2(x, z)) < 1e-8
                    assert abs(los['elevation_rad'] - math.atan2(-y, math.hypot(x, z))) < 1e-8
                targets += 1
            count += 1
    assert count > 0
    if expected_frames is not None:
        assert count == expected_frames, (count, expected_frames)
        assert previous_id == count - 1
    return {'frames': count, 'targets': targets, 'predicted_targets': predicted}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('jsonl')
    p.add_argument('--expected-frames', type=int)
    args = p.parse_args()
    print(check(args.jsonl, args.expected_frames))
