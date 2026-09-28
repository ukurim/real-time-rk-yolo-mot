#!/usr/bin/env python3
"""Compare actual per-frame latency, never infer latency from FPS.

Each run replays the same file in realtime with latest-frame selection. No
calibration is fabricated: without --calibration the benchmark passes --no-los.
The reported source latency is a replay-clock estimate, not sensor exposure age.
"""
import argparse
import json
import pathlib
import subprocess


def percentile(values, q):
    values = sorted(values)
    return values[int((len(values) - 1) * q)] if values else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', default='build/Aerial_detection_demo')
    p.add_argument('--config', default='config/video.yaml')
    p.add_argument('--frames', type=int, default=300)
    p.add_argument('--output-dir', default='build/benchmarks')
    p.add_argument('--calibration')
    p.add_argument('--model')
    p.add_argument('--input-mode', choices=['uint8', 'fp16_normalized'])
    p.add_argument('--decoder', choices=['auto', 'mpp'])
    p.add_argument('--cv-threads', type=int, help='OpenCV CPU thread count, 1..8')
    p.add_argument('--copy-input', action='store_true', help='copy selected decoder frames to cached CPU memory')
    p.add_argument('--cases', nargs='+', choices=['one_auto', 'one_all_cores', 'two_instances', 'three_instances'])
    p.add_argument('--warmup', type=int, default=10, help='discard first N output records for statistics')
    args = p.parse_args()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for workers, core_mask, name in ((1, 0, 'one_auto'), (1, 7, 'one_all_cores'),
                                    (2, 0, 'two_instances'), (3, 0, 'three_instances')):
        if args.cases and name not in args.cases:
            continue
        dest = out / (name + '.jsonl')
        cmd = [args.binary, '--config', args.config, '--workers', str(workers), '--core-mask', str(core_mask), '--realtime',
               '--max-frames', str(args.frames), '--output', str(dest)]
        cmd += ['--calibration', args.calibration] if args.calibration else ['--no-los']
        if args.model:
            cmd += ['--model', args.model]
        if args.input_mode:
            cmd += ['--input-mode', args.input_mode]
        if args.decoder:
            cmd += ['--decoder', args.decoder]
        if args.cv_threads is not None:
            cmd += ['--cv-threads', str(args.cv_threads)]
        if args.copy_input:
            cmd += ['--copy-input']
        with (out / (name + '.log')).open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = [json.loads(line) for line in dest.read_text().splitlines()]
        ids = [row['frame_id'] for row in rows]
        if any(a >= b for a, b in zip(ids, ids[1:])):
            raise RuntimeError('Non-monotonic results in %s' % dest)
        timings = rows[args.warmup:]
        if not timings:
            raise RuntimeError('Not enough results after warmup; increase --frames: %s' % dest)
        summary = {'name': name, 'workers': workers, 'core_mask': core_mask,
                   'command': cmd,
                   'output_frames': len(rows), 'measured_frames': len(timings),
                   'calibrated': bool(args.calibration), 'sensor_capture_latency_measured': False}
        for key in ('receiver_to_result', 'source_to_result_estimate', 'queue', 'completion_queue', 'preprocess',
                    'inference', 'input_submit', 'rknn_run', 'output_get', 'postprocess', 'tracking',
                    'tracking_gmc', 'tracking_association', 'los'):
            values = [r['latency_ms'][key] for r in timings if r['latency_ms'][key] is not None]
            summary[key] = {name: percentile(values, q) for name, q in [('p50', .5), ('p95', .95), ('p99', .99)]}
        wait = [(r['publish_started_monotonic_ns'] - r['result_ready_monotonic_ns']) / 1e6 for r in timings]
        summary['publisher_wait'] = {name: percentile(wait, q) for name, q in [('p50', .5), ('p95', .95), ('p99', .99)]}
        summaries.append(summary)
        print(json.dumps(summary, indent=2))
    (out / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')


if __name__ == '__main__':
    main()
