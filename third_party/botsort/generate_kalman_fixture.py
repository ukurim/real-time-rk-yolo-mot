#!/usr/bin/env python3
"""Optional verification only; requires numpy/scipy, never a runtime dependency.

Execute the unmodified, vendored upstream Kalman filter and print the fixture
used by tests/test_tracker.cpp::upstream_kalman_fixture.
"""
import importlib.util
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode = True
source = Path(__file__).parent / "upstream" / "kalman_filter.py"
spec = importlib.util.spec_from_file_location("upstream_kalman", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
kf = module.KalmanFilter()
mean, covariance = kf.initiate(np.array([25., 40., 50., 40.]))
for measurement in ([29., 40., 50., 40.], [33., 40., 50., 40.], [37., 42., 54., 42.]):
    mean, covariance = kf.predict(mean, covariance)
    mean, covariance = kf.update(mean, covariance, np.array(measurement))
    print(list(np.r_[mean[:2] - mean[2:4] / 2, mean[2:4]]))
mean, covariance = kf.predict(mean, covariance)
print(list(np.r_[mean[:2] - mean[2:4] / 2, mean[2:4]]))
