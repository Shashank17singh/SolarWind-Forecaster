import os
import sys
import tempfile
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from data_quality import compute_quality


def test_compute_quality_ok():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "omni.csv")
        times = [datetime.utcnow() - timedelta(minutes=i) for i in range(5)][::-1]
        df = pd.DataFrame(
            {
                "time": times,
                "bz_gsm": [0.1, -0.2, 0.0, 0.3, -0.1],
                "flow_speed": [400, 410, 405, 415, 420],
                "proton_density": [5, 6, 5.5, 6.2, 5.8],
            }
        )
        df.to_csv(path, index=False)
        result = compute_quality(path, window_min=10)
        assert result["ok"] is True
