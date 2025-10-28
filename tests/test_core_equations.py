import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from baines.core import Params, SizingInputs, design_impeller


class CoreEquationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = design_impeller(2.5, 0.0, 10.0, 5, Params(), SizingInputs())

    def test_finite_outputs(self) -> None:
        fields = ["U2A01", "PHI", "PSI", "THETA", "AM2R"]
        for row in self.rows:
            for field in fields:
                value = getattr(row, field)
                self.assertTrue(np.isfinite(value), msg=f"{field} is not finite")

    def test_matches_baseline_csv(self) -> None:
        cols = [
            "BETAB2",
            "B2R2",
            "U2A01",
            "AM2",
            "AM2R",
            "AM1",
            "AM1R",
            "AMR",
            "WR",
            "SS",
            "SSG",
            "THETA",
            "PHI",
            "PSI",
            "WND",
        ]
        df = pd.DataFrame([asdict(r) for r in self.rows])[cols]
        baseline_path = Path(__file__).resolve().parent / "data" / "pr2.5_baseline.csv"
        baseline = pd.read_csv(baseline_path)[cols]
        np.testing.assert_allclose(df.values, baseline.values, rtol=1e-6, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
