"""Counts the residual wide-spread rows quoted in Appendix B.

Reads only the stored per-day feature files, so it needs no kernel state and
runs in a few seconds. It rebuilds the predictive sample the same way the
notebook does, by dropping the final bin of each session, which has no forward
markout, and then counts rows whose quoted spread exceeds two dollars.
"""
import glob
import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_START, TEST_END = "2025-05-01", "2025-08-29"

files = sorted(glob.glob(os.path.join(ROOT, "Data", "Processed", "Databento", "lob_*.parquet")))
sel = [f for f in files if TEST_START <= os.path.basename(f)[4:14] <= TEST_END]
lob = pd.concat([pd.read_parquet(f) for f in sel])

lob["mid_next"] = lob.groupby("date")["mid"].shift(-1)
pred = lob.dropna(subset=["mid_next"])

wide = pred[pred["spread"] > 2.0]
print(f"test-window sessions            : {len(sel)}")
print(f"predictive sample rows          : {len(pred):,}")
print(f"rows with spread above $2       : {len(wide)}")
print(f"their times of day              : {[str(t)[11:19] for t in wide.index]}")
print(f"all before noon                 : {all(t.hour < 12 for t in wide.index)}")
