"""
Second pass over the raw Databento ticks: quote-based order-flow imbalance
(Cont, Kukanov and Stoikov, 2014) plus gross trade volume and quote
intensity on the 10-second grid.

Motivation: the corrected pipeline showed signed trade flow dominates the
10-second signal. CKS show the QUOTE-based order-flow imbalance at the best
bid and ask is the canonical microstructure driver of short-horizon price
changes, and it is the one major LOB feature the original pipeline did not
build. This script is resume-safe (skips existing per-day outputs) and can
be relaunched any number of times until all 494 days are done.

Output: Data/Processed/Databento_OFI/ofi_YYYY-MM-DD.parquet with columns
  ofi          level-1 order-flow imbalance summed per 10s bin
  gross_vol    total traded size per bin
  n_trades     trade count per bin
  n_quotes     book-update count per bin (quote intensity)
"""
import glob
import os
import sys
import warnings
from datetime import time as dtime, date as ddate

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import databento as db

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

ROOT    = os.path.expanduser(r'~\Desktop\Imperial College London\MSc Risk Management & Financial Engineering\Applied Project\Applied Project Analysis')
DBN_DIR = os.path.join(ROOT, 'Data', 'Raw', 'Databento')
OUT_DIR = os.path.join(ROOT, 'Data', 'Processed', 'Databento_OFI')
os.makedirs(OUT_DIR, exist_ok=True)

EX_DIV = {'2024-03-15', '2024-06-21', '2024-09-20', '2024-12-20',
          '2025-03-21', '2025-06-20', '2025-09-19', '2025-12-19'}
nyse = mcal.get_calendar('NYSE')
sched = nyse.schedule(start_date='2024-01-02', end_date='2025-12-31')
trading_days = set(sched.index.date)
# trim sessions to the ACTUAL market close (13:00 on half-days, the synthetic
# EQUS.MINI BBO carries abnormal one-sided quotes after early closes)
close_map = {d.date(): c.tz_convert('America/New_York').time()
             for d, c in sched['market_close'].items()}

files = sorted(glob.glob(os.path.join(DBN_DIR, '*.dbn.zst')))
print(f'{len(files)} raw files', flush=True)

done = skipped = 0
for f in files:
    date_str = os.path.basename(f).split('.')[0].split('-')[-1]
    d = ddate(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:]))
    out = os.path.join(OUT_DIR, f'ofi_{d}.parquet')
    if os.path.exists(out):
        done += 1
        continue
    if d not in trading_days or str(d) in EX_DIV:
        skipped += 1
        continue

    df = db.DBNStore.from_file(f).to_df()
    df['ts_et'] = df['ts_event'].dt.tz_convert('America/New_York')
    t_col = df['ts_et'].dt.time
    df = df[(t_col >= dtime(9, 30)) & (t_col < close_map.get(d, dtime(16, 0)))]

    q = df[df['action'] != 'T'].set_index('ts_et').sort_index()
    t = df[df['action'] == 'T'].set_index('ts_et').sort_index()

    pb, pa = q['bid_px_00'].values, q['ask_px_00'].values
    qb = q['bid_sz_00'].values.astype(np.int64)
    qa = q['ask_sz_00'].values.astype(np.int64)
    pb1, pa1 = np.roll(pb, 1), np.roll(pa, 1)
    qb1, qa1 = np.roll(qb, 1), np.roll(qa, 1)
    # Cont-Kukanov-Stoikov level-1 OFI event contribution
    e = ((pb >= pb1) * qb - (pb <= pb1) * qb1
         - (pa <= pa1) * qa + (pa >= pa1) * qa1).astype(np.float64)
    e[0] = 0.0
    ofi = pd.Series(e, index=q.index).resample('10s', label='left', closed='left').sum()
    n_quotes = pd.Series(1.0, index=q.index).resample('10s', label='left', closed='left').sum()

    if len(t):
        gross = t['size'].resample('10s', label='left', closed='left').sum()
        ntr   = pd.Series(1.0, index=t.index).resample('10s', label='left', closed='left').sum()
    else:
        gross = pd.Series(dtype=float)
        ntr   = pd.Series(dtype=float)

    res = pd.DataFrame({'ofi': ofi, 'n_quotes': n_quotes})
    res['gross_vol'] = gross.reindex(res.index).fillna(0.0)
    res['n_trades']  = ntr.reindex(res.index).fillna(0.0)
    res = res[(res.index.time >= dtime(9, 30)) & (res.index.time < dtime(16, 0))]
    res.to_parquet(out)
    done += 1
    if done % 25 == 0:
        print(f'  {done} days done...', flush=True)

print(f'OFI pass complete: {done} done, {skipped} skipped', flush=True)
