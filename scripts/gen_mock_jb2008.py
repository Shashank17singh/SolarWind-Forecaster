"""Generate mock JB2008 data files: SOLFSMY.TXT, DTCFILE.TXT, SW-All.csv"""
import math
import random
import os
from datetime import datetime, timedelta

random.seed(42)

OUT = "data/indices/jb2008"
os.makedirs(OUT, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


print("Generating SOLFSMY.TXT ...")
lines = []
for year in range(2020, 2026):
    for doy in range(1, 366):
        try:
            datetime.strptime(f"{year}-{doy:03d}", "%Y-%j")
        except ValueError:
            continue
        f10     = round(120 + 30 * math.sin(doy / 365 * 2 * math.pi) + random.gauss(0, 5), 1)
        f10_81  = round(f10 - random.gauss(0, 2), 1)
        s10     = round(f10 * 0.95 + random.gauss(0, 3), 1)
        s10_81  = round(s10 - random.gauss(0, 2), 1)
        m10     = round(f10 * 0.92 + random.gauss(0, 3), 1)
        m10_81  = round(m10 - random.gauss(0, 2), 1)
        y10     = round(f10 * 0.88 + random.gauss(0, 3), 1)
        y10_81  = round(y10 - random.gauss(0, 2), 1)
        lines.append(f"{year} {doy:03d} 0 {f10} {f10_81} {s10} {s10_81} {m10} {m10_81} {y10} {y10_81} 0")

with open(os.path.join(OUT, "SOLFSMY.TXT"), "w") as f:
    f.write("\n".join(lines))
print(f"  SOLFSMY.TXT: {len(lines)} rows")


print("Generating DTCFILE.TXT ...")
dtc_lines = []
dt = datetime(2020, 1, 1)
end = datetime(2026, 1, 1)
while dt < end:
    doy = dt.timetuple().tm_yday
    year = dt.year
    
    vals = []
    for h in range(24):
        base = 150 + 60 * math.sin(doy / 365 * 2 * math.pi)
        dtc = round(base + random.gauss(0, 15), 2)
        vals.append(f"{dtc:8.2f}")
    dtc_lines.append(f"DTC {year} {doy:03d} {''.join(vals)}")
    dt += timedelta(days=1)

with open(os.path.join(OUT, "DTCFILE.TXT"), "w") as f:
    f.write("\n".join(dtc_lines))
print(f"  DTCFILE.TXT: {len(dtc_lines)} rows")


print("Generating SW-All.csv ...")
header = "DATE,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,F10.7_OBS,F10.7_ADJ,F10.7_OBS_AVG,F10.7_ADJ_AVG"
rows = [header]
dt = datetime(2020, 1, 1)
end = datetime(2026, 1, 1)
while dt < end:
    doy = dt.timetuple().tm_yday
    f10 = round(120 + 30 * math.sin(doy / 365 * 2 * math.pi) + random.gauss(0, 5), 1)
    kp_vals = [round(max(0, min(9, random.gauss(2.5, 1.2))), 1) for _ in range(8)]
    ap_vals = [round(max(0, 4 * (2 ** kp)), 0) for kp in kp_vals]
    kp_str = ",".join(str(k) for k in kp_vals)
    ap_str = ",".join(str(a) for a in ap_vals)
    rows.append(f"{dt.strftime('%Y%m%d')},{kp_str},{ap_str},{f10},{f10},{f10},{f10}")
    dt += timedelta(days=1)

with open(os.path.join(OUT, "SW-All.csv"), "w") as f:
    f.write("\n".join(rows))
print(f"  SW-All.csv: {len(rows)-1} rows")
print("All mock JB2008 files generated.")
