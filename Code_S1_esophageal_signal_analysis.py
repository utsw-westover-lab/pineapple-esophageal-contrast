# -*- coding: utf-8 -*-
"""
Code S1. Analysis code for the quantitative comparison of esophageal signal.

Supplementary material to: "Positive Esophageal Contrast from Pineapple Juice with
Fat-Suppressed T2-Weighted MRI for MR-Guided Radiation Therapy: A Pilot Feasibility
Study" (Cancers).

This file contains the statistical analysis described in Sections 2.5-2.6:
floor normalization, the signal-to-floor ratio R_SF, Cliff's delta, the exact
permutation test, the family-wise correction over 585 threshold definitions
(one- and two-sided), leave-one-out, pixel bootstrap, subsampling to the smallest
contour, and the association with contour size. Its input is the set of pixel
intensities within each esophageal contour, extracted from the DICOM image series
and RT structure set as described in Section 2.5.

Input
-----
A CSV file in long format with one row per esophageal pixel and three columns:

    patient,group,intensity
    C1,contrast,143
    ...
    N5,no_contrast,61

where `group` is "contrast" or "no_contrast" and `intensity` is the stored pixel
value inside the esophageal contour. No intensity thresholding, smoothing or
exclusion is applied. Patient-level data are not publicly available (see the
Data Availability Statement); the code runs on any data in this format.

Usage
-----
    python Code_S1_esophageal_signal_analysis.py pixels.csv [--out results.xlsx]

Requirements: Python >= 3.10, NumPy, SciPy, pandas (openpyxl for --out).
"""

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy import stats

# ---- reported definition (Section 2.5) ------------------------------------
FLOOR_PCT = 2.0          # floor = 2nd percentile of the patient's own contour
DARK_CUT = 1.3           # floor-level: normalized intensity < 1.3
BRIGHT_CUT = 4.0         # signal-bearing: normalized intensity > 4.0

# ---- family of 585 candidate definitions (Section 2.6) ---------------------
FLOOR_GRID = (1.0, 2.0, 3.0)                                   # 3
DARK_GRID = np.round(np.arange(1.1, 2.55, 0.1), 2)             # 15: 1.1 ... 2.5
BRIGHT_GRID = (2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0,
               10.0, 12.0, 15.0, 20.0)                         # 13

# ---- robustness checks ------------------------------------------------------
SURV_GRID = np.logspace(0.0, np.log10(60.0), 240)  # thresholds for tail mass alone
N_BOOT, N_SUB, SEED = 2000, 500, 0
N_SPEARMAN_PERM = 100_000


def floor_of(v, pct=FLOOR_PCT):
    """Floor intensity: the given percentile (linear interpolation) of the contour."""
    return max(float(np.percentile(v, pct)), 1e-9)


def r_sf(v, dark=DARK_CUT, bright=BRIGHT_CUT, pct=FLOOR_PCT):
    """Signal-to-floor ratio: signal-bearing / floor-level pixels, 0.5 added to each."""
    f = floor_of(v, pct)
    n_signal = float((v > bright * f).sum())
    n_floor = float((v < dark * f).sum())
    return (n_signal + 0.5) / (n_floor + 0.5)


def cliffs_delta(a, b):
    """P(X > Y) - P(X < Y), ties counted as zero."""
    a = np.asarray(a, float)[:, None]
    b = np.asarray(b, float)[None, :]
    return float(np.sign(a - b).mean())


def exact_p(a, b, alternative="greater"):
    """Exact Mann-Whitney p-value (equivalent to the permutation test on ranks)."""
    return float(stats.mannwhitneyu(a, b, alternative=alternative, method="exact").pvalue)


def all_splits(lab):
    """Every assignment of the patients to groups of the observed sizes."""
    n, n1 = len(lab), int(lab.sum())
    return [np.isin(np.arange(n), idx).astype(int)
            for idx in itertools.combinations(range(n), n1)]


def family_grid(allv):
    """R_SF for every definition: array [floor pct, floor-level cut, signal cut, patient]."""
    return np.array([[[[r_sf(v, d, b, f) for v in allv]
                       for b in BRIGHT_GRID] for d in DARK_GRID] for f in FLOOR_GRID])


def family_deltas(grid, lab):
    a = grid[..., lab == 1][..., :, None]
    b = grid[..., lab == 0][..., None, :]
    return np.sign(a - b).mean(axis=(-2, -1))


def family_corrected_p(grid, lab):
    """Exact family-wise corrected p-values (max-statistic permutation test).

    One-sided: statistic = largest delta over all definitions.
    Two-sided: statistic = largest |delta| over all definitions.
    The null distribution recomputes the statistic for every group assignment.
    """
    splits = all_splits(lab)
    obs1 = family_deltas(grid, lab).max()
    obs2 = np.abs(family_deltas(grid, lab)).max()
    null1 = np.array([family_deltas(grid, l).max() for l in splits])
    null2 = np.array([np.abs(family_deltas(grid, l)).max() for l in splits])
    return dict(delta_max=float(obs1), n_assignments=len(splits),
                p_one_sided=float((null1 >= obs1 - 1e-12).mean()),
                p_two_sided=float((null2 >= obs2 - 1e-12).mean()))


def load(csv_path):
    df = pd.read_csv(csv_path)
    df["group"] = df["group"].str.lower().str.strip()
    out = []
    for group, flag in (("contrast", 1), ("no_contrast", 0)):
        for pid in sorted(df.loc[df.group == group, "patient"].unique()):
            v = df.loc[(df.group == group) & (df.patient == pid), "intensity"].to_numpy(float)
            out.append((pid, flag, v))
    return out


def analyze(patients):
    pids = [p for p, _, _ in patients]
    lab = np.array([g for _, g, _ in patients])
    allv = [v for _, _, v in patients]
    sizes = np.array([v.size for v in allv])

    # per-patient values at the reported definition (Table S1)
    rows = []
    for pid, g, v in patients:
        f = floor_of(v)
        rows.append(dict(patient=pid, group="contrast" if g else "no contrast",
                         contour_pixels=v.size, floor=f,
                         signal_bearing=int((v > BRIGHT_CUT * f).sum()),
                         floor_level=int((v < DARK_CUT * f).sum()),
                         pct_signal_bearing=100 * (v > BRIGHT_CUT * f).mean(),
                         pct_floor_level=100 * (v < DARK_CUT * f).mean(),
                         R_SF=r_sf(v)))
    per_patient = pd.DataFrame(rows)
    R = per_patient.R_SF.to_numpy()
    yes, no = lab == 1, lab == 0

    res = []

    def add(name, value, p=np.nan, lo=np.nan, hi=np.nan):
        res.append(dict(analysis=name, value=value, p=p, lo_2_5=lo, hi_97_5=hi))

    add("R_SF median, contrast", float(np.median(R[yes])))
    add("R_SF median, no contrast", float(np.median(R[no])))
    add("R_SF Cliff's delta, exact one-sided p (reported definition)",
        cliffs_delta(R[yes], R[no]), exact_p(R[yes], R[no]))

    grid = family_grid(allv)
    fam = family_corrected_p(grid, lab)
    add("Family-corrected, one-sided (%d definitions, %d assignments)"
        % (grid[..., 0].size, fam["n_assignments"]), fam["delta_max"], fam["p_one_sided"])
    add("Family-corrected, two-sided", fam["delta_max"], fam["p_two_sided"])
    d_all = family_deltas(grid, lab).ravel()
    add("Family: definitions favouring contrast (count)", int((d_all > 0).sum()))
    add("Family: delta median", float(np.median(d_all)), lo=float(d_all.min()), hi=float(d_all.max()))

    for i, pid in enumerate(pids):
        k = np.array([j for j in range(len(pids)) if j != i])
        f = family_corrected_p(grid[..., k], lab[k])
        add("Leave-one-out family-corrected one-sided, omitting %s" % pid,
            f["delta_max"], f["p_one_sided"])

    fl = per_patient.pct_floor_level.to_numpy()
    sb = per_patient.pct_signal_bearing.to_numpy()
    add("Floor-level %, Cliff's delta, exact two-sided p", cliffs_delta(fl[yes], fl[no]),
        exact_p(fl[yes], fl[no], "two-sided"))
    add("Signal-bearing %, Cliff's delta, exact two-sided p", cliffs_delta(sb[yes], sb[no]),
        exact_p(sb[yes], sb[no], "two-sided"))

    rng = np.random.default_rng(SEED)
    boot = np.array([cliffs_delta(*(lambda x: (x[yes], x[no]))(np.array(
        [r_sf(rng.choice(v, size=v.size, replace=True)) for v in allv])))
        for _ in range(N_BOOT)])
    n_min = int(sizes.min())
    sub = np.array([cliffs_delta(*(lambda x: (x[yes], x[no]))(np.array(
        [r_sf(rng.choice(v, size=n_min, replace=False)) for v in allv])))
        for _ in range(N_SUB)])
    add("Pixel bootstrap, mean delta (%d replicates)" % N_BOOT, float(boot.mean()),
        lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)))
    add("Subsampled to %d pixels, mean delta (%d replicates)" % (n_min, N_SUB),
        float(sub.mean()), lo=float(np.percentile(sub, 2.5)), hi=float(np.percentile(sub, 97.5)))

    rho = stats.spearmanr(R, sizes).statistic
    prng = np.random.default_rng(SEED)
    null = np.array([stats.spearmanr(R, prng.permutation(sizes)).statistic
                     for _ in range(N_SPEARMAN_PERM)])
    add("Spearman rho, R_SF vs contour size (two-sided permutation p)", float(rho),
        float((np.abs(null) >= abs(rho) - 1e-12).mean()))

    # normalized signal-bearing fraction alone, supremum over thresholds
    S = np.array([[float((v / floor_of(v) > t).mean()) for t in SURV_GRID] for v in allv])

    def sup(l):
        return float((S[l == 1].mean(0) - S[l == 0].mean(0)).max())

    obs = sup(lab)
    null = np.array([sup(l) for l in all_splits(lab)])
    add("Signal-bearing fraction alone, supremum over thresholds (one-sided)",
        obs, float((null >= obs - 1e-12).mean()))

    return per_patient, pd.DataFrame(res)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("csv")
    ap.add_argument("--out", help="optional .xlsx for the results")
    args = ap.parse_args()
    per_patient, results = analyze(load(args.csv))
    pd.set_option("display.width", 200)
    print(per_patient.to_string(index=False, float_format=lambda v: "%.2f" % v))
    print()
    print(results.to_string(index=False, float_format=lambda v: "%.3f" % v))
    if args.out:
        with pd.ExcelWriter(args.out) as xw:
            per_patient.to_excel(xw, sheet_name="Per patient", index=False)
            results.to_excel(xw, sheet_name="Results", index=False)


if __name__ == "__main__":
    main()
