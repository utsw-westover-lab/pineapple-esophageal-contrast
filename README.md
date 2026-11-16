# Positive esophageal contrast from pineapple juice: signal analysis code

Analysis code (Code S1) for the quantitative comparison of esophageal signal in:

> Positive Esophageal Contrast from Pineapple Juice with Fat-Suppressed T2-Weighted MRI
> for MR-Guided Radiation Therapy: A Pilot Feasibility Study. *Cancers* (under review).

## What it does

`Code_S1_esophageal_signal_analysis.py` implements the statistical analysis described in
Sections 2.5–2.6 of the paper:

- floor normalization and the signal-to-floor ratio R_SF
- Cliff's delta and the exact permutation test
- family-wise correction (max-statistic) over 585 threshold definitions, one- and two-sided
- leave-one-out, pixel bootstrap, and subsampling to the smallest contour
- association between R_SF and contour size (Spearman, permutation p)

Image extraction (DICOM series and RT structure sets) is not included; the input is the
set of pixel intensities inside each esophageal contour.

## Input

A CSV in long format, one row per esophageal pixel:

```
patient,group,intensity
C1,contrast,143
...
N5,no_contrast,61
```

`group` is `contrast` or `no_contrast`; `intensity` is the stored pixel value inside the
contour. No thresholding, smoothing or exclusion is applied.

## Usage

```
pip install -r requirements.txt
python Code_S1_esophageal_signal_analysis.py pixels.csv --out results.xlsx
```

Requires Python 3.10 or later. Values that depend on random resampling (bootstrap,
subsampling, Spearman permutation p) use a fixed seed but can differ in the last digit
depending on patient order and library versions.

## Data availability

Patient-level data are not publicly available because of patient privacy considerations.
They are available on request from the corresponding author, as described in the paper's
Data Availability Statement. The code runs on any data in the format above.

## License

MIT; see `LICENSE`.
