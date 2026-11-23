# Reproducing the MIT-BIH PVC Experiment

Biomedical Signal Processing | Fall 2026 | Huaqiao University

This project tests whether adding RR intervals to ECG waveform features improves beat-level classification of normal beats (N) and premature ventricular contractions (V) on held-out subjects. The experiment uses Python and runs on a CPU.

## 1. Set up Python

Open a terminal in the folder containing `experiment.py` and `requirements.txt`. The reference run uses Python 3.13. Create a virtual environment and install the pinned dependencies.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS or Linux, use `python3 -m venv .venv` to create the environment, then use `.venv/bin/python` in place of the Windows interpreter path.

## 2. Get the data

Open the [PhysioNet release page](https://physionet.org/content/mitdb/1.0.0/) and choose **Download the ZIP file**. Extract release 1.0.0. The `mit-bih-arrhythmia-database-1.0.0` directory contains `RECORDS` and files such as `100.hea`, `100.dat`, and `100.atr`. The example below assumes that you saved this directory as `data/mitdb`.

The release has 48 two-channel ECG recordings from 47 subjects. The experiment selects the modified limb lead II (MLII) by name, loads the associated signal and reference beat annotations, and records any exclusions. Obtain the source data separately from PhysioNet.

## 3. Run the experiment

```powershell
.venv\Scripts\python.exe experiment.py --data-dir data/mitdb --output-dir results
```

Pass the extracted folder to `--data-dir` and a writable folder to `--output-dir`. Quote either path if it contains spaces. The script saves the figures and evaluation files in the output folder.

<!-- pagebreak -->

## What the experiment does

1. **Preprocess and segment.** Read the 360 Hz signal and apply a Butterworth 0.5-40 Hz band-pass filter forward and backward, with the order parameter set to 4. Cut each beat from 200 ms before to 400 ms after the reference annotation.
2. **Extract features.** Summarize each segment by 18 window means, peak-to-peak amplitude, RMS, mean absolute amplitude, and mean absolute successive difference. Compute the previous RR, next RR, and their ratio from neighboring annotated beats before selecting N and V.
3. **Train and compare.** Keep records 201 and 202 in the same subject group. Use a seeded held-out split by subject and train a class-weighted random forest with waveform features, RR features, and both together. The same subjects and eligible beats are used in all comparisons.

## What to check after running

- **`metrics.json`** contains the run configuration, subject split, class counts, confusion matrices, and metrics for all feature sets. Confirm that the training and testing subject lists have no overlap.
- **`metrics.csv`** gives one row per feature set. Compare PVC F1 with precision, the proportion of predicted PVCs that are correct, and recall, the proportion of annotated PVCs that are detected.
- **The PNG figures** show filtering, example beats, model comparison, the combined model's confusion matrix, and example mistakes.

The `reference_results/` folder in the source package is a saved full run. The `results/` folder in the example command is a fresh run, so it can be checked against the reference.

## Troubleshooting and scope

- If Python cannot import a dependency, run the installation command with the same virtual-environment interpreter used for the experiment.
- If the input files are missing, point `--data-dir` at the extracted folder that directly contains `RECORDS` and the numbered files.
- This is an offline experiment. Reference annotations provide the beat locations, and the next RR interval requires a subsequent beat. The evaluation covers the two selected beat labels in one database.

## Data citation

Moody GB and Mark RG. MIT-BIH Arrhythmia Database, version 1.0.0. PhysioNet. [doi:10.13026/C2F305](https://doi.org/10.13026/C2F305)
