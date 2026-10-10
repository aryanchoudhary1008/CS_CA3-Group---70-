# AI-Driven Zero Trust Prototype: Detecting Credential Misuse (Snowflake Case Study)

**Course:** Cyber Security, Theory CA-2 (Part 2) – Case Study Implementation
**Institute:** Symbiosis Institute of Technology, Pune
**Group number:** 70
**Members:** Aryan Choudhary (23070122052), Arsh Ansari (23070122047), Aliraza Shaikh (23070122023), Yash Patel (23070122235)

**GitHub repository:** https://github.com/aryanchoudhary1008/CS_CA3-Group---70-  

> **Important – synthetic data only.** This project never connects to Snowflake or any real system.
> All users, sessions, IP reputations and attacks are **simulated** by the code (the "Synthetic Cloud User Activity Dataset").
> The scenarios are inspired by the identity-security problem in the 2024 Snowflake customer-account incidents
> (stolen customer credentials, accounts without MFA), but nothing here is real Snowflake data and the results
> are **not** real-world performance.

## 1. What the project does
1. Generates a synthetic dataset: 60 users with normal behaviour profiles; 200 injected attack sessions in 5 scenarios
   (stolen credential login, bulk exfiltration, credential validation, low-and-slow, stolen login + exfiltration) with an attacker-**stealth** parameter.
2. Builds per-user baselines (from the fit split only) and 11 behavioural features.
3. Trains an **Isolation Forest** on normal sessions only; the threshold is the 97.5th percentile of scores on a held-out normal calibration split.
4. Compares it with **Rules A** (global static rules), **Rules B** (per-user rules, no ML) and an **Isolation Forest without per-user features** (ablation).
5. Computes a weighted **risk score (0-100)** and applies a **Zero Trust policy**: Allow (<30), Step-up MFA (30-49), Restrict (50-69), Block (>=70).
   Restrict/Block are simulated labels; nothing is actually blocked.
6. Evaluates with precision, recall, F1, FPR, ROC-AUC, PR-AUC, recall at fixed FPR, per-scenario recall, bootstrap confidence intervals,
   a stealth sweep (0, 0.25, 0.5, 0.75) over 20 independent datasets, threshold-sensitivity and prevalence analyses, and example security alerts.

## 2. Folder structure
```
CA2_Snowflake_ZeroTrust_Implementation/
├── README.md
├── requirements.txt          Python packages
├── run_project.bat           Windows: one-click setup + run
├── run_project.sh            Linux/macOS: one-click setup + run
├── build_exe.bat             Windows: builds ZeroTrustPrototype.exe (see section 6)
├── src/
│   ├── zero_trust_prototype.py        MAIN program (version 2, used for all results in the report)
│   └── preliminary_v1_prototype.py    first version (kept for transparency; see report Section 4.5)
├── sample_outputs/           the actual CSV files and figures produced by our run of the main program
└── dist/                     place the built .exe here
```

## 3. Requirements
- Python 3.10 or newer (we used 3.13)
- Packages: numpy, pandas, scikit-learn, scipy, matplotlib (`pip install -r requirements.txt`)
- About 200 MB disk space; no internet is needed after installing packages; no GPU.

## 4. How to run
**Windows (easiest):** double-click `run_project.bat` (or run it in Command Prompt).
**Linux / macOS:** `chmod +x run_project.sh && ./run_project.sh`

**Manual steps (any OS):**
```
python -m venv .venv
.venv\Scripts\activate            (Windows)   |   source .venv/bin/activate   (Linux/macOS)
pip install -r requirements.txt
python src/zero_trust_prototype.py            # full run: 20 datasets x 4 stealth levels (about 2-4 minutes)
python src/zero_trust_prototype.py --seeds 3  # quick demo (about 15 seconds)
```
Optional: `python src/preliminary_v1_prototype.py` runs the first version (writes to `outputs_ca2/`).

## 5. Output
Running the main program prints all tables to the console and creates the folder **`outputs_ca2_v2/`** containing:

| File | Content |
|---|---|
| `metrics_main_run.csv` | confusion counts, precision, recall, F1, FPR for the main run (seed 42, stealth 0.5) |
| `bootstrap_ci.csv` | 95% bootstrap intervals (1,000 resamples) |
| `scenario_recall.csv` | recall per attack scenario |
| `decisions_by_class.csv` | Allow / Step-up MFA / Restrict / Block counts for normal and attack sessions |
| `multiseed_raw.csv`, `multiseed_summary.csv` | per-dataset and summarised results for all stealth levels |
| `test_results_main_run.csv` | every test session with its scores and decision |
| `fig1` ... `fig7` (`.png`) | confusion matrices, score and risk distributions, decisions, ROC, scenario recall, stealth sweep |

The `sample_outputs/` folder holds the files from our own full run. The numbers in the report (Section 5) come from these files.
The main-run results are reproducible: the random seed is fixed (42). Small differences (about 0.001) can appear in the multi-dataset
tables between machines or library versions.

## 6. Building the Windows executable (.exe)
An `.exe` can only be built on Windows. On a Windows PC with Python installed, double-click `build_exe.bat`.
It installs PyInstaller and creates `dist\ZeroTrustPrototype.exe`, which runs the same program
(`ZeroTrustPrototype.exe` for the full run, `ZeroTrustPrototype.exe --seeds 3` for the quick demo; results appear in `outputs_ca2_v2` in the current folder).
Note: a one-file build that bundles numpy, pandas, scikit-learn, scipy and matplotlib can be large (often 100 MB or more).
If it is too big for the submission portal, upload it to the GitHub *Releases* page (or a drive) and link it here instead.

## 7. Troubleshooting
- `ModuleNotFoundError`: run `pip install -r requirements.txt` inside the activated virtual environment.
- `python` not found on Windows: install Python from python.org and tick "Add Python to PATH", or use `py` instead of `python`.
- The program is silent for a while: the full run trains many models; use `--seeds 3` to check quickly.
- No figure windows appear: that is intended; figures are saved as PNG files in `outputs_ca2_v2/`.

## 8. Ethics and limitations
Simulation only; no real systems or data were used. Detection results measure behaviour on scenarios we designed
(per-user rules may benefit from our knowledge of the generator). Restrict/Block are simulated labels. Drift, cold-start users and
real telemetry were not evaluated. See the report (Sections 5 and 6) for details.
