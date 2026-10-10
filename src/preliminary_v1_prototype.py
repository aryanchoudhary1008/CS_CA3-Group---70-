"""
CA-2 Cyber Security Case Study (Unit 4)
AI-Driven Zero Trust Prototype: Isolation Forest + rule-based baseline
Dataset: SYNTHETIC CLOUD USER ACTIVITY DATASET (generated here, not real data).

SAFETY / ETHICS: This script never connects to Snowflake or any real system.
All users, IPs, devices and sessions are simulated. The scenarios are
INSPIRED BY the identity-security problem identified in the Snowflake case
study (stolen credentials, accounts without MFA, access from untrusted
locations). It is NOT real Snowflake data.

Run:  python ca2_zero_trust_prototype.py
Needs: numpy, pandas, scikit-learn, matplotlib
"""
import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # save figures to files (no GUI needed)
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, accuracy_score, roc_auc_score,
                             average_precision_score, roc_curve)
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# DESIGN PARAMETERS (experimental choices, NOT scientifically optimal values)
# ----------------------------------------------------------------------
MAIN_SEED        = 42
N_USERS          = 60
TRAIN_PER_USER   = 150      # normal sessions per user for training
TEST_NORMAL_PER_USER = 60   # normal sessions per user in the test set
N_ANOMALIES      = 200      # injected anomalous sessions in the test set
THRESH_PERCENTILE = 97.5    # IF flag threshold = this percentile of TRAIN scores
IF_PARAMS = dict(n_estimators=200, max_samples=256,
                 contamination="auto", random_state=MAIN_SEED)

# Risk-score weights (sum = 1.0) -- design parameters
WEIGHTS = dict(anomaly=0.35, location=0.10, device=0.10, time=0.05,
               volume=0.15, auth=0.10, ip=0.15)
SENS_BONUS = {1: 0, 2: 5, 3: 10}       # extra risk points by resource sensitivity
# Zero Trust policy thresholds on 0-100 risk score -- design parameters
T_STEPUP, T_RESTRICT, T_BLOCK = 30, 50, 70

# Static baseline rules -- design parameters (no per-user baseline, no ML)
RULE_FAILED_LOGINS = 5
RULE_VOLUME_MB     = 1000
RULE_IP_REP        = 0.7

OUT_DIR = "outputs_ca2"
os.makedirs(OUT_DIR, exist_ok=True)

COUNTRIES = ["IN", "US", "GB", "DE", "SG", "AU", "BR", "CA"]
FEATURES = ["hour_dev", "vol_z", "res_ratio", "api_ratio", "is_new_country",
            "is_new_device", "ip_reputation", "failed_logins", "mfa_used",
            "session_duration", "resource_sensitivity"]

# ----------------------------------------------------------------------
# 1. SYNTHETIC CLOUD USER ACTIVITY DATASET
# ----------------------------------------------------------------------
def make_profiles(rng):
    """Each simulated user has a stable behavioural baseline."""
    rows = []
    for u in range(N_USERS):
        rows.append(dict(
            user_id=f"U{u:03d}",
            home_country=rng.choice(COUNTRIES, p=[.35,.2,.1,.1,.1,.05,.05,.05]),
            hour_mu=rng.uniform(8, 15),          # usual login hour (local)
            hour_sigma=rng.uniform(1.0, 2.0),
            base_vol_mb=float(np.exp(rng.normal(4.0, 0.8))),  # typical MB per session
            vol_sigma=rng.uniform(0.3, 0.6),     # log-scale spread
            res_mean=int(rng.integers(3, 15)),   # typical tables/objects touched
            api_mean=rng.uniform(5, 40),         # typical requests per minute
            mfa_enrolled=bool(rng.random() < 0.6),
            n_devices=int(rng.integers(1, 3)),
        ))
    return pd.DataFrame(rows)

def normal_session(rng, p):
    new_country = rng.random() < 0.03            # occasional legitimate travel
    new_device  = rng.random() < 0.03            # occasional new laptop
    country = rng.choice([c for c in COUNTRIES if c != p.home_country]) if new_country else p.home_country
    return dict(
        user_id=p.user_id, country=country, home_country=p.home_country,
        hour=float(rng.normal(p.hour_mu, p.hour_sigma) % 24),
        is_new_country=int(new_country), is_new_device=int(new_device),
        ip_reputation=float(rng.beta(1, 12)),    # mostly low (clean) values
        failed_logins=int(rng.poisson(0.3)),
        mfa_used=int(p.mfa_enrolled),
        session_duration=float(np.exp(rng.normal(3.4, 0.5))),   # minutes
        data_volume_mb=float(p.base_vol_mb * np.exp(rng.normal(0, p.vol_sigma))),
        resources_accessed=int(rng.poisson(p.res_mean) + 1),
        api_rate=float(max(1, rng.normal(p.api_mean, p.api_mean * 0.2))),
        resource_sensitivity=int(rng.choice([1, 2, 3], p=[.6, .3, .1])),
        label=0, scenario="normal")

def anomalous_session(rng, p, scenario):
    s = normal_session(rng, p)
    s["label"] = 1; s["scenario"] = scenario
    if scenario == "stolen_credential_login":      # valid creds, new location/device, no MFA
        s.update(is_new_country=1, is_new_device=1, ip_reputation=float(rng.beta(4, 3)),
                 hour=float(rng.uniform(0, 24)), mfa_used=0,
                 data_volume_mb=s["data_volume_mb"] * rng.uniform(1, 4))
        s["country"] = rng.choice([c for c in COUNTRIES if c != p.home_country])
    elif scenario == "bulk_exfiltration":          # very large data pull
        s.update(data_volume_mb=s["data_volume_mb"] * rng.uniform(15, 60),
                 resources_accessed=s["resources_accessed"] + int(rng.integers(10, 30)),
                 api_rate=s["api_rate"] * rng.uniform(3, 8),
                 resource_sensitivity=int(rng.choice([2, 3], p=[.3, .7])))
    elif scenario == "credential_validation":      # many failures then success
        s.update(failed_logins=int(rng.integers(6, 25)), is_new_country=1, is_new_device=1,
                 ip_reputation=float(rng.beta(5, 2)), mfa_used=0,
                 data_volume_mb=s["data_volume_mb"] * rng.uniform(0.5, 1.5))
        s["country"] = rng.choice([c for c in COUNTRIES if c != p.home_country])
    elif scenario == "low_and_slow":               # subtle: new device, modest increase
        s.update(is_new_device=1, ip_reputation=float(rng.beta(2, 6)),
                 data_volume_mb=s["data_volume_mb"] * rng.uniform(3, 6),
                 hour=float((p.hour_mu + rng.choice([-1, 1]) * rng.uniform(3, 5)) % 24))
    elif scenario == "stolen_login_plus_exfil":    # combined chain
        s.update(is_new_country=1, is_new_device=1, ip_reputation=float(rng.beta(4, 3)),
                 hour=float(rng.uniform(0, 24)), mfa_used=0,
                 data_volume_mb=s["data_volume_mb"] * rng.uniform(15, 60),
                 resources_accessed=s["resources_accessed"] + int(rng.integers(10, 30)),
                 api_rate=s["api_rate"] * rng.uniform(3, 8),
                 resource_sensitivity=3)
        s["country"] = rng.choice([c for c in COUNTRIES if c != p.home_country])
    return s

SCENARIOS = ["stolen_credential_login", "bulk_exfiltration", "credential_validation",
             "low_and_slow", "stolen_login_plus_exfil"]
NO_MFA_SCENARIOS = {"stolen_credential_login", "credential_validation", "stolen_login_plus_exfil"}

def generate_dataset(seed):
    rng = np.random.default_rng(seed)
    prof = make_profiles(rng)
    P = {r.user_id: r for r in prof.itertuples()}
    train = [normal_session(rng, p) for p in P.values() for _ in range(TRAIN_PER_USER)]
    test  = [normal_session(rng, p) for p in P.values() for _ in range(TEST_NORMAL_PER_USER)]
    no_mfa_users = [u for u, p in P.items() if not p.mfa_enrolled]
    all_users = list(P.keys())
    for i in range(N_ANOMALIES):
        sc = SCENARIOS[i % len(SCENARIOS)]
        pool = no_mfa_users if sc in NO_MFA_SCENARIOS else all_users
        test.append(anomalous_session(rng, P[rng.choice(pool)], sc))
    train_df = pd.DataFrame(train)
    test_df = pd.DataFrame(test).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train_df, test_df

# ----------------------------------------------------------------------
# 2. FEATURE ENGINEERING (per-user baselines learned from TRAIN only)
# ----------------------------------------------------------------------
def circ_dist(h, mu):
    d = np.abs(h - mu)
    return np.minimum(d, 24 - d)

def fit_user_baselines(train_df):
    b = {}
    for u, g in train_df.groupby("user_id"):
        ang = g["hour"].values / 24 * 2 * np.pi
        mu = (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2*np.pi)) / (2*np.pi) * 24
        lv = np.log1p(g["data_volume_mb"])
        b[u] = dict(hour_mu=mu, lv_mu=lv.mean(), lv_sd=max(lv.std(), 0.1),
                    res_mu=g["resources_accessed"].mean(), api_mu=g["api_rate"].mean())
    return b

def add_features(df, b):
    df = df.copy()
    df["hour_dev"]  = [circ_dist(h, b[u]["hour_mu"]) for h, u in zip(df.hour, df.user_id)]
    df["vol_z"]     = [(np.log1p(v) - b[u]["lv_mu"]) / b[u]["lv_sd"] for v, u in zip(df.data_volume_mb, df.user_id)]
    df["res_ratio"] = [r / b[u]["res_mu"] for r, u in zip(df.resources_accessed, df.user_id)]
    df["api_ratio"] = [a / b[u]["api_mu"] for a, u in zip(df.api_rate, df.user_id)]
    return df

def preprocess(df):
    """Basic checks. (Isolation Forest is scale-invariant, so no scaling is applied.)"""
    assert not df[FEATURES].isna().any().any(), "missing values found"
    df = df.copy()
    for c in ["vol_z", "res_ratio", "api_ratio"]:     # limit extreme outliers for plotting/risk maths
        df[c] = df[c].clip(-10, 100)
    return df

# ----------------------------------------------------------------------
# 3. BASELINE: static rules (no learning, no per-user context)
# ----------------------------------------------------------------------
def rule_baseline(df):
    r = ((df.failed_logins >= RULE_FAILED_LOGINS) |
         (df.is_new_country == 1) |
         (df.data_volume_mb > RULE_VOLUME_MB) |
         (df.ip_reputation > RULE_IP_REP))
    return r.astype(int).values

# ----------------------------------------------------------------------
# 4. RISK SCORE + ZERO TRUST POLICY
# ----------------------------------------------------------------------
def risk_scores(df, anom_norm):
    comp = dict(
        anomaly  = anom_norm,
        location = df.is_new_country.values.astype(float),
        device   = df.is_new_device.values.astype(float),
        time     = np.clip(df.hour_dev.values / 6.0, 0, 1),
        volume   = np.clip(df.vol_z.values / 4.0, 0, 1),
        auth     = np.clip(0.5*np.clip(df.failed_logins.values/5.0, 0, 1) + 0.5*(1 - df.mfa_used.values), 0, 1),
        ip       = np.clip(df.ip_reputation.values, 0, 1))
    key = {"anomaly":"anomaly","location":"location","device":"device","time":"time",
           "volume":"volume","auth":"auth","ip":"ip"}
    base = sum(WEIGHTS[k] * comp[key[k]] for k in WEIGHTS) * 100
    bonus = df.resource_sensitivity.map(SENS_BONUS).values
    return np.clip(base + bonus, 0, 100)

def zt_decision(score):
    if score >= T_BLOCK:    return "Block"
    if score >= T_RESTRICT: return "Restrict"
    if score >= T_STEPUP:   return "Step-up MFA"
    return "Allow"

def reasons(r):
    out = []
    if r.is_new_country: out.append("new country")
    if r.is_new_device: out.append("unrecognised device")
    if r.ip_reputation > 0.6: out.append(f"poor IP reputation ({r.ip_reputation:.2f})")
    if r.failed_logins >= 5: out.append(f"{int(r.failed_logins)} failed logins")
    if r.vol_z > 3: out.append(f"data volume {r.data_volume_mb:.0f} MB (z={r.vol_z:.1f} vs user baseline)")
    if r.res_ratio > 3: out.append(f"{r.res_ratio:.1f}x usual resources accessed")
    if r.api_ratio > 3: out.append(f"{r.api_ratio:.1f}x usual API rate")
    if r.hour_dev > 5: out.append(f"unusual hour (off by {r.hour_dev:.1f} h)")
    if r.mfa_used == 0: out.append("no MFA on session")
    return "; ".join(out) if out else "model anomaly only"

# ----------------------------------------------------------------------
# 5. METRICS
# ----------------------------------------------------------------------
def metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(TN=tn, FP=fp, FN=fn, TP=tp,
                accuracy=accuracy_score(y, pred),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0),
                FPR=fp / (fp + tn), FNR=fn / (fn + tp))

# ----------------------------------------------------------------------
# 6. FULL PIPELINE
# ----------------------------------------------------------------------
def run_pipeline(seed):
    train_df, test_df = generate_dataset(seed)
    b = fit_user_baselines(train_df)
    train_df = preprocess(add_features(train_df, b))
    test_df  = preprocess(add_features(test_df, b))

    params = dict(IF_PARAMS); params["random_state"] = seed
    iso = IsolationForest(**params).fit(train_df[FEATURES])      # trained on NORMAL data only
    s_train = -iso.score_samples(train_df[FEATURES])             # higher = more anomalous
    thr = np.percentile(s_train, THRESH_PERCENTILE)
    q50, q99 = np.percentile(s_train, 50), np.percentile(s_train, 99)

    t0 = time.perf_counter()
    s_test = -iso.score_samples(test_df[FEATURES])
    t_if = (time.perf_counter() - t0) / len(test_df)
    t0 = time.perf_counter()
    pred_rule = rule_baseline(test_df)
    t_rule = (time.perf_counter() - t0) / len(test_df)

    pred_if = (s_test >= thr).astype(int)
    anom_norm = np.clip((s_test - q50) / (q99 - q50), 0, 1)
    risk = risk_scores(test_df, anom_norm)
    decision = np.array([zt_decision(x) for x in risk])
    pred_zt = (decision != "Allow").astype(int)                      # any friction (step-up/restrict/block)
    pred_zt_hard = np.isin(decision, ["Restrict", "Block"]).astype(int)  # alert-level actions only

    test_df = test_df.assign(if_score=s_test, risk=risk, decision=decision,
                             pred_rule=pred_rule, pred_if=pred_if, pred_zt=pred_zt, pred_zt_hard=pred_zt_hard)
    return dict(train=train_df, test=test_df, thr=thr, t_if=t_if, t_rule=t_rule,
                iso=iso, s_train=s_train)

# ----------------------------------------------------------------------
# 7. MAIN: detailed run (seed 42) + robustness over several seeds
# ----------------------------------------------------------------------
def main():
    R = run_pipeline(MAIN_SEED)
    test, train = R["test"], R["train"]
    y = test["label"].values

    print("="*70)
    print("SYNTHETIC CLOUD USER ACTIVITY DATASET (simulated; NOT real Snowflake data)")
    print("="*70)
    print(f"Users: {N_USERS} | Train (normal only): {len(train)} | Test: {len(test)}")
    print(f"Test normal: {(y==0).sum()} | Test anomalous: {(y==1).sum()} ({y.mean()*100:.2f}%)")
    print("\nAnomaly scenarios in test set:\n", test[test.label==1].scenario.value_counts().to_string())
    print(f"\nIF threshold = {THRESH_PERCENTILE}th percentile of train scores = {R['thr']:.4f} (design parameter)")
    print(f"Risk thresholds (design parameters): step-up>={T_STEPUP}, restrict>={T_RESTRICT}, block>={T_BLOCK}")

    res = {"Rule-based baseline": metrics(y, test.pred_rule.values),
           "Isolation Forest": metrics(y, test.pred_if.values),
           "IF + risk/Zero Trust (non-Allow)": metrics(y, test.pred_zt.values),
           "IF + risk/Zero Trust (Restrict/Block only)": metrics(y, test.pred_zt_hard.values)}
    tab = pd.DataFrame(res).T
    print("\n--- Detection metrics on the test set ---")
    print(tab.round(4).to_string())
    tab.round(4).to_csv(f"{OUT_DIR}/metrics_main_run.csv")

    auc_if  = roc_auc_score(y, test.if_score);  ap_if  = average_precision_score(y, test.if_score)
    auc_rk  = roc_auc_score(y, test.risk);      ap_rk  = average_precision_score(y, test.risk)
    print(f"\nROC-AUC  IF score: {auc_if:.4f} | risk score: {auc_rk:.4f}")
    print(f"PR-AUC   IF score: {ap_if:.4f} | risk score: {ap_rk:.4f}")
    print("ROC-AUC / PR-AUC for the rule baseline: not evaluated (binary output, no score).")
    print(f"\nMean scoring time per session: IF = {R['t_if']*1e6:.2f} us, rules = {R['t_rule']*1e6:.2f} us")

    print("\n--- Detection rate (recall) per anomaly scenario ---")
    sc = (test[test.label==1].groupby("scenario")[["pred_rule","pred_if","pred_zt","pred_zt_hard"]].mean()
          .rename(columns={"pred_rule":"Rules","pred_if":"IsolationForest","pred_zt":"IF+Risk(non-Allow)","pred_zt_hard":"IF+Risk(Restrict/Block)"}))
    print(sc.round(3).to_string()); sc.round(4).to_csv(f"{OUT_DIR}/scenario_detection.csv")

    print("\n--- Zero Trust decisions by true class (counts) ---")
    order = ["Allow", "Step-up MFA", "Restrict", "Block"]
    ct = pd.crosstab(test.label.map({0:"normal",1:"anomalous"}), test.decision).reindex(columns=order, fill_value=0)
    print(ct.to_string()); ct.to_csv(f"{OUT_DIR}/decisions_by_class.csv")
    print("\nDecisions for anomalous sessions by scenario:")
    print(pd.crosstab(test[test.label==1].scenario, test[test.label==1].decision).reindex(columns=order, fill_value=0).to_string())

    print("\n--- Risk score summary by class ---")
    print(test.groupby("label").risk.describe().round(2).to_string())

    # Sensitivity of IF threshold (design parameter) -- reported, not tuned
    print("\n--- Sensitivity: IF threshold percentile (train scores) ---")
    rows = []
    for p in [90, 95, 97.5, 99, 99.5]:
        t = np.percentile(R["s_train"], p); pr = (test.if_score.values >= t).astype(int); m = metrics(y, pr)
        rows.append(dict(percentile=p, precision=m["precision"], recall=m["recall"], f1=m["f1"], FPR=m["FPR"]))
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # Example alerts
    print("\n--- Example security alerts (highest-risk sessions) ---")
    top = test[test.decision.isin(["Block", "Restrict"])].sort_values("risk", ascending=False).head(5)
    for _, r in top.iterrows():
        print(f"[ALERT] user={r.user_id} risk={r.risk:.1f} action={r.decision} "
              f"true_label={'anomalous('+r.scenario+')' if r.label else 'normal'} | {reasons(r)}")
    print("\n--- Example FALSE POSITIVE (normal session flagged by IF+risk) ---")
    fp = test[(test.label==0) & (test.pred_zt==1)].sort_values("risk", ascending=False).head(3)
    for _, r in fp.iterrows():
        print(f"[FP] user={r.user_id} risk={r.risk:.1f} action={r.decision} | {reasons(r)}")
    print("\n--- Example FALSE NEGATIVE (anomaly NOT restricted/blocked by IF+risk) ---")
    fn = test[(test.label==1) & (test.pred_zt_hard==0)].head(3)
    if fn.empty: print("(none: every anomalous session got Restrict or Block in this run)")
    for _, r in fn.iterrows():
        print(f"[FN] user={r.user_id} scenario={r.scenario} risk={r.risk:.1f} | {reasons(r)}")
    test.to_csv(f"{OUT_DIR}/test_results_with_decisions.csv", index=False)

    # ---------------- Graphs ----------------
    # 1 Confusion matrices
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, (name, key) in zip(ax, [("Rule baseline","pred_rule"),("Isolation Forest","pred_if"),("IF + risk (Restrict/Block)","pred_zt_hard")]):
        cm = confusion_matrix(y, test[key], labels=[0,1]); a.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2): a.text(j, i, cm[i,j], ha="center", va="center", fontsize=12)
        a.set_xticks([0,1]); a.set_yticks([0,1]); a.set_xticklabels(["Pred normal","Pred anomaly"]); a.set_yticklabels(["Actual normal","Actual anomaly"]); a.set_title(name)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig1_confusion_matrices.png", dpi=200); plt.close()
    # 2 IF score distribution
    plt.figure(figsize=(7,4))
    plt.hist(test[test.label==0].if_score, bins=50, alpha=.7, label="Normal"); plt.hist(test[test.label==1].if_score, bins=50, alpha=.7, label="Anomalous")
    plt.axvline(R["thr"], color="k", ls="--", label="IF threshold"); plt.xlabel("Isolation Forest anomaly score (higher = more anomalous)"); plt.ylabel("Sessions"); plt.legend(); plt.title("Anomaly score distribution")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig2_if_score_distribution.png", dpi=200); plt.close()
    # 3 Risk score distribution
    plt.figure(figsize=(7,4))
    plt.hist(test[test.label==0].risk, bins=50, alpha=.7, label="Normal"); plt.hist(test[test.label==1].risk, bins=50, alpha=.7, label="Anomalous")
    for t, n in [(T_STEPUP,"Step-up"),(T_RESTRICT,"Restrict"),(T_BLOCK,"Block")]: plt.axvline(t, color="k", ls=":"); plt.text(t, plt.ylim()[1]*.9, n, rotation=90, va="top", ha="right")
    plt.xlabel("Risk score (0-100)"); plt.ylabel("Sessions"); plt.yscale("log"); plt.legend(); plt.title("Risk score distribution (log y-axis)")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig3_risk_distribution.png", dpi=200); plt.close()
    # 4 Decisions by class
    ct.T.plot(kind="bar", figsize=(7,4)); plt.ylabel("Sessions"); plt.yscale("log"); plt.title("Zero Trust decisions by true class (log y-axis)"); plt.xticks(rotation=0)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig4_decisions_by_class.png", dpi=200); plt.close()
    # 5 Metric comparison
    tab[["precision","recall","f1","FPR"]].plot(kind="bar", figsize=(8,4)); plt.xticks(rotation=15); plt.title("Detection metrics comparison"); plt.ylim(0,1.05)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig5_metric_comparison.png", dpi=200); plt.close()
    # 6 ROC
    plt.figure(figsize=(5.5,5))
    for name, s in [("IF score", test.if_score), ("Risk score", test.risk)]:
        f, t, _ = roc_curve(y, s); plt.plot(f, t, label=name)
    plt.plot([0,1],[0,1],"k--", lw=.8); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend(); plt.title("ROC curves")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig6_roc.png", dpi=200); plt.close()
    # 7 Per-scenario detection
    sc.plot(kind="bar", figsize=(9,4)); plt.ylabel("Detection rate"); plt.xticks(rotation=20); plt.ylim(0,1.05); plt.title("Detection rate per anomaly scenario")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig7_scenario_detection.png", dpi=200); plt.close()

    # ---------------- Robustness: repeat with other seeds ----------------
    print("\n--- Robustness: 10 independent synthetic datasets (seeds 0-9) ---")
    rows = []
    for sd in range(10):
        Rr = run_pipeline(sd); yy = Rr["test"].label.values
        for name, key in [("Rule-based baseline","pred_rule"),("Isolation Forest","pred_if"),("IF + risk (non-Allow)","pred_zt"),("IF + risk (Restrict/Block)","pred_zt_hard")]:
            m = metrics(yy, Rr["test"][key].values)
            rows.append(dict(method=name, precision=m["precision"], recall=m["recall"], f1=m["f1"], FPR=m["FPR"]))
    rb = pd.DataFrame(rows).groupby("method").agg(["mean","std"]).round(4)
    print(rb.to_string()); rb.to_csv(f"{OUT_DIR}/robustness_10_seeds.csv")
    print(f"\nAll tables and figures saved to ./{OUT_DIR}/")

if __name__ == "__main__":
    main()
