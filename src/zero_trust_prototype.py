"""
CA-2 Cyber Security Case Study -- Prototype v2 (post-audit revision)
Dataset: SYNTHETIC CLOUD USER ACTIVITY DATASET (simulated; NOT real Snowflake data).
No real system is contacted. Scenarios are INSPIRED BY the identity-security
problem in the Snowflake case study (stolen credentials, accounts without MFA,
access from untrusted locations).

Changes from v1 (documented in the report; none were tuned to improve scores):
 C1  Anomaly "stealth" parameter + difficulty sweep (anomalies no longer always
     carry every strong indicator at once).
 C2  Threshold / score-normalisation calibrated on a HELD-OUT normal validation
     split (not on the same sessions the Isolation Forest was fitted on).
 C3  Stronger baseline B: per-user rules (same engineered features, no ML).
 C4  Ablation: Isolation Forest WITHOUT per-user (personalised) features.
 C5  Statistics: bootstrap 95% CIs, recall at fixed FPR, prevalence analysis,
     20 seeds with t-based 95% CIs and paired differences.
 C6  Zero Trust threshold sensitivity (+/-10 shift) and count of normal blocks.
Note: because RNG consumption and the train/validation split changed, v2
numbers are NOT expected to reproduce v1 numbers, even at stealth = 0.

Run: python zero_trust_prototype.py            (full run, 20 seeds, as in the report)
     python zero_trust_prototype.py --seeds 3   (quick demo)
Needs: numpy, pandas, scikit-learn, scipy, matplotlib
"""
import os, time, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (confusion_matrix, roc_auc_score, average_precision_score, roc_curve)
warnings.filterwarnings("ignore")

# ---------------- DESIGN PARAMETERS (experimental, NOT optimal) ----------------
MAIN_SEED, MAIN_STEALTH = 42, 0.5          # fixed BEFORE running; 0.5 = "moderate" stealth
STEALTH_LEVELS = [0.0, 0.25, 0.5, 0.75]    # all levels are reported (no cherry-picking)
SEEDS = list(range(20))
N_USERS, TRAIN_PER_USER, FIT_PER_USER = 60, 150, 100   # 100 fit + 50 validation per user
TEST_NORMAL_PER_USER, N_ANOMALIES = 60, 200
THRESH_PERCENTILE = 97.5                    # = target FPR budget of ~2.5% on VALIDATION normals
WEIGHTS = dict(anomaly=0.35, location=0.10, device=0.10, time=0.05, volume=0.15, auth=0.10, ip=0.15)
SENS_BONUS = {1: 0, 2: 5, 3: 10}
T_STEPUP, T_RESTRICT, T_BLOCK = 30, 50, 70
RULE_FAILED_LOGINS, RULE_VOLUME_MB, RULE_IP_REP = 5, 1000, 0.7      # baseline A (global rules)
OUT_DIR = "outputs_ca2_v2"; os.makedirs(OUT_DIR, exist_ok=True)

COUNTRIES = ["IN", "US", "GB", "DE", "SG", "AU", "BR", "CA"]
FEATURES = ["hour_dev", "vol_z", "res_ratio", "api_ratio", "is_new_country", "is_new_device",
            "ip_reputation", "failed_logins", "mfa_used", "session_duration", "resource_sensitivity"]
FEATURES_GLOBAL = ["hour_sin", "hour_cos", "log_vol", "resources_accessed", "api_rate",
                   "is_new_country", "is_new_device", "ip_reputation", "failed_logins",
                   "mfa_used", "session_duration", "resource_sensitivity"]
INDICATORS = ["is_new_country", "is_new_device", "ip_reputation", "failed_logins",
              "data_volume_mb", "resources_accessed", "api_rate", "hour"]

# ---------------- 1. Synthetic Cloud User Activity Dataset ----------------
def make_profiles(rng):
    rows = []
    for u in range(N_USERS):
        rows.append(dict(user_id=f"U{u:03d}",
            home_country=rng.choice(COUNTRIES, p=[.35,.2,.1,.1,.1,.05,.05,.05]),
            hour_mu=rng.uniform(8, 15), hour_sigma=rng.uniform(1.0, 2.0),
            base_vol_mb=float(np.exp(rng.normal(4.0, 0.8))), vol_sigma=rng.uniform(0.3, 0.6),
            res_mean=int(rng.integers(3, 15)), api_mean=rng.uniform(5, 40),
            mfa_enrolled=bool(rng.random() < 0.6)))
    return pd.DataFrame(rows)

def normal_session(rng, p):
    new_country, new_device = rng.random() < 0.03, rng.random() < 0.03
    country = rng.choice([c for c in COUNTRIES if c != p.home_country]) if new_country else p.home_country
    return dict(user_id=p.user_id, country=country, home_country=p.home_country,
        hour=float(rng.normal(p.hour_mu, p.hour_sigma) % 24),
        is_new_country=int(new_country), is_new_device=int(new_device),
        ip_reputation=float(rng.beta(1, 12)), failed_logins=int(rng.poisson(0.3)),
        mfa_used=int(p.mfa_enrolled), session_duration=float(np.exp(rng.normal(3.4, 0.5))),
        data_volume_mb=float(p.base_vol_mb * np.exp(rng.normal(0, p.vol_sigma))),
        resources_accessed=int(rng.poisson(p.res_mean) + 1),
        api_rate=float(max(1, rng.normal(p.api_mean, p.api_mean * 0.2))),
        resource_sensitivity=int(rng.choice([1, 2, 3], p=[.6, .3, .1])), label=0, scenario="normal")

def anomalous_session(rng, p, scenario, stealth):
    n = normal_session(rng, p); s = dict(n); s["label"] = 1; s["scenario"] = scenario
    other = lambda: rng.choice([c for c in COUNTRIES if c != p.home_country])
    if scenario == "stolen_credential_login":
        s.update(is_new_country=1, is_new_device=1, ip_reputation=float(rng.beta(4, 3)),
                 hour=float(rng.uniform(0, 24)), mfa_used=0,
                 data_volume_mb=n["data_volume_mb"] * rng.uniform(1, 4), country=other())
    elif scenario == "bulk_exfiltration":
        s.update(data_volume_mb=n["data_volume_mb"] * rng.uniform(15, 60),
                 resources_accessed=n["resources_accessed"] + int(rng.integers(10, 30)),
                 api_rate=n["api_rate"] * rng.uniform(3, 8),
                 resource_sensitivity=int(rng.choice([2, 3], p=[.3, .7])))
    elif scenario == "credential_validation":
        s.update(failed_logins=int(rng.integers(6, 25)), is_new_country=1, is_new_device=1,
                 ip_reputation=float(rng.beta(5, 2)), mfa_used=0,
                 data_volume_mb=n["data_volume_mb"] * rng.uniform(0.5, 1.5), country=other())
    elif scenario == "low_and_slow":
        s.update(is_new_device=1, ip_reputation=float(rng.beta(2, 6)),
                 data_volume_mb=n["data_volume_mb"] * rng.uniform(3, 6),
                 hour=float((p.hour_mu + rng.choice([-1, 1]) * rng.uniform(3, 5)) % 24))
    elif scenario == "stolen_login_plus_exfil":
        s.update(is_new_country=1, is_new_device=1, ip_reputation=float(rng.beta(4, 3)),
                 hour=float(rng.uniform(0, 24)), mfa_used=0,
                 data_volume_mb=n["data_volume_mb"] * rng.uniform(15, 60),
                 resources_accessed=n["resources_accessed"] + int(rng.integers(10, 30)),
                 api_rate=n["api_rate"] * rng.uniform(3, 8), resource_sensitivity=3, country=other())
    if stealth > 0:            # C1: each indicator independently reverts to "normal-looking" with prob = stealth
        for k in INDICATORS:
            if rng.random() < stealth:
                s[k] = n[k]
                if k == "is_new_country": s["country"] = n["country"]
    return s

SCENARIOS = ["stolen_credential_login", "bulk_exfiltration", "credential_validation",
             "low_and_slow", "stolen_login_plus_exfil"]
NO_MFA_SCENARIOS = {"stolen_credential_login", "credential_validation", "stolen_login_plus_exfil"}

def generate_dataset(seed, stealth):
    rng = np.random.default_rng(seed)
    P = {r.user_id: r for r in make_profiles(rng).itertuples()}
    train = [normal_session(rng, p) for p in P.values() for _ in range(TRAIN_PER_USER)]
    test = [normal_session(rng, p) for p in P.values() for _ in range(TEST_NORMAL_PER_USER)]
    no_mfa = [u for u, p in P.items() if not p.mfa_enrolled]; allu = list(P)
    for i in range(N_ANOMALIES):
        sc = SCENARIOS[i % 5]
        test.append(anomalous_session(rng, P[rng.choice(no_mfa if sc in NO_MFA_SCENARIOS else allu)], sc, stealth))
    train_df = pd.DataFrame(train); train_df["k"] = train_df.groupby("user_id").cumcount()
    return train_df, pd.DataFrame(test).sample(frac=1, random_state=seed).reset_index(drop=True)

# ---------------- 2. Feature engineering (baselines from FIT split only) ----------------
def circ_dist(h, mu):
    d = np.abs(h - mu); return np.minimum(d, 24 - d)

def fit_user_baselines(df):
    b = {}
    for u, g in df.groupby("user_id"):
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
    df["hour_sin"], df["hour_cos"] = np.sin(df.hour / 24 * 2*np.pi), np.cos(df.hour / 24 * 2*np.pi)  # global (non-personal) features
    df["log_vol"] = np.log1p(df.data_volume_mb)
    for c in ["vol_z", "res_ratio", "api_ratio"]: df[c] = df[c].clip(-10, 100)
    assert not df[FEATURES + FEATURES_GLOBAL].isna().any().any()
    return df

# ---------------- 3. Baselines ----------------
def rules_A(df):   # global static rules, no per-user context
    return ((df.failed_logins >= RULE_FAILED_LOGINS) | (df.is_new_country == 1) |
            (df.data_volume_mb > RULE_VOLUME_MB) | (df.ip_reputation > RULE_IP_REP)).astype(int).values

def rules_B(df):   # C3: per-user rules using the SAME engineered features, no ML (design parameters)
    return ((df.failed_logins >= 5) | (df.vol_z > 3) | (df.res_ratio > 3) | (df.api_ratio > 3) |
            ((df.is_new_country == 1) & (df.is_new_device == 1)) | (df.ip_reputation > 0.7)).astype(int).values

# ---------------- 4. Risk score + Zero Trust policy ----------------
def risk_scores(df, anom_norm, sens_bonus=SENS_BONUS):
    comp = dict(anomaly=anom_norm, location=df.is_new_country.values.astype(float),
        device=df.is_new_device.values.astype(float), time=np.clip(df.hour_dev.values / 6.0, 0, 1),
        volume=np.clip(df.vol_z.values / 4.0, 0, 1),
        auth=np.clip(0.5*np.clip(df.failed_logins.values/5.0, 0, 1) + 0.5*(1 - df.mfa_used.values), 0, 1),
        ip=np.clip(df.ip_reputation.values, 0, 1))
    return np.clip(sum(WEIGHTS[k] * comp[k] for k in WEIGHTS) * 100 + df.resource_sensitivity.map(sens_bonus).values, 0, 100)

def zt_decision(x, shift=0):
    if x >= T_BLOCK + shift: return "Block"
    if x >= T_RESTRICT + shift: return "Restrict"
    if x >= T_STEPUP + shift: return "Step-up MFA"
    return "Allow"

def reasons(r):
    out = []
    if r.is_new_country: out.append("new country")
    if r.is_new_device: out.append("unrecognised device")
    if r.ip_reputation > 0.6: out.append(f"poor IP reputation ({r.ip_reputation:.2f})")
    if r.failed_logins >= 5: out.append(f"{int(r.failed_logins)} failed logins")
    if r.vol_z > 3: out.append(f"data volume {r.data_volume_mb:.0f} MB (z={r.vol_z:.1f} vs user baseline)")
    if r.res_ratio > 3: out.append(f"{r.res_ratio:.1f}x usual resources")
    if r.api_ratio > 3: out.append(f"{r.api_ratio:.1f}x usual API rate")
    if r.hour_dev > 5: out.append(f"unusual hour (off by {r.hour_dev:.1f} h)")
    if r.mfa_used == 0: out.append("no MFA on session")
    return "; ".join(out) if out else "model anomaly only"

# ---------------- 5. Metrics ----------------
def metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn)
    return dict(TN=tn, FP=fp, FN=fn, TP=tp, accuracy=(tn+tp)/len(y), precision=pr, recall=rc,
                f1=(2*pr*rc/(pr+rc) if pr+rc else 0.0), FPR=fp/(fp+tn), FNR=fn/(fn+tp))

def bootstrap_ci(y, preds, B=1000, seed=0):
    rng = np.random.default_rng(seed); n = len(y); rows = {k: [] for k in preds}
    for _ in range(B):
        idx = rng.integers(0, n, n); yy = y[idx]
        for k, p in preds.items():
            pp = p[idx]; tp = ((yy==1)&(pp==1)).sum(); fp = ((yy==0)&(pp==1)).sum()
            fn = ((yy==1)&(pp==0)).sum(); tn = ((yy==0)&(pp==0)).sum()
            pr = tp/(tp+fp) if tp+fp else 0.0; rc = tp/(tp+fn) if tp+fn else 0.0
            rows[k].append([pr, rc, 2*pr*rc/(pr+rc) if pr+rc else 0.0, fp/(fp+tn) if fp+tn else 0.0])
    out = []
    for k, v in rows.items():
        v = np.array(v); lo, hi = np.percentile(v, 2.5, axis=0), np.percentile(v, 97.5, axis=0)
        out.append(dict(method=k, **{f"{m}_95CI": f"[{l:.3f}, {h:.3f}]" for m, l, h in zip(["precision","recall","f1","FPR"], lo, hi)}))
    return pd.DataFrame(out)

def recall_at_fpr(y, score, target):
    f, t, _ = roc_curve(y, score); return t[f <= target].max()

# ---------------- 6. Pipeline ----------------
def run_pipeline(seed, stealth):
    tr, te = generate_dataset(seed, stealth)
    fit_df, cal_df = tr[tr.k < FIT_PER_USER], tr[tr.k >= FIT_PER_USER]     # C2: held-out normal validation split
    b = fit_user_baselines(fit_df)
    fit_f, cal_f, te_f = add_features(fit_df, b), add_features(cal_df, b), add_features(te, b)
    out = {}
    for tag, feats in [("if", FEATURES), ("ifg", FEATURES_GLOBAL)]:
        iso = IsolationForest(n_estimators=200, max_samples=256, contamination="auto", random_state=seed).fit(fit_f[feats])
        s_cal = -iso.score_samples(cal_f[feats])
        t0 = time.perf_counter(); s_te = -iso.score_samples(te_f[feats]); tm = (time.perf_counter()-t0)/len(te_f)
        out[tag] = dict(score=s_te, thr=np.percentile(s_cal, THRESH_PERCENTILE),
                        q50=np.percentile(s_cal, 50), q99=np.percentile(s_cal, 99), time=tm)
    te_f["if_score"], te_f["ifg_score"] = out["if"]["score"], out["ifg"]["score"]
    te_f["pred_ruleA"], te_f["pred_ruleB"] = rules_A(te_f), rules_B(te_f)
    te_f["pred_ifg"] = (out["ifg"]["score"] >= out["ifg"]["thr"]).astype(int)
    te_f["pred_if"] = (out["if"]["score"] >= out["if"]["thr"]).astype(int)
    anom_norm = np.clip((out["if"]["score"] - out["if"]["q50"]) / (out["if"]["q99"] - out["if"]["q50"]), 0, 1)
    te_f["risk"] = risk_scores(te_f, anom_norm)
    te_f["decision"] = [zt_decision(x) for x in te_f.risk]
    te_f["pred_zt"] = (te_f.decision != "Allow").astype(int)
    te_f["pred_zt_hard"] = te_f.decision.isin(["Restrict", "Block"]).astype(int)
    return dict(test=te_f, out=out, cal_scores=None)

METHODS = {"Rules A (global static)": "pred_ruleA", "Rules B (per-user, no ML)": "pred_ruleB",
           "IF without per-user features (ablation)": "pred_ifg", "Isolation Forest": "pred_if",
           "IF + Risk/ZT (non-Allow) [friction]": "pred_zt", "IF + Risk/ZT (Restrict/Block) [alert]": "pred_zt_hard"}

# ---------------- 7. Main ----------------
def main():
    R = run_pipeline(MAIN_SEED, MAIN_STEALTH); te = R["test"]; y = te.label.values
    print("="*72); print("SYNTHETIC CLOUD USER ACTIVITY DATASET (simulated; NOT real Snowflake data)"); print("="*72)
    print(f"seed={MAIN_SEED}, stealth={MAIN_STEALTH} | users={N_USERS} | fit/user={FIT_PER_USER}, validation/user={TRAIN_PER_USER-FIT_PER_USER}")
    print(f"Test: {len(te)} sessions | normal={(y==0).sum()} | anomalous={(y==1).sum()} ({y.mean()*100:.2f}%)")
    print(f"Trivial 'predict all normal' accuracy = {(y==0).mean():.4f}  (why accuracy is not emphasised)")
    print(f"IF threshold = {THRESH_PERCENTILE}th pct of VALIDATION-normal scores = {R['out']['if']['thr']:.4f}")
    tab = pd.DataFrame({k: metrics(y, te[v].values) for k, v in METHODS.items()}).T
    print("\n--- Main-run metrics (single dataset; see bootstrap CIs and multi-seed results) ---")
    print(tab.round(4).to_string()); tab.round(4).to_csv(f"{OUT_DIR}/metrics_main_run.csv")
    print("\n--- Bootstrap 95% CIs over test sessions (1000 resamples) ---")
    ci = bootstrap_ci(y, {k: te[v].values for k, v in METHODS.items()}); print(ci.to_string(index=False)); ci.to_csv(f"{OUT_DIR}/bootstrap_ci.csv", index=False)
    print("\n--- Score-based metrics ---")
    for nm, sc in [("IF score", te.if_score), ("IF-no-personal score", te.ifg_score), ("Risk score", te.risk)]:
        print(f"{nm:22s} ROC-AUC={roc_auc_score(y, sc):.4f}  PR-AUC={average_precision_score(y, sc):.4f}  (prevalence baseline PR-AUC={y.mean():.4f})")
        print("   recall @ FPR 1% / 2.5% / 5%: " + " / ".join(f"{recall_at_fpr(y, sc, t):.3f}" for t in (0.01, 0.025, 0.05)))
    print("Rule baselines output binary decisions: ROC-AUC / PR-AUC not evaluated.")
    print(f"\nScoring time/session: IF={R['out']['if']['time']*1e6:.1f} us")

    print("\n--- Recall per scenario ---")
    sc = te[te.label==1].groupby("scenario")[[v for v in METHODS.values()]].mean(); sc.columns = list(METHODS.keys())
    print(sc.round(3).to_string()); sc.round(4).to_csv(f"{OUT_DIR}/scenario_recall.csv")

    order = ["Allow", "Step-up MFA", "Restrict", "Block"]
    ct = pd.crosstab(te.label.map({0:"normal",1:"anomalous"}), te.decision).reindex(columns=order, fill_value=0)
    print("\n--- Zero Trust decisions by true class ---"); print(ct.to_string()); ct.to_csv(f"{OUT_DIR}/decisions_by_class.csv")
    print("\n--- Risk score summary by class ---"); print(te.groupby("label").risk.describe().round(2).to_string())

    print("\n--- Prevalence analysis (analytic: precision = r*p / (r*p + FPR*(1-p)); assumes recall & FPR unchanged) ---")
    rows = []
    for k in ["Rules A (global static)", "Isolation Forest", "IF + Risk/ZT (Restrict/Block) [alert]"]:
        m = metrics(y, te[METHODS[k]].values)
        rows.append({"method": k, **{f"prev={p}": round(m["recall"]*p/(m["recall"]*p + m["FPR"]*(1-p)), 4) for p in (0.0526, 0.01, 0.001, 0.0001)}})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n--- IF threshold sensitivity (reported, NOT used to choose the threshold) ---")
    s_te = R["out"]["if"]["score"]; rows = []
    # percentiles of the held-out validation scores are not stored; use the test-normal quantile only for reporting FPR targets
    for t_fpr in [0.10, 0.05, 0.025, 0.01, 0.005]:
        thr = np.quantile(s_te[y == 0], 1 - t_fpr); m = metrics(y, (s_te >= thr).astype(int))
        rows.append(dict(target_FPR=t_fpr, precision=m["precision"], recall=m["recall"], f1=m["f1"], FPR=m["FPR"]))
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\n--- Zero Trust threshold sensitivity (all three thresholds shifted) ---")
    rows = []
    for sh in (-10, 0, 10):
        dec = np.array([zt_decision(x, sh) for x in te.risk]); alert = np.isin(dec, ["Restrict", "Block"]).astype(int); m = metrics(y, alert)
        rows.append(dict(shift=sh, precision=m["precision"], recall=m["recall"], f1=m["f1"], FPR=m["FPR"],
                         normal_blocked=int(((dec=="Block") & (y==0)).sum()), normal_stepup=int(((dec=="Step-up MFA") & (y==0)).sum())))
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\n--- Example alerts / FP / FN (IF + Risk, Restrict/Block) ---")
    for _, r in te[te.decision.isin(["Block","Restrict"])].sort_values("risk", ascending=False).head(4).iterrows():
        print(f"[ALERT] user={r.user_id} risk={r.risk:.1f} action={r.decision} true={'anomalous('+r.scenario+')' if r.label else 'normal'} | {reasons(r)}")
    for _, r in te[(te.label==0) & (te.pred_zt_hard==1)].sort_values("risk", ascending=False).head(3).iterrows():
        print(f"[FP] user={r.user_id} risk={r.risk:.1f} action={r.decision} | {reasons(r)}")
    fn = te[(te.label==1) & (te.pred_zt_hard==0)]
    print(f"[FN count] {len(fn)}")
    for _, r in fn.head(3).iterrows():
        print(f"[FN] user={r.user_id} scenario={r.scenario} risk={r.risk:.1f} action={r.decision} | {reasons(r)}")
    te.to_csv(f"{OUT_DIR}/test_results_main_run.csv", index=False)

    # ---- figures ----
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, (nm, key) in zip(ax, [("Rules A","pred_ruleA"),("Isolation Forest","pred_if"),("IF+Risk (Restrict/Block)","pred_zt_hard")]):
        cm = confusion_matrix(y, te[key], labels=[0,1]); a.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2): a.text(j, i, cm[i,j], ha="center", va="center", fontsize=12)
        a.set_xticks([0,1]); a.set_yticks([0,1]); a.set_xticklabels(["Pred normal","Pred anomaly"]); a.set_yticklabels(["Actual normal","Actual anomaly"]); a.set_title(nm)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig1_confusion_matrices.png", dpi=200); plt.close()
    for fname, col, xl, ttl in [("fig2_if_score_distribution.png","if_score","IF anomaly score","Anomaly score distribution"),
                                ("fig3_risk_distribution.png","risk","Risk score (0-100)","Risk score distribution (log y)")]:
        plt.figure(figsize=(7,4)); plt.hist(te[te.label==0][col], bins=50, alpha=.7, label="Normal"); plt.hist(te[te.label==1][col], bins=50, alpha=.7, label="Anomalous")
        if col == "if_score": plt.axvline(R["out"]["if"]["thr"], color="k", ls="--", label="threshold")
        else:
            plt.yscale("log")
            for t, n in [(T_STEPUP,"Step-up"),(T_RESTRICT,"Restrict"),(T_BLOCK,"Block")]: plt.axvline(t, color="k", ls=":")
        plt.xlabel(xl); plt.ylabel("Sessions"); plt.legend(); plt.title(ttl); plt.tight_layout(); plt.savefig(f"{OUT_DIR}/{fname}", dpi=200); plt.close()
    ct.T.plot(kind="bar", figsize=(7,4)); plt.yscale("log"); plt.xticks(rotation=0); plt.ylabel("Sessions"); plt.title("Zero Trust decisions by true class (log y)")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig4_decisions_by_class.png", dpi=200); plt.close()
    plt.figure(figsize=(5.5,5))
    for nm, s in [("IF score", te.if_score), ("IF no personal feats", te.ifg_score), ("Risk score", te.risk)]:
        f, t, _ = roc_curve(y, s); plt.plot(f, t, label=nm)
    plt.plot([0,1],[0,1],"k--", lw=.8); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.title("ROC curves"); plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig5_roc.png", dpi=200); plt.close()
    sc.plot(kind="bar", figsize=(11,4.5)); plt.xticks(rotation=20); plt.ylabel("Recall"); plt.ylim(0,1.05); plt.title("Recall per anomaly scenario"); plt.legend(fontsize=6)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig6_scenario_recall.png", dpi=200); plt.close()

    # ---- multi-seed x stealth sweep ----
    print(f"\n--- {len(SEEDS)} seeds x stealth levels: mean, SD, 95% t-CI of the mean ---")
    recs = []
    for st in STEALTH_LEVELS:
        for sd in SEEDS:
            Rr = run_pipeline(sd, st); t_ = Rr["test"]; yy = t_.label.values
            for nm, key in METHODS.items():
                m = metrics(yy, t_[key].values)
                recs.append(dict(stealth=st, seed=sd, method=nm, precision=m["precision"], recall=m["recall"], f1=m["f1"], FPR=m["FPR"]))
            recs.append(dict(stealth=st, seed=sd, method="_normal_blocked", precision=0, recall=0, f1=0, FPR=0,
                             nblock=int(((t_.decision=="Block") & (yy==0)).sum())))
    D = pd.DataFrame(recs); D.to_csv(f"{OUT_DIR}/multiseed_raw.csv", index=False)
    n = len(SEEDS); tcrit = stats.t.ppf(0.975, n-1); rows = []
    for (st, nm), g in D[D.method != "_normal_blocked"].groupby(["stealth", "method"]):
        r = {"stealth": st, "method": nm}
        for m in ["precision", "recall", "f1", "FPR"]:
            mu, sd_ = g[m].mean(), g[m].std(ddof=1); r[m] = f"{mu:.3f} ± {sd_:.3f}"; r[m+"_95CI"] = f"[{mu - tcrit*sd_/np.sqrt(n):.3f}, {mu + tcrit*sd_/np.sqrt(n):.3f}]" if m == "f1" else ""
        rows.append(r)
    S = pd.DataFrame(rows)[["stealth","method","precision","recall","f1","f1_95CI","FPR"]]
    print(S.to_string(index=False)); S.to_csv(f"{OUT_DIR}/multiseed_summary.csv", index=False)

    print("\n--- Paired per-seed F1 differences (same datasets) ---")
    pairs = [("IF + Risk/ZT (Restrict/Block) [alert]", "Rules A (global static)"), ("Isolation Forest", "Rules A (global static)"),
             ("Isolation Forest", "Rules B (per-user, no ML)"), ("Isolation Forest", "IF without per-user features (ablation)"),
             ("IF + Risk/ZT (Restrict/Block) [alert]", "Isolation Forest")]
    for st in STEALTH_LEVELS:
        for a, b_ in pairs:
            da = D[(D.stealth==st) & (D.method==a)].set_index("seed").f1; db = D[(D.stealth==st) & (D.method==b_)].set_index("seed").f1
            d = (da - db); print(f"stealth={st:<4} F1[{a}] - F1[{b_}]: mean={d.mean():+.4f} SD={d.std(ddof=1):.4f} seeds_positive={(d>0).sum()}/{n}")
    nb = D[D.method == "_normal_blocked"].groupby("stealth").nblock.agg(["mean", "max"])
    print("\nNormal sessions BLOCKED per test set (mean, max over seeds):"); print(nb.to_string())

    plt.figure(figsize=(7,4.5))
    for nm in ["Rules A (global static)", "Rules B (per-user, no ML)", "Isolation Forest", "IF + Risk/ZT (Restrict/Block) [alert]"]:
        g = D[D.method == nm].groupby("stealth").f1.mean(); plt.plot(g.index, g.values, marker="o", label=nm)
    plt.xlabel("Anomaly stealth (prob. each indicator looks normal)"); plt.ylabel("Mean F1 over seeds"); plt.legend(fontsize=7); plt.title("Detection under increasing attacker stealth")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig7_stealth_sweep.png", dpi=200); plt.close()
    print(f"\nAll tables/figures saved to ./{OUT_DIR}/")

if __name__ == "__main__":
    import argparse, sys
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    ap = argparse.ArgumentParser(description="Synthetic-data Zero Trust prototype (Isolation Forest + rule baselines).")
    ap.add_argument("--seeds", type=int, default=20, help="number of independent synthetic datasets for the robustness runs (default 20, as in the report)")
    args = ap.parse_args()
    SEEDS[:] = list(range(args.seeds))
    t0 = time.time(); main(); print(f"Total runtime: {time.time()-t0:.0f} s")
