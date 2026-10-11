"""CaseHOLD: TF-IDF logistic regression vs LightGBM on pairwise similarity + within-question ranking features."""
import io, json, os, re, sys, time
import numpy as np, pandas as pd, requests, scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier

SEED, N_TRAIN, N_TEST, K = 42, 4000, 1000, 5
HF = "https://huggingface.co/datasets/coastalcph/lex_glue/resolve/refs%2Fconvert%2Fparquet/case_hold/{}/0000.parquet"
API = "https://datasets-server.huggingface.co/parquet?dataset=coastalcph/lex_glue"
np.random.seed(SEED)
t0 = time.time()

def get(url):
    """GET with timeout and exponential backoff on 429/5xx."""
    for i in range(5):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                return r
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        except requests.RequestException:
            pass
        time.sleep(2 ** i)
    return None

def load(split, n):
    r = get(HF.format(split))
    if r is None:  # fallback: ask the datasets-server for the parquet file list
        files = get(API).json()["parquet_files"]
        url = [f["url"] for f in files if f["config"] == "case_hold" and f["split"] == split][0]
        r = get(url)
    df = pd.read_parquet(io.BytesIO(r.content))
    df = df.sample(n=min(n, len(df)), random_state=SEED).reset_index(drop=True)
    return df["context"].tolist(), [list(e) for e in df["endings"]], df["label"].to_numpy()

ctx_tr, end_tr, lab_tr = load("train", N_TRAIN)
ctx_te, end_te, lab_te = load("test", N_TEST)
print(f"loaded train={len(ctx_tr)} test={len(ctx_te)} questions ({time.time()-t0:.0f}s)")
if len(ctx_tr) < 1000 or len(ctx_te) < 500:
    sys.exit("not enough data")

# Shared vectorisers, fitted on training texts only
flat = lambda ends: [c for e in ends for c in e]
fit_txt = ctx_tr + flat(end_tr)
wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True).fit(fit_txt)
cvz = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=100000, sublinear_tf=True).fit(fit_txt)
tok = lambda s: re.findall(r"\w+", s.lower())

def rowcos(A, B, rep):
    """Cosine of each (context, candidate) pair (rows are L2-normalised)."""
    P = A[rep].multiply(B).tocsr()
    return P, np.asarray(P.sum(1)).ravel()

def build(ctx, ends):
    """Return sparse pair matrix (baseline) and dense ranking features (improved)."""
    N = len(ctx); rep = np.repeat(np.arange(N), K); cands = flat(ends)
    Wx, Wc = wv.transform(ctx), wv.transform(cands)
    P, wcos = rowcos(Wx, Wc, rep)
    Xbase = sp.hstack([Wc, P]).tocsr()  # candidate words + context*candidate word interactions
    _, ccos = rowcos(cvz.transform(ctx), cvz.transform(cands), rep)
    ts = [set(tok(c)) for c in ctx]; tc = [set(tok(c)) for c in cands]
    inter = np.array([len(ts[i] & tc[j]) for j, i in enumerate(rep)], float)
    clen = np.array([len(t) for t in tc], float)
    jac = inter / np.array([len(ts[i] | tc[j]) + 1 for j, i in enumerate(rep)])
    cov = inter / (clen + 1)
    base = {"wcos": wcos, "ccos": ccos, "jac": jac, "cov": cov, "clen": clen}
    cols = [v for v in base.values()] + [np.array([len(ts[i]) for i in rep], float)]
    for v in base.values():  # within-question rank, z-score, gap to question max
        m = v.reshape(N, K)
        rank = m.argsort(1).argsort(1)
        z = (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-9)
        cols += [rank.ravel(), z.ravel(), (m - m.max(1, keepdims=True)).ravel()]
    return Xbase, np.column_stack(cols)

Xb_tr, Xg_tr = build(ctx_tr, end_tr)
Xb_te, Xg_te = build(ctx_te, end_te)
print(f"features built ({time.time()-t0:.0f}s)")
pairy = lambda lab: (np.arange(K)[None, :] == lab[:, None]).ravel().astype(int)
y_tr, y_te = pairy(lab_tr), pairy(lab_te)

def make_lr(p, seed):
    return LogisticRegression(C=p["C"], max_iter=300, random_state=seed)
def make_gb(p, seed):
    return LGBMClassifier(**p, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          random_state=seed, n_jobs=-1, verbose=-1)

def scores(make, p, seed, Xa, ya, Xb):
    return make(p, seed).fit(Xa, ya).predict_proba(Xb)[:, 1].reshape(-1, K)

def top1(s, lab):
    """Per-question correctness of the argmax candidate."""
    return (s.argmax(1) == lab).astype(float)

def cv_select(make, grid, X, seed=SEED):
    """3-fold CV over training questions (all 5 candidates stay in the same fold)."""
    folds = list(KFold(3, shuffle=True, random_state=SEED).split(np.arange(len(lab_tr))))
    best, best_acc = None, -1
    for p in grid:
        accs = []
        for a, b in folds:
            ia, ib = (a[:, None] * K + np.arange(K)).ravel(), (b[:, None] * K + np.arange(K)).ravel()
            accs.append(top1(scores(make, p, seed, X[ia], y_tr[ia], X[ib]), lab_tr[b]).mean())
        print(f"  {p} cv_acc={np.mean(accs):.4f}")
        if np.mean(accs) > best_acc:
            best, best_acc = p, np.mean(accs)
    return best, best_acc

print("tuning baseline (CV on train)")
p_lr, cv_lr = cv_select(make_lr, [{"C": c} for c in (0.3, 3, 30)], Xb_tr)
print("tuning LightGBM (CV on train)")
grid_gb = [{"num_leaves": nl, "learning_rate": 0.05, "n_estimators": ne, "min_child_samples": 30}
           for nl in (7, 15) for ne in (100, 300)]
p_gb, cv_gb = cv_select(make_gb, grid_gb, Xg_tr)

# Final: fit on all training questions, evaluate once on the test set, seeds 42-44
def evaluate(make, p, Xa, Xb):
    corr, accs, f1s = [], [], []
    for seed in (42, 43, 44):
        s = scores(make, p, seed, Xa, y_tr, Xb)
        c = top1(s, lab_te)
        pred = (np.arange(K)[None, :] == s.argmax(1)[:, None]).ravel().astype(int)
        corr.append(c); accs.append(c.mean()); f1s.append(f1_score(y_te, pred, average="macro"))
    return np.mean(corr, 0), accs, f1s

c_b, a_b, f_b = evaluate(make_lr, p_lr, Xb_tr, Xb_te)
c_i, a_i, f_i = evaluate(make_gb, p_gb, Xg_tr, Xg_te)
rng = np.random.default_rng(SEED)
idx = rng.integers(0, len(c_b), (2000, len(c_b)))
diffs = (c_i[idx] - c_b[idx]).mean(1)
lo, hi = np.percentile(diffs, [2.5, 97.5])
bs, is_ = float(np.mean(a_b)), float(np.mean(a_i))
print(f"baseline acc={bs:.4f} improved acc={is_:.4f} diff CI=({lo:.4f},{hi:.4f})")
if not np.isfinite([bs, is_]).all():
    sys.exit("non-finite metrics")

os.makedirs("figures", exist_ok=True)
plt.bar(["TF-IDF LR", "LightGBM pairwise"], [bs, is_], yerr=[np.std(a_b), np.std(a_i)], color=["gray", "tab:blue"])
plt.axhline(1 / K, ls="--", c="k", lw=0.8); plt.ylabel("Top-1 accuracy (test)"); plt.title("CaseHOLD holding selection")
plt.savefig("figures/accuracy.png", dpi=120, bbox_inches="tight")

res = {
    "question": "Can a pairwise-similarity gradient-boosting model with within-question ranking features beat a standard TF-IDF logistic-regression classifier at recognizing which cited holding a judgement passage relies on (CaseHOLD)?",
    "method": "Subsampled CaseHOLD questions (5 candidates each); baseline TF-IDF LR on (context, candidate) pairs vs LightGBM on similarity/overlap/length features plus within-question rank and z-scores; hyperparameters chosen by 3-fold CV on train only; one test evaluation, seeds 42-44.",
    "dataset": {"name": "CaseHOLD (LexGLUE case_hold), random subsample", "source_url": "https://huggingface.co/datasets/coastalcph/lex_glue",
                "n_rows": len(ctx_tr) + len(ctx_te), "n_train": len(ctx_tr), "n_test": len(ctx_te),
                "same_as_paper": False, "baseline_is_papers_method": False},
    "metric_name": "top-1 accuracy of choosing the correct holding per question (test)",
    "higher_is_better": True,
    "baseline": f"TF-IDF word 1-2gram logistic regression on [candidate tfidf, context*candidate tfidf] pair features; best candidate per question; tuned {p_lr}",
    "improved": f"LightGBM on word/char TF-IDF cosine, token overlap, length features and per-question rank/z-score/gap-to-max; tuned {p_gb}",
    "metrics": {
        "baseline_score": bs, "improved_score": is_, "improvement": is_ - bs,
        "baseline_score_std": float(np.std(a_b)), "improved_score_std": float(np.std(a_i)),
        "baseline_f1": float(np.mean(f_b)), "improved_f1": float(np.mean(f_i)),
        "baseline_f1_std": float(np.std(f_b)), "improved_f1_std": float(np.std(f_i)),
        "diff_ci95_low": float(lo), "diff_ci95_high": float(hi),
        "baseline_cv_acc": float(cv_lr), "improved_cv_acc": float(cv_gb),
        "random_chance_acc": 1 / K, "runtime_seconds": time.time() - t0},
    "notes": "Stand-in for the paper's non-public corpus; tests recognising a cited principle, not cited facts. Official CaseHOLD train/test splits are randomly subsampled. TF-IDF vectorisers are fitted on training text only (unsupervised, also used inside CV). Macro-F1 is on pair labels (chosen candidate=1). Bootstrap CI resamples test questions using per-question correctness averaged over seeds. Similarity features are lexical, so absolute accuracy is expected to be far below transformer models.",
}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print(f"done in {time.time()-t0:.0f}s")
