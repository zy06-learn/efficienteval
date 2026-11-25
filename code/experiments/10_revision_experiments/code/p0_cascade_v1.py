#!/usr/bin/env python3
"""The strongest cascade we can build on this pool, against the router's own frontier.

Three things the paper's single cascade operating point does not establish:
  1. the cascade is made to start from the cheapest action; every starting action is tried here
  2. its trigger threshold is picked under a 1.5x latency ceiling; the whole threshold grid is
     swept here, with no ceiling, so the cascade gets a frontier rather than a point
  3. its learned trigger reads the cheap features; here it also reads the cheap verifier's own
     calibrated score, which is the union of both information sets

The router is swept over its own beta grid, so the two families are compared frontier to
frontier rather than point to point.
"""
from __future__ import annotations
import itertools, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v3core as V           # noqa: E402
import core                  # noqa: E402

ART = HERE / "artifacts"
C0 = json.loads((ART / "part1_main_pooled_v1" / "00_contract" / "FROZEN_v3.json").read_text())
HPS = json.loads((ART / "part1c_main_full_v1" / "00_contract" / "HP_SELECTED.json").read_text())
POOL, FEATURES = list(C0["pool"]), list(C0["features"])
HP_B, GRID, SEEDS = dict(HPS["hyperparameters_B"]), tuple(C0["beta_grid"]), list(V.SEEDS)
NICE = {"factcc": "FactCC", "lettuce_v2": "Lettuce-v2", "granite_guardian_3_1_2b": "Granite-3.1-2B"}
OUT = HERE / "results_p0exp"; OUT.mkdir(exist_ok=True)


def au(y, p):
    return float(roc_auc_score(y, p)) if len(set(np.asarray(y).tolist())) > 1 else float("nan")


def stage2(p_val, y_val, p_ev):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_val, y_val)
    return np.clip(iso.predict(p_ev), 1e-6, 1 - 1e-6)


def pre_ms(f):
    return (f["feature_latency_ms"].to_numpy(float) + f["compact16_feature_latency_ms"].to_numpy(float))


def lat_matrix(f):
    return np.column_stack([f[f"latency_ms__{a}"].fillna(0).to_numpy(float) for a in POOL])


class Fitted:
    """Everything both families share, fitted once per seed."""

    def __init__(self, fit, val, ev, seed):
        self.fit, self.val, self.ev, self.seed = fit, val, ev, seed
        cals = core.platt(fit, actions=POOL)
        self.Cf = np.column_stack([core.apply_platt(fit, cals, actions=POOL)[a] for a in POOL])
        c_val = core.apply_platt(val, cals, actions=POOL)
        c_ev = core.apply_platt(ev, cals, actions=POOL)
        self.Cv = np.column_stack([c_val[a] for a in POOL])
        self.Ce = np.column_stack([c_ev[a] for a in POOL])
        hv, he = core.fit_heads(fit, [val, ev], seed, {a: self.Cf[:, j] for j, a in enumerate(POOL)},
                                actions=POOL, features=FEATURES, hp=HP_B,
                                target="regret", learner="rf")
        self.Hv, self.He = np.asarray(hv, float), np.asarray(he, float)
        cost = core.fold_costs(fit, actions=POOL)
        self.cv = np.array([cost[a] for a in POOL], float)
        self.cn = self.cv / self.cv.max()
        self.yv = val["label_supported"].to_numpy(int)
        self.ye = ev["label_supported"].to_numpy(int)
        self.Lv, self.Le = lat_matrix(val), lat_matrix(ev)
        self.pv, self.pe = pre_ms(val), pre_ms(ev)
        self._sig = {}

    # ---------------------------------------------------------------- router
    def router(self, beta):
        uv = self.Hv * np.exp(-beta * self.cn)
        ue = self.He * np.exp(-beta * self.cn)
        sv, se = uv.argmax(1), ue.argmax(1)
        rv, re_ = np.arange(len(self.Cv)), np.arange(len(self.Ce))
        q = stage2(self.Cv[rv, sv], self.yv, self.Ce[re_, se])
        ms = self.Le[re_, se] + self.pe
        return q, float(ms.mean()), 1.0

    # ---------------------------------------------------------------- cascades
    def signals(self, first, signal):
        """Escalation signal on validation and evaluation. Cached: it does not depend on tau."""
        key = (first, signal)
        if key in self._sig:
            return self._sig[key]
        if signal == "confidence":
            out = (np.abs(self.Cv[:, first] - 0.5), np.abs(self.Ce[:, first] - 0.5))
        else:
            wrong = ((self.Cf[:, first] >= 0.5).astype(int)
                     != self.fit["label_supported"].to_numpy(int)).astype(int)
            if len(set(wrong.tolist())) < 2:
                self._sig[key] = None
                return None
            Xf = self.fit[FEATURES].to_numpy(float)
            Xv = self.val[FEATURES].to_numpy(float)
            Xe = self.ev[FEATURES].to_numpy(float)
            if signal == "learned+score":
                Xf = np.column_stack([Xf, self.Cf[:, first]])
                Xv = np.column_stack([Xv, self.Cv[:, first]])
                Xe = np.column_stack([Xe, self.Ce[:, first]])
            clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, n_jobs=8,
                                         random_state=self.seed)
            clf.fit(Xf, wrong)
            out = (-clf.predict_proba(Xv)[:, 1], -clf.predict_proba(Xe)[:, 1])
        self._sig[key] = out
        return out

    def cascade(self, first, second, tau, signal):
        """Escalate to `second` when the signal is below tau. second=-1 picks the escalation
        target per row, as the highest-head-value action other than `first`, which is what the
        paper's own cascade arms do."""
        got = self.signals(first, signal)
        if got is None:
            return None
        sv, se = got
        ev_v, ev_e = sv < tau, se < tau
        rv, re_ = np.arange(len(self.Cv)), np.arange(len(self.Ce))
        if second < 0:
            hv, he = self.Hv.copy(), self.He.copy()
            hv[:, first] = -np.inf
            he[:, first] = -np.inf
            tv, te = hv.argmax(1), he.argmax(1)
        else:
            tv = np.full(len(rv), second)
            te = np.full(len(re_), second)
        p_val = np.where(ev_v, self.Cv[rv, tv], self.Cv[:, first])
        p_ev = np.where(ev_e, self.Ce[re_, te], self.Ce[:, first])
        q = stage2(p_val, self.yv, p_ev)
        ms = self.Le[:, first] + ev_e * self.Le[re_, te] + self.pe
        return q, float(ms.mean()), float(1.0 + ev_e.mean())

def frontier(points):
    """points: list of (label, auroc, ms). Keep the non-dominated ones."""
    out = []
    for lab, a, m in sorted(points, key=lambda r: r[2]):
        if not out or a > out[-1][1]:
            out.append((lab, a, m))
    return out


def main():
    TRAIN, TEST, _A, _v = V.load(with_test_labels=True)
    fits = []
    for seed in SEEDS:
        held = V.stratified_group_split(TRAIN, seed)
        fits.append(Fitted(TRAIN.loc[~held].reset_index(drop=True),
                           TRAIN.loc[held].reset_index(drop=True), TEST, seed))
        print(f"  fitted seed {seed}", flush=True)
    y = TEST["label_supported"].to_numpy(int)

    def agg(fn):
        Q, MS, CA = [], [], []
        for F in fits:
            r = fn(F)
            if r is None:
                return None
            q, ms, ca = r
            Q.append(q); MS.append(ms); CA.append(ca)
        P = np.column_stack(Q)
        return (float(np.nanmean([au(y, P[:, i]) for i in range(P.shape[1])])),
                float(np.mean(MS)), float(np.mean(CA)))

    rows = []
    for beta in GRID:
        a, m, c = agg(lambda F, b=beta: F.router(b))
        rows.append({"family": "router (pre-call)", "config": f"beta={beta}", "auroc": a,
                     "ms": m, "calls": c})
        print(f"  router beta={beta}: {a:.4f} @ {m:.1f} ms", flush=True)

    TAU_Q = np.linspace(0.0, 0.95, 20)
    for signal in ("confidence", "learned", "learned+score"):
        pairs = list(itertools.permutations(range(len(POOL)), 2))
        pairs += [(f, -1) for f in range(len(POOL))]      # per-row escalation target
        for first, second in pairs:
            for qi in TAU_Q:
                qs = [F.signals(first, signal) for F in fits]
                if any(q is None for q in qs):
                    continue
                tau = float(np.mean([np.quantile(q[0], qi) for q in qs]))
                r = agg(lambda F, f=first, s=second, t=tau, g=signal: F.cascade(f, s, t, g))
                if r is None:
                    continue
                a, m, c = r
                rows.append({"family": f"cascade ({signal})",
                             "config": (f"{NICE[POOL[first]]}->"
                                        f"{'best remaining' if second < 0 else NICE[POOL[second]]}"
                                        f" q={qi:.2f}"),
                             "auroc": a, "ms": m, "calls": c})
        print(f"  swept {signal}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "E5_cascade_sweep.csv", index=False)

    print("\n===== per-family frontier (non-dominated points) =====")
    for fam in df["family"].unique():
        sub = df[df.family == fam]
        fr = frontier(list(zip(sub["config"], sub["auroc"], sub["ms"])))
        print(f"\n--- {fam} ---")
        for lab, a, m in fr:
            print(f"   {a:.4f}  {m:7.1f} ms   {lab}")

    print("\n===== best cascade at or below the router's 106 ms =====")
    ref = df[(df.family == "router (pre-call)")].sort_values("ms")
    for fam in df["family"].unique():
        if fam.startswith("router"):
            continue
        sub = df[(df.family == fam)]
        for budget in (106.1, 120.0, 160.0, 1e9):
            ok = sub[sub.ms <= budget]
            if len(ok):
                b = ok.loc[ok.auroc.idxmax()]
                print(f"  {fam:26s} <= {budget:7.1f} ms : {b.auroc:.4f} @ {b.ms:6.1f} "
                      f"({b.calls:.2f} calls)  {b.config}")
    print(f"\nrouter operating points:\n{ref[['config','auroc','ms']].to_string(index=False)}")


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"\n[cascade] {time.time()-t0:.0f}s -> {OUT}")
