"""Expected-profit scoring and realised-profit evaluation (paper Sec. III-B, III-E).

Decision scores
  paper rule   G_i  = q * pi_i * V_i - c,  V_i = m*MC_i*sum_t gamma^t S(t)         (Eq. 1-3)
  refined rule G2_i = q * L_i - c,  L_i = m*MC_i*sum_t gamma^t (1 - S(t-1))
                (expected margin lost inside the horizon if left alone; this is the
                 quantity the realised-profit evaluation below measures, so it is an
                 addition to the paper, reported separately and clearly labelled)

Realised profit (no intervention data exists, so this is an *assumed-q* evaluation):
  contacting customer i earns  q * lost_i - c,  where lost_i = m*MC_i*sum_{t=T_i+1..H} gamma^t
  if the customer actually churned within H months, else 0.
  Customers censored before H have unknown status; they get weight 0 and the rest are
  re-weighted by inverse probability of censoring (Kaplan-Meier of the censoring time).
"""
import numpy as np
from lifelines import KaplanMeierFitter

from data import MAX_T


def disc(gamma, n=MAX_T):
    """cum[t] = sum_{s=1..t} gamma^s, cum[0] = 0."""
    return np.concatenate([[0.0], np.cumsum(gamma ** np.arange(1, n + 1))])


def scores(S, MC, H, m, q, c, gamma):
    """S: (n,72) with S[:, t-1] = S(t). Returns dict of pi, V, L, G (paper), G2 (refined)."""
    t = np.arange(1, H + 1)
    g = gamma ** t
    S_H = S[:, :H]
    S_prev = np.concatenate([np.ones((len(S), 1)), S[:, :H - 1]], axis=1)
    pi = 1.0 - S[:, H - 1]
    V = m * MC * (S_H * g).sum(axis=1)
    L = m * MC * ((1.0 - S_prev) * g).sum(axis=1)
    return {"pi": pi, "V": V, "L": L, "G": q * pi * V - c, "G2": q * L - c}


def ipcw_outcomes(T, E, MC, H, m, gamma):
    """Return (weight, lost) arrays for the realised-profit evaluation at horizon H."""
    T, E = np.asarray(T), np.asarray(E)
    cum = disc(gamma)
    churn_in_H = (E == 1) & (T <= H)
    known = churn_in_H | (T > H)
    tt = np.minimum(T, H).astype(float)
    km = KaplanMeierFitter().fit(T, 1 - E)  # censoring distribution
    Gc = km.survival_function_at_times(np.maximum(tt - 1e-9, 0)).values
    w = np.where(known, 1.0 / np.clip(Gc, 1e-3, None), 0.0)
    lost = np.where(churn_in_H, m * MC * (cum[H] - cum[np.minimum(T, H)]), 0.0)
    return w, lost


def profit_curve(score, w, lost, q, c):
    """Net profit when contacting the top-k by score, k = 0..n. Returns array length n+1."""
    order = np.argsort(-score, kind="stable")
    gain = q * w[order] * lost[order] - c  # benefit is IPCW-corrected; cost is paid per contact
    return np.concatenate([[0.0], np.cumsum(gain)])


def random_curve(w, lost, q, c):
    n = len(w)
    total = (q * w * lost - c).sum()
    return total * np.arange(n + 1) / n


def profit_at(score, threshold, w, lost, q, c):
    """Realised profit when contacting everyone whose score exceeds `threshold`."""
    sel = score > threshold
    return float((q * w[sel] * lost[sel] - c).sum()), int(sel.sum())
