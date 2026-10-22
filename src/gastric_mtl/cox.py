"""Audited Cox objectives for the 2026-09-14 submission analysis.

All objectives use the original project's 2/n loss normalization. Ties use
Breslow risk sets. The group and L1 penalties are smooth approximations, not
exact sparse proximal solutions; ``lambda_max`` is a NONSMOOTH reference scale.
The coupling is a differentiable approximation to multireg24 with the correct
derivative. Coefficients are never thresholded during optimization.
"""
from __future__ import annotations

import time as _time
import numpy as np
from scipy.optimize import minimize


def _group_info(groups, p):
    if groups is None:
        labels = np.arange(p, dtype=int)
    elif isinstance(groups, (list, tuple)) and len(groups) and isinstance(groups[0], (list, tuple, np.ndarray)):
        labels = np.full(p, -1, dtype=int)
        for i, members in enumerate(groups):
            members = np.asarray(members, dtype=int)
            if np.any(labels[members] >= 0):
                raise ValueError("Groups must be disjoint.")
            labels[members] = i
        if np.any(labels < 0):
            raise ValueError("Every feature must belong to a group.")
    else:
        labels = np.asarray(groups)
        if labels.shape != (p,):
            raise ValueError("groups must contain one label per feature.")
    _, labels = np.unique(labels, return_inverse=True)
    sizes = np.bincount(labels).astype(float)
    return labels, sizes


class CoxObjective:
    """Stable, order-independent Breslow negative partial likelihood."""

    def __init__(self, X, time, event):
        X = np.asarray(X, dtype=np.float64)
        time = np.asarray(time, dtype=float)
        event = np.asarray(event, dtype=float)
        if X.ndim != 2 or time.shape != (len(X),) or event.shape != time.shape:
            raise ValueError("Inconsistent X, time and event shapes.")
        if not np.isfinite(X).all() or not np.isfinite(time).all() or not np.isfinite(event).all():
            raise ValueError("Nonfinite input.")
        if not np.isin(event, [0, 1]).all() or event.sum() == 0:
            raise ValueError("event must be binary with at least one observed event.")
        if np.any(time < 0):
            raise ValueError("Negative survival time.")
        order = np.argsort(time, kind="stable")
        self.X = np.ascontiguousarray(X[order])
        self.time = time[order]
        self.event = event[order]
        self.n, self.p = X.shape
        self.first = np.searchsorted(self.time, self.time, side="left")
        counts = np.bincount(self.first, weights=self.event, minlength=self.n)
        self.log_counts = np.full(self.n, -np.inf)
        positive = counts > 0
        self.log_counts[positive] = np.log(counts[positive])

    def value_grad(self, beta):
        eta = self.X @ beta
        log_risk = np.logaddexp.accumulate(eta[::-1])[::-1]
        value = 2.0 / self.n * np.dot(self.event, log_risk[self.first] - eta)
        # eta_i + log(sum_{event times <= t_i} d_t / R_t) is stable even
        # when linear predictors are very large or very small.
        log_accum = np.logaddexp.accumulate(self.log_counts - log_risk)
        weighted = np.exp(eta + log_accum)
        grad_eta = 2.0 / self.n * (weighted - self.event)
        return float(value), self.X.T @ grad_eta


def penalty_value_grad(beta, labels, sizes, penalty="group", alpha=0.5, eps=1e-4):
    if penalty == "group":
        norms = np.sqrt(np.bincount(labels, weights=beta * beta, minlength=len(sizes)) + eps * eps)
        weights = np.sqrt(sizes)
        return float(np.dot(weights, norms - eps)), beta * (weights / norms)[labels]
    if penalty in ("lasso", "l1", "elasticnet", "en"):
        norms = np.sqrt(beta * beta + eps * eps)
        val = float(np.sum(norms - eps))
        grad = beta / norms
        if penalty in ("elasticnet", "en"):
            return alpha * val + (1 - alpha) * 0.5 * float(beta @ beta), alpha * grad + (1 - alpha) * beta
        return val, grad
    if penalty in ("ridge", "l2"):
        return 0.5 * float(beta @ beta), beta.copy()
    if penalty in (None, "none"):
        return 0.0, np.zeros_like(beta)
    raise ValueError("Unknown penalty: %s" % penalty)


def coupling_value_grad(beta1, beta2, labels, sizes, eps=1e-4):
    """Smooth multireg24: sqrt(2 sum_g |g|(a_g b_g - dot_g)+eps²)-eps.

    a_g=sqrt(||beta1_g||²+eps²), likewise b_g. As eps tends to zero this
    recovers the original norm of weighted direction differences. Unlike the
    legacy implementation it has no discontinuous group-activation threshold.
    """
    ng = len(sizes)
    a = np.sqrt(np.bincount(labels, weights=beta1 * beta1, minlength=ng) + eps * eps)
    b = np.sqrt(np.bincount(labels, weights=beta2 * beta2, minlength=ng) + eps * eps)
    dot = np.bincount(labels, weights=beta1 * beta2, minlength=ng)
    squared = max(0.0, 2.0 * float(np.dot(sizes, a * b - dot)))
    den = np.sqrt(squared + eps * eps)
    g1 = (sizes / den)[labels] * ((b / a)[labels] * beta1 - beta2)
    g2 = (sizes / den)[labels] * ((a / b)[labels] * beta2 - beta1)
    return float(den - eps), g1, g2


def lambda_max(X, time, event, groups=None, penalty="group", alpha=0.5):
    """Non-smoothed penalty's all-zero reference threshold (ties corrected).

    Ridge has no finite all-zero threshold; for ridge this returns the maximum
    absolute Cox gradient as a reproducible reference scale only.
    """
    obj = CoxObjective(X, time, event)
    _, grad = obj.value_grad(np.zeros(obj.p))
    if penalty == "group":
        labels, sizes = _group_info(groups, obj.p)
        return float(np.max(np.sqrt(np.bincount(labels, weights=grad * grad)) / np.sqrt(sizes)))
    scale = float(np.max(np.abs(grad)))
    if penalty in ("elasticnet", "en") and alpha > 0:
        scale /= alpha
    return scale


def _minimize(fun, beta0, max_iter, tol, gtol):
    started = _time.perf_counter()
    result = minimize(fun, np.asarray(beta0, dtype=float), method="L-BFGS-B", jac=True,
                      options={"maxiter": int(max_iter), "ftol": float(tol),
                               "gtol": float(gtol), "maxls": 40, "maxcor": 10})
    return {"beta": result.x, "converged": bool(result.success),
            "n_iter": int(result.nit), "n_eval": int(result.nfev),
            "objective": float(result.fun),
            "gradient_norm": float(np.max(np.abs(result.jac))),
            "status": int(result.status), "message": str(result.message),
            "seconds": float(_time.perf_counter() - started)}


def fit_cox(X, time, event, groups=None, lam=0.1, penalty="group", alpha=0.5,
            max_iter=200, tol=1e-8, gtol=1e-5, beta0=None, eps=1e-4):
    """Fit a smoothed penalized Cox model using L-BFGS-B."""
    obj = CoxObjective(X, time, event)
    labels, sizes = _group_info(groups, obj.p)
    if beta0 is None:
        beta0 = np.zeros(obj.p)

    def fun(beta):
        value, grad = obj.value_grad(beta)
        if lam:
            pv, pg = penalty_value_grad(beta, labels, sizes, penalty, alpha, eps)
            value += lam * pv
            grad += lam * pg
        return value, grad

    return _minimize(fun, beta0, max_iter, tol, gtol)


def fit_multitask(tasks, groups, lam, mu=0.0, max_iter=200, tol=1e-8,
                  gtol=1e-5, beta0=None, eps=1e-4):
    """Joint fit, with task dictionaries containing X, time and event."""
    objs = [CoxObjective(t["X"], t["time"], t["event"]) for t in tasks]
    p = objs[0].p
    if any(o.p != p for o in objs):
        raise ValueError("Tasks must share feature order and dimension.")
    labels, sizes = _group_info(groups, p)
    nt = len(objs)
    lam = np.broadcast_to(np.asarray(lam, dtype=float), (nt,))
    mu_array = np.full((nt, nt), float(mu)) if np.ndim(mu) == 0 else np.asarray(mu, dtype=float)
    if beta0 is None:
        beta0 = np.zeros((nt, p))

    def fun(flat):
        betas = flat.reshape(nt, p)
        grads = np.zeros_like(betas)
        value = 0.0
        for k, obj in enumerate(objs):
            cv, cg = obj.value_grad(betas[k])
            pv, pg = penalty_value_grad(betas[k], labels, sizes, eps=eps)
            value += cv + lam[k] * pv
            grads[k] = cg + lam[k] * pg
        for k in range(nt):
            for j in range(k + 1, nt):
                if mu_array[k, j] > 0:
                    cv, gk, gj = coupling_value_grad(betas[k], betas[j], labels, sizes, eps)
                    value += mu_array[k, j] * cv
                    grads[k] += mu_array[k, j] * gk
                    grads[j] += mu_array[k, j] * gj
        return float(value), grads.ravel()

    result = _minimize(fun, np.asarray(beta0).ravel(), max_iter, tol, gtol)
    result["betas"] = list(result.pop("beta").reshape(nt, p))
    return result


def c_index(time, event, risk):
    """Harrell C, independent of row order; higher risk means earlier event.

    Event/censor pairs tied on time are comparable with the event first;
    event/event pairs tied on time are excluded. Equal risk receives 0.5.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    risk = np.asarray(risk, dtype=float)
    if not (time.shape == event.shape == risk.shape):
        raise ValueError("Inconsistent input lengths.")
    valid = event[:, None] & ((time[:, None] < time[None, :]) |
                              ((time[:, None] == time[None, :]) & ~event[None, :]))
    n = int(valid.sum())
    if n == 0:
        return float("nan")
    diff = risk[:, None] - risk[None, :]
    return float((np.sum((diff > 0) & valid) + 0.5 * np.sum((diff == 0) & valid)) / n)
