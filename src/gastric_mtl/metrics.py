"""Training-derived Breslow survival estimates and IPCW Brier scores.

IPCW assumes independent censoring. This implementation uses training patients
to estimate the censoring distribution and baseline hazard. Test outcomes never
enter either nuisance estimate. No clipping or extrapolation hides unsupported
horizons; those scores are returned as None with an explicit reason.
"""
import numpy as np


def _arrays(time, event, risk=None):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    if time.ndim != 1 or event.shape != time.shape or len(time) == 0:
        raise ValueError("Inconsistent or empty survival arrays.")
    if not np.isfinite(time).all() or not np.isfinite(event).all() or np.any(time < 0):
        raise ValueError("Invalid survival arrays.")
    if not np.isin(event, [0, 1]).all():
        raise ValueError("Events must be binary.")
    if risk is None:
        return time, event.astype(bool)
    risk = np.asarray(risk, dtype=float)
    if risk.shape != time.shape or not np.isfinite(risk).all():
        raise ValueError("Invalid risk array.")
    return time, event.astype(bool), risk


def censoring_km(train_time, train_event):
    """Reverse Kaplan-Meier; events precede censoring for tied times.

    At a tied time, original events are removed from the denominator before
    estimating censoring probability (reverse-KM convention).
    """
    time, event = _arrays(train_time, train_event)
    unique, inverse, counts = np.unique(time, return_inverse=True, return_counts=True)
    at_risk = np.cumsum(counts[::-1])[::-1].astype(float)
    deaths = np.bincount(inverse, weights=event, minlength=len(unique))
    censored = counts - deaths
    denom = at_risk - deaths
    decrement = np.ones(len(unique))
    nonzero = censored > 0
    decrement[nonzero] = 1 - censored[nonzero] / denom[nonzero]
    return unique, np.cumprod(decrement)


def _step(times, values, points, left=False, initial=1.0):
    points = np.asarray(points, dtype=float)
    indices = np.searchsorted(times, points, side="left" if left else "right") - 1
    result = np.full(points.shape, initial, dtype=float)
    mask = indices >= 0
    result[mask] = values[indices[mask]]
    return result


def predict_survival(train_time, train_event, train_risk, test_risk, horizons):
    """Breslow baseline fitted only on training data; output n_test x n_times."""
    time, event, risk = _arrays(train_time, train_event, train_risk)
    test_risk = np.atleast_1d(np.asarray(test_risk, dtype=float))
    horizons = np.atleast_1d(np.asarray(horizons, dtype=float))
    order = np.argsort(time, kind="stable")
    time, event, risk = time[order], event[order], risk[order]
    unique, first, inverse = np.unique(time, return_index=True, return_inverse=True)
    deaths = np.bincount(inverse, weights=event, minlength=len(unique))
    has_events = deaths > 0
    event_times = unique[has_events]
    if not len(event_times):
        return np.ones((len(test_risk), len(horizons)))
    log_risk = np.logaddexp.accumulate(risk[::-1])[::-1]
    log_increment = np.log(deaths[has_events]) - log_risk[first[has_events]]
    log_cum_hazard = np.logaddexp.accumulate(log_increment)
    log_h0 = _step(event_times, log_cum_hazard, horizons, initial=-np.inf)
    log_total = test_risk[:, None] + log_h0[None, :]
    # Beyond exp(7) the survival probability is numerically zero; capping only
    # avoids overflow and does not impose any nonzero probability floor.
    return np.exp(-np.exp(np.minimum(log_total, 7.0)))


def evaluate_survival(train_time, train_event, train_risk,
                      test_time, test_event, test_risk,
                      horizons=(365.25, 1095.75), min_g=1e-8):
    """Training-weighted IPCW Brier score at prespecified horizons.

    Score at tau is n_test^-1 sum_i [I(T_i<=tau,delta_i=1) S_i(tau)^2 /
    G(T_i-) + I(T_i>tau) (1-S_i(tau))^2 / G(tau)]. Patients censored by tau
    contribute zero. G is reverse KM on the training fold only.
    """
    train_time, train_event, train_risk = _arrays(train_time, train_event, train_risk)
    test_time, test_event, test_risk = _arrays(test_time, test_event, test_risk)
    horizons = np.asarray(horizons, dtype=float)
    km_time, km_g = censoring_km(train_time, train_event)
    survival = predict_survival(train_time, train_event, train_risk, test_risk, horizons)
    output = {"details": {}}
    for j, horizon in enumerate(horizons):
        key = str(float(horizon))
        deaths = test_event & (test_time <= horizon)
        survivors = test_time > horizon
        g_h = float(_step(km_time, km_g, [horizon])[0])
        g_death = _step(km_time, km_g, test_time[deaths], left=True)
        detail = {"horizon_days": float(horizon), "n_test": len(test_time),
                  "n_events_by_horizon": int(deaths.sum()), "n_survivors": int(survivors.sum()),
                  "n_censored_by_horizon": int((~test_event & (test_time <= horizon)).sum()),
                  "g_horizon": g_h, "train_max_followup": float(train_time.max()),
                  "min_g_at_test_events": float(g_death.min()) if len(g_death) else None,
                  "supported": True, "reason": None}
        reason = None
        if horizon < 0 or horizon >= train_time.max():
            reason = "horizon must be nonnegative and strictly below maximum training follow-up"
        elif g_h <= min_g:
            reason = "training censoring survival at horizon is zero or too small"
        elif np.any(g_death <= min_g):
            reason = "training censoring survival before a test event is zero or too small"
        elif not np.any(deaths | survivors):
            reason = "no test outcomes are ascertainable at this horizon"
        if reason:
            detail.update(supported=False, reason=reason)
            score = None
        else:
            contributions = np.zeros(len(test_time))
            contributions[deaths] = survival[deaths, j] ** 2 / g_death
            contributions[survivors] = (1 - survival[survivors, j]) ** 2 / g_h
            score = float(contributions.mean())
        output["brier_" + key] = score
        output["details"][key] = detail
    return output
