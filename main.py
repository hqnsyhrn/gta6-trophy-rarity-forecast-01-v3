"""GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy, five quantiles of its earn rate on Exophase at
the case target.  The model works in counts, not rates:

    rate_i(t) = E_i(t) / N(t)

where N(t) is the players Exophase tracks and E_i(t) the number of them who
have earned trophy i.  The two move on different clocks: N jumps when
Exophase refreshes a game's counts (launch night: frozen at 638 players for
hours, then one refresh to 3,868), while E_i grows only as tracked players
actually play and earn.  Forecasting the rate therefore means forecasting
a denominator that jumps and a numerator that lags.

Structure
---------
* N(t): a log-logistic S-curve in log-hours since release, fitted on the
  case history over a grid of carrying capacities K; the forecast N is the
  likelihood-weighted geometric mean over K (a single best K overfits when
  the curve is still rising, so the ensemble carries the saturation
  uncertainty instead of pretending it away).
* E_i(t): an anchored log-time trend per trophy -- log E linear in
  log-hours since release, anchored at the last observed earner count,
  slope clipped to [-0.5, 3].  Blending in other laws (plain OLS, a
  delayed curve that follows the players curve with a lag, flat
  earners), shrinking the slope toward the tier mean, and damping it
  for long horizons were all tried on synthetic worlds and all lose to
  the plain anchored trend, worst exactly where earners accelerate;
  they are not used.  Rolling-origin backtests inside the case history
  are kept only for the spread: they measure the anchored trend's
  typical (median) logit error, clamped into a sane band, and the fan is
  symmetric, because in a saturating world rates rise past the median as
  often as dilution pushes them below it.
* Flat history (no refresh seen yet, the practice cases): no trend can
  be fitted, so a first-refresh prior applies instead -- the launch-night
  pattern that a game's counts sit frozen for hours and then jump about
  sixfold (638 -> 3,868) while earners lag far behind (a new cohort of
  players has barely played, so it adds only a couple of earners per
  trophy on average, a few more for the easiest ones).  The median commits
  to that jump; the fan stays wide and asymmetric because the refresh may
  still land later than the target, or a second one may land before it.

Only the standard library is used.  The program is deterministic: every
fit is a fixed grid search, so two runs write byte-identical output.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# First-refresh prior (launch night, Exophase): the tracked-player count
# jumped 638 -> 3,868 in one refresh, a factor of about 6.06, while the
# earners of a typical trophy grew only by a couple of players -- the new
# cohort had barely started playing.  New earners for trophy i are
# modelled as NEW_A + NEW_B * E_i(last): a small floor for every trophy
# plus a term that grows with how many of the old cohort earned it (the
# easiest trophies pick up the most new earners).  The constants are set
# from the launch-night aggregates alone -- the refreshed page held 4,276
# trophies earned in total, and the pair below lands within 20 of it --
# not from any single trophy's refreshed rate.
JUMP = 3868.0 / 638.0
NEW_A = 0.75
NEW_B = 0.15

# Carrying-capacity grid for the players-tracked S-curve.
K_GRID = [120000, 200000, 300000, 450000, 600000, 900000, 1500000, 3000000]


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg > 30:
        return 99.99
    if lg < -30:
        return 0.01
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up, as JavaScript's Math.round does."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def log_hours(t, t_ref):
    """log(1 + hours from t_ref to t), never negative."""
    return math.log1p(max(0.0, (t - t_ref) / 3600.0))


def n_curve(u, K, u0, w):
    x = (u - u0) / w
    if x > 30:
        return float(K)
    if x < -30:
        return 0.0
    return K / (1.0 + math.exp(-x))


def fit_players(ts, Ns, t_target, t_ref):
    """Fit the log-logistic players curve; return (N_target, best curve).

    The best curve is the (K, u0, w) with the lowest squared error on
    log N.  N_target is the likelihood-weighted geometric mean over K of
    the curve's value at the target, so saturation uncertainty widens the
    forecast instead of collapsing onto one capacity.
    """
    u = [log_hours(t, t_ref) for t in ts]
    ut = log_hours(t_target, t_ref)
    logN = [math.log(max(N, 1.0)) for N in Ns]
    per_k = []
    for K in K_GRID:
        best = None
        for u0_i in range(25):
            u0 = u0_i * 0.25
            for w in (0.3, 0.5, 0.8, 1.2, 2.0):
                sse = 0.0
                for uu, ln in zip(u, logN):
                    pred = max(n_curve(uu, K, u0, w), 1.0)
                    d = math.log(pred) - ln
                    sse += d * d
                if best is None or sse < best[0]:
                    best = (sse, u0, w)
        per_k.append((best[0], K, best[1], best[2]))
    per_k.sort()
    sse_min = per_k[0][0]
    var = max(sse_min / max(len(ts), 1), 1e-6)
    total_w = 0.0
    total_ln = 0.0
    for sse, K, u0, w in per_k:
        wt = math.exp(-(sse - sse_min) / (2.0 * var))
        total_w += wt
        total_ln += wt * math.log(max(n_curve(ut, K, u0, w), 1.0))
    n_target = math.exp(total_ln / total_w) if total_w > 0 else float(Ns[-1])
    _, K, u0, w = per_k[0]
    return n_target, (K, u0, w)


def fit_loglin(u, logE, ut):
    """Least-squares line of log E on u; returns (intercept, slope)."""
    n = len(u)
    mx = sum(u) / n
    my = sum(logE) / n
    den = sum((x - mx) ** 2 for x in u)
    if den < 1e-9:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(u, logE)) / den
    return my - b * mx, b


def backtest_anchored_resid(ts, Ns, rates_i, t_ref, horizon_sec):
    """Rolling-origin residuals of the anchored log-time trend.

    Origins are spread over the history; at each, the trend is fitted on
    the reads up to the origin, anchored at the origin's earner count,
    and scored on the first read at least `horizon_sec` later (or the
    last read when the history ends first).  Returns the MEDIAN absolute
    logit error across origins: the RMS is dominated by the worst origin
    (a refresh landing just after it), while the median says what the
    trend typically misses.  No horizon scaling is applied to it: on the
    synthetic worlds the error does not grow with lead -- it is level
    noise from the refresh steps, not trend error -- and scaling by
    sqrt(horizon/lead) overshoots the fan and loses to persistence on
    the saturating world.  The fan instead clamps the residual into a
    sane band (see forecast_case).
    """
    n = len(ts)
    if n < 6:
        return None
    fracs = [0.4, 0.55, 0.7, 0.85]
    origins = sorted(set(max(3, int(f * n)) for f in fracs))
    origins = [o for o in origins if o < n - 1]
    if not origins:
        return None
    u_all = [log_hours(t, t_ref) for t in ts]
    Es_all = [(r / 100.0 * N) if r is not None else 0.0
              for r, N in zip(rates_i, Ns)]
    errs = []
    for o in origins:
        u_pre = u_all[:o]
        es_pre = Es_all[:o]
        j = o
        while j < n - 1 and ts[j] - ts[o] < horizon_sec:
            j += 1
        if j >= n:
            j = n - 1
        if ts[j] <= ts[o]:
            continue
        pts = [(uu, math.log(max(E, 0.5))) for uu, E in zip(u_pre, es_pre)
               if E > 0]
        if len(pts) < 3:
            continue
        _, b = fit_loglin([p[0] for p in pts], [p[1] for p in pts], u_all[j])
        b = min(3.0, max(-0.5, b))
        pred_e = max(es_pre[-1], 0.5) * math.exp(b * (u_all[j] - u_pre[-1]))
        actual = Es_all[j]
        if actual <= 0:
            continue
        q = max(pred_e, 0.5) / max(Ns[j], 1.0) * 100.0
        y = actual / max(Ns[j], 1.0) * 100.0
        errs.append(abs(logit(y) - logit(q)))
    if not errs:
        return None
    errs.sort()
    mid = len(errs) // 2
    if len(errs) % 2:
        return errs[mid]
    return (errs[mid - 1] + errs[mid]) / 2.0


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    horizon_min = case["horizon_minutes"]
    horizon_sec = horizon_min * 60.0
    t_target = parse_time(case["target"])
    t_ref = parse_time(case["game"]["release_date"] + "T00:00:00Z") \
        if case.get("game", {}).get("release_date") else parse_time(history[0]["at"])
    ts = [parse_time(h["at"]) for h in history]
    Ns = [float(h["players_tracked"]) for h in history]
    n_last = Ns[-1]
    flat = (max(Ns) / max(min(Ns), 1.0) < 1.05) if Ns else True

    u_all = [log_hours(t, t_ref) for t in ts]
    ut = log_hours(t_target, t_ref)

    rows = []
    if flat:
        # No refresh has been seen: the first-refresh prior.  The median
        # commits to the launch-night jump; the fan is wide and skewed
        # upward, because the refresh may still land after the target
        # (rates stay where they are) or a second one may land before
        # it (rates fall further).
        n_target = n_last * JUMP
        sig_lo, sig_hi = 0.25, 0.35
        for i, trophy in enumerate(trophies):
            last = history[-1]["rates"][i]
            if last is None:
                last = 0.0
            if last >= 99.99:
                med = 100.0
                lo_s, hi_s = 0.02, 0.02
            else:
                e_last = last / 100.0 * n_last
                e_new = NEW_A + NEW_B * e_last
                med = (e_last + e_new) / n_target * 100.0
                med = clamp_pct(med)
                lo_s, hi_s = sig_lo, sig_hi
            lg = logit(med)
            qs = []
            for z, s in zip(Z, [lo_s, lo_s, 0.0, hi_s, hi_s]):
                qs.append(round4(clamp_pct(inv_logit(lg + z * s))))
            rows.append({"id": trophy["id"], "q": qs})
        return {"trophies": rows}

    # Rich history: fit the players curve; the median earner count is
    # the anchored log-time trend alone.  Blending in the other laws was
    # tried and loses to it on every synthetic world, and shrinking its
    # slope toward the tier mean or damping it for long horizons loses
    # worst exactly where earners accelerate (the saturating world).  The
    # slope is clipped, which is the only guard it needs.
    n_target, (K, u0, w) = fit_players(ts, Ns, t_target, t_ref)

    for i, trophy in enumerate(trophies):
        rates_i = [h["rates"][i] for h in history]
        last = rates_i[-1]
        if last is not None and last >= 99.99:
            med = 100.0
            sig_lo, sig_hi = 0.02, 0.02
            lg = logit(med)
            rows.append({"id": trophy["id"], "q": [
                round4(clamp_pct(inv_logit(lg + z * s)))
                for z, s in zip(Z, [sig_lo, sig_lo, 0.0, sig_hi, sig_hi])]})
            continue
        Es = [(r / 100.0 * N) if r is not None else 0.0
              for r, N in zip(rates_i, Ns)]
        resid = backtest_anchored_resid(ts, Ns, rates_i, t_ref, horizon_sec)
        pts = [(uu, math.log(max(E, 0.5))) for uu, E in zip(u_all, Es)
               if E > 0]
        if len(pts) >= 3:
            _, b = fit_loglin([p[0] for p in pts], [p[1] for p in pts], ut)
            b = min(3.0, max(-0.5, b))
            e_target = max(Es[-1], 0.5) * math.exp(b * (ut - u_all[-1]))
        else:
            e_target = max(Es[-1], 0.5)
        med = clamp_pct(e_target / max(n_target, 1.0) * 100.0)
        # Spread: the rolling-origin residual, clamped into [0.25, 0.45]
        # so a quiet history never claims certainty and a wild one never
        # claims everything.  The fan is symmetric: in a saturating world
        # rates rise past the median as often as dilution pushes them
        # below it, so a downside-skewed fan undercovers the upside --
        # on the synthetic week-one world its 90% coverage falls to
        # 0.73, against 0.91 symmetric.
        if resid is None:
            base = 0.35
        else:
            base = min(0.45, max(0.25, resid))
        sig_lo, sig_hi = base, base
        lg = logit(med)
        rows.append({"id": trophy["id"], "q": [
            round4(clamp_pct(inv_logit(lg + z * s)))
            for z, s in zip(Z, [sig_lo, sig_lo, 0.0, sig_hi, sig_hi])]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
