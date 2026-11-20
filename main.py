"""GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, the
percent of the players Exophase tracks for the game who will have earned it
at the case's target, as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95).

Method: cohort-holder model with refresh-aware dilution.
---------------------------------------------------------
A trophy's earn rate is a ratio, r = H / P, where H is the number of
tracked players holding the trophy and P is the number of players tracked.
The two move for different reasons, so they are forecast separately and
the rate is the ratio of the two forecasts:

* Players tracked P(t): Exophase only counts a player once their profile
  syncs, so P grows in jumps at each refresh of the page, not smoothly.
  We fit ln P against ln(1 + hours since release) by least squares over
  the case's history and read the line at the target. Log-time is used
  because tracking growth is fast at launch and saturates: a line in
  log-time extrapolates much more gently than a line in raw time.

* Holders H(t): per trophy, H = r * P / 100 in each read (the page prints
  r to 2 decimals, so H is approximate for small counts). We fit
  ln(H + 0.5) against ln(1 + hours since release) the same way, with the
  slope shrunk 30% toward the median slope over all trophies (small
  holder counts are noisy, and most trophies move together). The +0.5 is
  a continuity correction so a trophy nobody holds yet still has a
  finite log.

* Refresh prior: Exophase refreshes a game's counts on its own schedule.
  Launch night the page sat unchanged at 638 players tracked from the
  first probe (2026-11-18T23:48Z) through the 04:00Z read, then jumped
  to 3,868 players by 04:30Z -- a 6.06x jump -- and every rate but the
  one at 100% fell, because the newly tracked players held fewer
  trophies. The same published note gives the totals: trophies earned
  went 943 -> 4,276, so the 3,230 newly tracked players added only
  ~103 holders across the 54 trophies below 100% -- about 7% of the old
  cohort's hold rate. When a case's history shows no refresh at all
  (players tracked identical in every read), a refresh is overdue: we
  assume one refresh before a short-horizon target, multiplying P by
  6.06 (the one refresh the competition's own rules document), and
  credit the new cohort with holding each trophy at 7% of its current
  rate. For longer horizons we assume
  one such refresh per ~4.5 hours of horizon, capped at a 40x total
  multiplier. When the history does show refreshes, the fits above
  already carry their average effect and no prior is added.

* Spread: the logit spread s combines a random-walk term
  (0.5 * sqrt(horizon in days), the house baseline's scale), the fit
  residuals (how wobbly each trophy's holder and player series were),
  and a refresh term (large when no refresh has been seen yet, small
  when refreshes are visible in the history). Quantiles are
  inv_logit(logit(median) + z_tau * s), z = (-1.645, -0.674, 0, 0.674,
  1.645). A trophy last seen at 100% is forecast to stay at ~100%:
  everybody who launches the game holds it.

Assumptions, stated plainly (no fake data source):
* The only data used is the case file itself. No network, no outside
  trophy database; the numbers above (6.06x, 35%) are priors taken from
  the competition's own published rules, not from a measurement of the
  answer, and they only bite when the history is flat.
* Trophy names/descriptions are not used to set rates, except that the
  100% trophy is pinned near 100%. Tier only enters through the
  slope-shrinkage median, which is computed from the case itself.
* The model cannot foresee a refresh landing exactly on the target, a
  day-one patch changing a trophy, or a streamer event pulling players
  in faster than the fits imply; that is what the spread is for.

Runs offline, standard library only, deterministic (no randomness,
no dict-order dependence: trophies are output in case order).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a helper
# module shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Random-walk spread of a rate's logit move, per square-root day.
SIGMA = 0.5
# Refresh prior, from the one refresh the competition rules document
# (638 -> 3,868 players tracked, launch night): jump factor and the
# share of the current rate a newly tracked player is credited with.
# The 7% comes from the same note's totals (trophies earned 943 ->
# 4,276): the new cohort added only ~103 holders across the 54
# trophies below 100%.
REFRESH_JUMP = 3868.0 / 638.0
NEW_COHORT_HOLD = 0.07
REFRESH_EVERY_HOURS = 4.5
# Caps so an extrapolation can never run away.
MAX_PLAYER_MULT = 40.0
MAX_HOLDER_MULT = 50.0


def parse_time(text):
    """ISO-8601 UTC timestamp -> aware datetime (handles trailing Z)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (matches JavaScript Math.round)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def least_squares(xs, ys):
    """Ordinary least squares y ~ a + b*x. Returns (intercept, slope)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def residuals_sd(xs, ys, a, b):
    """Residual standard deviation of the least-squares fit."""
    n = len(xs)
    if n < 3:
        return 0.0
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return math.sqrt(sse / (n - 2))


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])

    # Time axis: hours since the release day began (game.release_date).
    release = case.get("game", {}).get("release_date", "2026-11-19")
    release_start = datetime.fromisoformat(release + "T00:00:00+00:00")

    def log_time(at):
        hours = (parse_time(at) - release_start).total_seconds() / 3600.0
        return math.log1p(max(0.0, hours))

    reads = []
    for read in history:
        players = read.get("players_tracked")
        rates = read.get("rates") or []
        if not players or players <= 0:
            continue
        reads.append({"x": log_time(read["at"]), "players": float(players),
                      "rates": rates})
    if not reads:
        raise ValueError(f"{name} has no usable history read")
    last = reads[-1]
    x_last = last["x"]
    x_target = log_time(case["target"])
    # Never extrapolate backwards; a target before the last read is a
    # degenerate case, forecast persistence.
    x_target = max(x_target, x_last)

    # --- players tracked: log-time line over the whole history. ---
    xs = [r["x"] for r in reads]
    a_p, b_p = least_squares(xs, [math.log(r["players"]) for r in reads])
    resid_p = residuals_sd(xs, [math.log(r["players"]) for r in reads], a_p, b_p)
    players_last = last["players"]
    players_flat = max(r["players"] for r in reads) == min(r["players"] for r in reads)
    if players_flat:
        # No refresh visible in the whole history: one is overdue.
        # Short horizon: one documented-size jump. Longer: one per
        # ~4.5 h of horizon, capped.
        jumps = max(1.0, (case["horizon_minutes"] / 60.0) / REFRESH_EVERY_HOURS)
        mult = min(MAX_PLAYER_MULT, REFRESH_JUMP ** jumps)
        players_target = players_last * mult
        refresh_spread = 0.8
    else:
        log_p_target = a_p + b_p * x_target
        players_target = math.exp(log_p_target)
        mult = players_target / players_last
        mult = min(MAX_PLAYER_MULT, max(1.0, mult))
        players_target = players_last * mult
        # Observed refresh jumps: how jumpy P was, in log terms.
        jumps = [math.log(reads[i]["players"] / reads[i - 1]["players"])
                 for i in range(1, len(reads))
                 if reads[i]["players"] > reads[i - 1]["players"] * 1.01]
        jump_sd = (math.sqrt(sum(j * j for j in jumps) / len(jumps))
                   if jumps else 0.3)
        refresh_spread = min(0.8, 0.25 + 0.5 * jump_sd)

    # --- holders per trophy: log-time line of ln(H + 0.5). ---
    slopes = []
    fits = []
    for i in range(len(trophies)):
        txs, tys = [], []
        for r in reads:
            if i < len(r["rates"]) and r["rates"][i] is not None:
                holders = r["rates"][i] / 100.0 * r["players"]
                txs.append(r["x"])
                tys.append(math.log(holders + 0.5))
        if len(txs) >= 2:
            a_h, b_h = least_squares(txs, tys)
        elif txs:
            a_h, b_h = tys[0], 0.0
        else:
            a_h, b_h = math.log(0.5), 0.0
        slopes.append(b_h)
        fits.append((txs, tys, a_h, b_h))
    slope_median = median(slopes)

    rows = []
    for i, trophy in enumerate(trophies):
        # Last non-null rate for this trophy (a read can lack one; the
        # last read normally lists every trophy, but do not trust that).
        rate_last = None
        players_at_last = players_last
        for r in reversed(reads):
            if i < len(r["rates"]) and r["rates"][i] is not None:
                rate_last = r["rates"][i]
                players_at_last = r["players"]
                break
        if rate_last is None:
            # No read ever listed this trophy: forecast near zero, wide.
            rows.append({"id": trophy["id"],
                         "q": [0.01, 0.01, 0.05, 0.5, 5.0]})
            continue
        if rate_last >= 99.99:
            # Everybody who launches the game holds it; it stays ~100%.
            rows.append({"id": trophy["id"],
                         "q": [99.0, 99.5, 99.9, 99.99, 99.99]})
            continue

        txs, tys, a_h, b_h = fits[i]
        # Shrink the slope 30% toward the median trophy's slope: small
        # holder counts are noisy, trophies mostly move together.
        b_h = 0.7 * b_h + 0.3 * slope_median
        b_h = max(-1.0, min(6.0, b_h))
        holders_last = rate_last / 100.0 * players_at_last
        if players_flat:
            # Holders from the fit stay put (flat history); the new
            # cohort adds holders at NEW_COHORT_HOLD of the current rate.
            holders_target = holders_last + NEW_COHORT_HOLD * (
                rate_last / 100.0) * (players_target - players_last)
        else:
            log_h_target = a_h + b_h * x_target
            holders_target = math.exp(log_h_target) - 0.5
            holders_target = min(holders_target,
                                 holders_last * MAX_HOLDER_MULT)
            holders_target = max(holders_target, 0.0)
        holders_target = min(holders_target, players_target)

        rate_med = 100.0 * holders_target / players_target
        rate_med = clamp_rate(rate_med)

        resid_h = residuals_sd(txs, tys, a_h, b_h) if len(txs) >= 3 else 0.3
        spread = math.sqrt((SIGMA * math.sqrt(horizon_days)) ** 2
                           + resid_h ** 2 + resid_p ** 2
                           + refresh_spread ** 2)
        spread = max(0.35, min(2.5, spread))

        centre = logit(rate_med)
        qs = [round4(inv_logit(centre + z * spread)) for z in Z]
        # Rounding must not break the non-decreasing order.
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
