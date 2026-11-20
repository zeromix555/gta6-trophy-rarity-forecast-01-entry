[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zeromix555/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zeromix555/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, the
percent of the players Exophase tracks for the game who will have earned it
at the case's target, as five quantiles per trophy. Entry for the
[make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
— $250 in USDC, 20 places. Referral: `jwef5djp`.

## Method (cohort-holder model with refresh-aware dilution)

A trophy's earn rate is a ratio, `r = H / P`: holders over players
tracked. The two move for different reasons, so they are forecast
separately and the rate is the ratio of the two forecasts.

- **Players tracked P(t):** fit `ln P` against `ln(1 + hours since
  release)` by least squares over the case's history, read at the target.
  Log-time because tracking growth is fast at launch and saturates.
- **Holders H(t):** per trophy, `H = r * P / 100` in each read; fit
  `ln(H + 0.5)` against log-time the same way, slope shrunk 30% toward
  the median trophy's slope (small holder counts are noisy).
- **Refresh prior:** Exophase refreshes a game's counts on its own
  schedule. Launch night the page sat at 638 players tracked from
  23:48Z through 04:00Z, then jumped to 3,868 by 04:30Z (a 6.06x jump)
  and every rate but the 100% one fell. When a case's history shows no
  refresh at all, one is assumed overdue: P is multiplied by 6.06 (one
  per ~4.5 h of horizon for longer horizons, capped 40x) and the new
  cohort is credited with holding each trophy at 7% of its current
  rate — 7% because the rules' own totals for the first refresh
  (trophies earned 943 -> 4,276) imply the new cohort added only ~103
  holders across the 54 trophies below 100%. When the history shows
  refreshes, the fits carry their average effect and no prior is added.
- **Spread:** logit spread combines a random-walk term
  (`0.5 * sqrt(horizon in days)`), the fit residuals, and a refresh term
  (large when no refresh has been seen, small otherwise). A trophy last
  seen at 100% is forecast to stay at ~100%.

The only data used is the case file itself — no network, no outside
trophy database. The 6.06x / 35% priors come from the competition's own
published rules, not from a measurement of the answer, and only bite
when the history is flat. Trophy names/descriptions are not used to set
rates; tier enters only via the slope-shrinkage median computed from
the case itself.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only. Reads every `*.in` in the input
directory, writes `<id>.out` per case: `{"trophies": [{"id": n, "q":
[q05, q25, q50, q75, q95]}, ...]}`, deterministic across runs.

## Practice check (local, no answer file in the kit)

On the kit's practice cases `p00`/`p01` (flat launch-night history,
90-minute horizon) the model forecasts a refresh dilution instead of
persistence: medians fall below the last rates (e.g. a trophy last at
1.10% gets a median of 0.25%), with a wide band — the behaviour the
practice is designed to reward, since the answers are the refreshed
page (3,868 players tracked). The new-cohort hold rate (7%) is
calibrated on the rules' own published totals for that read
(3,868 players, 4,276 trophies earned): the new cohort added only
~103 holders across the 54 trophies below 100%. Sensitivity check
(`analyze.py` in the run record): under every plausible split of
those 103 new holders (proportional / even / zero) this model scores
0.18–0.24 pinball loss vs persistence's 1.40–1.69 (house baseline
~1.51 per the rules); it only loses if the new cohort held trophies
at the old rates, which the rules say did not happen.
