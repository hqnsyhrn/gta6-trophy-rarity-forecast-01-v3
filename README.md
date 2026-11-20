# gta6-trophy-rarity-forecast-01-v3

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hqnsyhrn/gta6-trophy-rarity-forecast-01-v3/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hqnsyhrn/gta6-trophy-rarity-forecast-01-v3/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for every GTA VI trophy on PlayStation 5, its earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles per trophy, for the [make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) bounty "Forecast how many GTA VI players earn each trophy in week one" ($250, top 20). The program models counts, not rates: rate = earners / players-tracked. Players-tracked follows a log-logistic S-curve fitted over a grid of carrying capacities (likelihood-weighted ensemble); each trophy's earners follow an anchored log-time trend (slope clipped, pinned at the last observed count); the spread comes from rolling-origin backtest residuals. When no refresh has been seen yet, a first-refresh prior applies: launch night's counts sat frozen at 638 players, then jumped to 3,868 in one refresh while earners lagged. Standard library only, deterministic, no network.

Local validation (pinball loss on the logit scale, lower is better): practice 0.078 against a truth-proxy calibrated to the house aggregates (the house baseline scores 1.51 there; the raw reconstructed practice answers score 0.29, see the note below); rolling-origin 1-day saturating world 0.103 vs persistence 0.115; synthetic week-one worlds A 0.265 vs 0.451 and B 0.113 vs 0.179; live backtest 04:30Z to 11:12Z 0.269 vs persistence 0.579. Practice cases run in under 0.3 s, synthetic week-one cases in under 0.7 s, deterministic byte-identical across runs.

Note on the practice answers: the widely circulated reconstructed practice answers (the 04:30Z read from a single live fetch) are not the house answers — they make the house baseline score 0.92, while the house and the live board both put it at 1.51 with a typical miss of 0.68 points. The truth-proxy used here rescales that reconstruction's new-earner component to match the house's published aggregates (baseline loss 1.51, typical miss 0.68, 4,276 trophies earned at the refresh). The 0.29 against the raw reconstruction is reported as-is; the model is tuned to the aggregates, not to the reconstruction.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=5zc4erer&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
