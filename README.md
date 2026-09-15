# Existence and Reportability of Ratio Confidence Intervals under Budgeted Adaptive Sampling

Code, data and raw outputs for the ICASSP 2027 submission of the same name.

A budgeted scheduler spends a fixed budget over `K` arms and the analyst reports
each arm's ratio `theta_a = E[r_a] / E[c_a]` of a Bernoulli outcome over a skewed
cost. The paper gives the exact criterion for when a Fieller interval exists, the
form of the set when it does not, and an accounting that makes the Fieller against
delta comparison well defined. Every number in the paper is a LaTeX macro produced
by the pipeline below; nothing is typed by hand.

## Layout

The tree mirrors the working directory the paper was built in, so the scripts run
with their paths unchanged.

```
experiments/                    new sampling for the paper
  ratio_ci.py                   the six constructions with explicit set-kind classification
  icassp_r3_selection.py        selected-arm and n-screen study
  icassp_r4_policy.py           fallback ablation, budgets, 30 environments, 12 policies
  icassp_r5_designs.py          regularised indices, two-stage design, Bayes and conf. seq.
  *_results.json, *_out.txt     their results and raw console output
archive/tmp/experiments/        the frozen collections and the simulator they came from
  expD_lib.py                   environment, sampler and library policies
  icassp_general.py             cost families and helpers
  icassp_r2_holdout.py          the frozen independent-seed holdout generator
  icassp_r2_holdout_*.json      its manifest and results
  icassp_realcost_out.txt       measured token-cost calibration (Qwen3-0.6B)
  *.pkl                         sufficient statistics of the frozen collections (11 MB)
reportability/                  the existence-condition study merged into the paper
  reportability.py              identity check, design rule, real success rates
  reportability.json            its results
paper-overleaf/tools/           the number pipeline
paper-overleaf/data/            generated JSON consumed by the pipeline
paper-overleaf/numbers.tex      the generated macros, committed for the diff check
paper-overleaf/table_*.tex      the generated tables
```

## Environment

Python 3.13.13 with `numpy==2.4.6`, `scipy==1.18.1`, `matplotlib==3.11.0`
(`pip install -r requirements.txt`). Nothing else is required; no GPU, no network.

## Reproducing the paper's numbers

From a clean clone, in order. Timings are wall clock on 40 cores.

```bash
# 1. recompute the frozen holdout from its sufficient statistics       ~4 min
python3 paper-overleaf/tools/analysis_r2.py

# 2. rebuild data/*.json from the frozen collections and raw outputs   ~10 s
python3 paper-overleaf/tools/extract_data.py

# 3. regenerate every macro and every table                           ~2 s
python3 paper-overleaf/tools/make_numbers.py

# 4. regenerate the two figures                                       ~15 s
python3 paper-overleaf/tools/make_figs.py
```

Step 3 overwrites `paper-overleaf/numbers.tex` and `paper-overleaf/table_*.tex`.
`git diff --stat` after the four commands is the reproduction check: it must be
empty.

Re-running the sampling itself is optional and slower.

```bash
python3 experiments/icassp_r4_policy.py      # ~3 min on 40 cores
python3 experiments/icassp_r5_designs.py     # ~6 min on 40 cores
python3 experiments/icassp_r3_selection.py   # ~50 s
python3 reportability/reportability.py       # 1316 s on 48 cores
```

## Seeds and settings

| what | value |
|---|---|
| frozen holdout | `MC = 400`, trajectory seeds `910000` to `910399`, environment seed `1`, manifest `icassp_r2_holdout_frozen.json`, script SHA-256 `a4a2afb6a1fc33ba99b156b9fc96128a2b9806394289f1577695bfba2ecb9bbb` |
| new sampling (R4) | `SEED0 = 940000`, `MC = 400` per configuration |
| 30-environment replication | environment seeds `1` to `30`, `MC_ENV = 100` each, trajectory seed `940000 + 1000 * env + i` |
| R5 designs | `SEED0 = 960000`, `MC = 400`, two-stage `eta = 0.05`, cap `n0 <= 40`, first-stage fractions `0.2, 0.4, 0.6, 0.8` |
| level | `alpha = 0.10` everywhere, unbiased (`n-1`) variance for every construction |
| benchmark | `K = 24` arms, LogNormal costs at log-scale `sigma = 0.35`, budget `30x` the expected cost of one pull of every arm |

## Data provenance

The `.pkl` files are the sufficient statistics (`n`, `sum r`, `sum c`, `sum r^2`,
`sum c^2`, `sum rc` per arm per trajectory) of collections drawn by the scripts in
`archive/tmp/experiments/`, not raw traces. Real success rates come from two
Qwen2.5-1.5B temperature-sampled evaluation suites of the project the study grew
out of; the token-cost calibration was measured locally with Qwen3-0.6B and is
recorded verbatim in `icassp_realcost_out.txt`.

One artefact of the original project, a 402 MB `.npz` of per-draw Fieller
diagnostics, is not in this repository; no result in the paper depends on it. The
raw `*_out.txt` files are kept verbatim, including the absolute paths of the
machines that produced them.

## License

MIT, see `LICENSE`.
