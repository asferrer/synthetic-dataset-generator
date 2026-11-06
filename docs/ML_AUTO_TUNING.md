# Effect Auto-Tuning

Two mechanisms tune the effect parameters against a set of real reference
images, using the domain-gap score (0 = indistinguishable from the references,
100 = maximum gap) as the objective. For the service layout see
[ARCHITECTURE.md](../ARCHITECTURE.md); for the UI workflow see
[USER_GUIDE.md](USER_GUIDE.md).

| Mechanism | Endpoint prefix | Evaluates a candidate by | Status |
|---|---|---|---|
| Closed-loop auto-tune | `/auto-tune` | Generating a probe batch with the candidate config and measuring it | Complete; used by the Generate page of the UI |
| ML-guided optimization | `/ml-optimize` | Measuring an existing image directory (see limitation below) | Experimental |

Both need a reference set of real images (`/domain-gap/references/*`).

## 1. Closed-loop auto-tune (`/auto-tune`)

Implemented in `services/gateway/app/routers/auto_tune.py`. Each iteration:

1. Generate a probe batch of `probe_size` images with the current
   `effects_config` (augmentor `POST /compose-batch`).
2. Compute the gap score against the reference set (domain_gap
   `POST /metrics/compute`).
3. Stop if the score is below `target_gap_score`, or if it has not improved for
   two consecutive iterations.
4. Otherwise ask the advisor (`POST /analyze`) for parameter suggestions, apply
   them to `effects_config` and repeat, up to `max_tune_iterations`.

When tuning ends and `auto_start_full` is true, the full generation job is
started with the tuned configuration.

Start a job with `POST /auto-tune/start`. Besides the generation fields shared
with `POST /augment/compose-batch` (`output_dir`, `effects`, `effects_config`,
`max_objects_per_image`, source and target fields, ...), the request takes:

| Field | Default | Range | Meaning |
|---|---|---|---|
| `reference_set_id` | required | | Reference set used to measure the gap |
| `target_gap_score` | 25 | 0-100 | Stop when the score falls below this |
| `max_tune_iterations` | 5 | 1-15 | Iteration limit |
| `probe_size` | 50 | 10-200 | Images per probe batch (50 or more for reliable metrics) |
| `auto_start_full` | true | | Start the full generation after tuning |

Poll `GET /auto-tune/jobs/{job_id}`. The status reports `status` (`pending`,
`tuning`, `generating`, `completed`, `failed`, `cancelled`), the current
`phase` (`probe_generating`, `computing_metrics`, `analyzing`, `adjusting`,
`full_generating`), `current_gap_score`, the per-iteration `history`, the
`tuned_effects_config`, `improvement_pct` and `full_generation_job_id`.
`DELETE /auto-tune/jobs/{job_id}` cancels a job and `GET /auto-tune/jobs`
lists them.

Computing the RADIO metrics is the slow step; the gateway allows up to 45
minutes per metrics call on small GPUs.

## 2. ML-guided optimization (`/ml-optimize`)

Implemented in `services/domain_gap/app/routers/ml_optimizer.py` and
`app/engines/` (`bayesian_optimizer_engine.py`, `predictor_engine.py`), proxied
by the gateway.

Components:

- **Predictor** (XGBoost): learns gap score as a function of the effect
  configuration from the recorded history and answers what-if queries without
  generating images.
- **Bayesian optimizer** (Optuna, TPE sampler): proposes configurations; the
  study is stored in SQLite.
- **Hybrid optimizer**: warm-starts the Optuna study with configurations the
  predictor ranks highly, then retrains the predictor on the new evaluations.

### Limitation

Each trial currently measures the gap of the images already in
`synthetic_dir`; it does not regenerate images with the proposed configuration
(see the `evaluate_config` docstring). All trials therefore see the same
score, and `best_config` reflects the sampler's search rather than an
experimentally validated result. Use `/auto-tune` to obtain validated
parameters.

### Endpoints

| Method and path | Purpose |
|---|---|
| `POST /ml-optimize/start` | Start a job; returns `job_id` |
| `GET /ml-optimize/jobs/{job_id}` | Status, best config, history, feature importance |
| `GET /ml-optimize/jobs` | List jobs |
| `DELETE /ml-optimize/jobs/{job_id}` | Cancel a job |
| `GET /ml-optimize/feature-importance?source=combined` | Parameter importance (`bayesian`, `ml_predictor` or `combined`) |
| `POST /ml-optimize/predict` | Predicted gap score for a config, with `confidence` (`low` below 15 recorded configurations, `medium` below 30, `high` otherwise) |

`POST /ml-optimize/start` body:

```json
{
  "synthetic_dir": "/app/output/baseline/images",
  "reference_set_id": "<set id>",
  "n_trials": 20,
  "probe_size": 50,
  "warm_start": true,
  "timeout_seconds": 3600,
  "parameter_ranges": {"caustics_intensity": [0.0, 0.2]},
  "current_config": {"color_intensity": 0.25}
}
```

`n_trials` is 3-50 (default 10), `probe_size` 20-200 (default 50),
`timeout_seconds` 60-7200. `parameter_ranges` and `current_config` are
optional. Job status values are `pending`, `optimizing`/`running`,
`completed`, `failed` and `cancelled`.

### Search space

| Parameter | Range |
|---|---|
| `color_intensity` | 0.0 - 0.5 |
| `blur_strength` | 0.0 - 2.0 |
| `underwater_intensity` | 0.0 - 0.5 |
| `caustics_intensity` | 0.0 - 0.3 |
| `shadow_opacity` | 0.0 - 0.3 |
| `lighting_intensity` | 0.2 - 1.0 |
| `motion_blur_probability` | 0.0 - 0.5 |
| `lighting_type` | categorical |
| `water_clarity` | categorical |

The categorical choices are defined in `bayesian_optimizer_engine.py` and do
not match the values accepted by the augmentor `EffectsConfig` schema exactly
(for example `moderate` for `water_clarity`); check a suggested config against
the schema before using it for generation.

### Persistence

Paths inside the `domain_gap` container, on the shared volume:

| File | Content |
|---|---|
| `/shared/config_history.json` | Evaluated configurations used to train the predictor |
| `/shared/gap_predictor_model.pkl` | Trained predictor, loaded at start-up |
| `/shared/optuna_studies/gap_optimization.db` | Optuna study (SQLite) |

The predictor is trained once there are enough recorded configurations
(roughly 10-15); before that `/ml-optimize/predict` returns
`"Predictor not trained yet"`.

## 3. Reading the results

Gap score: below about 30 is low, 30-60 moderate, above 60 high (these are the
bands used by the UI). Feature importance shows which parameters move the score
most; tune the highest-ranked ones first. Always compare scores computed
against the same reference set and a comparable `probe_size`.

## 4. Practical notes

- Use at least 50 images per probe; smaller batches give noisy metrics.
- Keep the reference set fixed while comparing configurations, and make sure it
  is representative of the target domain.
- If the score does not improve, the cause is usually outside the effect
  parameters: background and object quality, placement settings, or the choice
  of reduction technique (randomization, style transfer, diffusion refinement
  under `/domain-gap`).

## References

- [Optuna](https://optuna.readthedocs.io/)
- [XGBoost](https://xgboost.readthedocs.io/)
- Bergstra et al., [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)
