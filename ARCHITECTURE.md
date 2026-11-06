# Architecture

The generator builds synthetic object-detection datasets by compositing object
crops onto background images and emitting COCO annotations. It is split into
FastAPI microservices behind a single gateway, with a Vue 3 web UI.

For day-to-day usage see [docs/USER_GUIDE.md](docs/USER_GUIDE.md). For the
closed-loop effect tuning see [docs/ML_AUTO_TUNING.md](docs/ML_AUTO_TUNING.md).
For the synthesis profiles (`v10`, `v11`) see
[docs/SYNTHESIS_PROFILE.md](docs/SYNTHESIS_PROFILE.md).

## 1. Services

```
            +---------------------------------------+
            |   frontend-vue (nginx, host :3001)    |
            |   /api/* -> gateway                   |
            |   /segmentation/* -> segmentation     |
            +-------------------+-------------------+
                                |
                      +---------v---------+
                      |  gateway  :8000   |
                      +--+----+-----+---+-+
                         |    |     |   |
            +------------+  +-+     |   +------------+
            |               |       |                |
       +----v----+   +------v---+ +-v--------+  +----v-------+
       | depth   |   | segment- | | augmentor|  | domain_gap |
       | :8001   |   | ation    | | :8004    |  | :8005      |
       | (GPU)   |   | :8002    | | (GPU)    |  | (GPU)      |
       +---------+   | (GPU)    | +----------+  +------------+
                     +----------+
```

The augmentor calls depth and segmentation directly. Effects are applied
in-process inside the augmentor (`composition/`, `effects_underwater.py`).

| Service | Port | GPU | Memory limit | Role |
|---|---|---|---|---|
| gateway | 8000 | no | 4 GB | Single entry point: routing, health aggregation, dataset/filesystem/domain endpoints, job listing |
| depth | 8001 | yes | 12 GB | Monocular depth estimation (Depth Anything V3) |
| segmentation | 8002 | yes | 12 GB | SAM3 scene analysis, object extraction, text-prompted segmentation |
| augmentor | 8004 | yes | 24 GB | Composition, effects, semantic placement, validation, batch jobs |
| domain_gap | 8005 | yes | 16 GB | Domain-gap metrics, reference sets, randomization, style transfer, optimization |
| frontend | 3001 (container :3000) | no | 256 MB | Vue 3 + TypeScript + Pinia + Tailwind + vue-i18n (ES/EN), served by nginx |

Ports, limits and volumes are defined in `docker-compose.microservices.yml`.
Job state for the gateway, augmentor and domain_gap is kept in a shared SQLite
database (`JOBS_DB_PATH`, default `/shared/db/jobs.db`, implemented in
`services/shared/job_database.py`), which lets interrupted jobs be resumed
after a restart.

### 1.1 Gateway

- Proxies to the downstream services through an HTTP client with automatic
  retries on connection errors and timeouts (`app/services/client.py`).
- Own endpoints (`app/main.py`): `GET /ping`, `GET /health`, `GET /info`,
  `POST /generate/image`, `POST /generate/batch`, `GET /jobs/all`,
  `GET /jobs/active`, `GET /jobs/{job_id}`, object-size configuration under
  `/config/object-sizes`.
- Routers (`app/routers/`):

| Prefix | Target |
|---|---|
| `/augment` | augmentor (`compose`, `compose-batch`, `jobs`, `validate`, `lighting`, `datasets`) |
| `/segmentation` | segmentation (`extract/*`, `sam3/*`, `segment-text`) |
| `/domain-gap` | domain_gap (references, metrics, analyze, randomize, style transfer, optimize, diffusion refinement) |
| `/ml-optimize` | domain_gap (see ML_AUTO_TUNING.md) |
| `/auto-tune` | gateway-side closed loop over augmentor and domain_gap (see ML_AUTO_TUNING.md) |
| `/domains` | domain configuration CRUD, activation, import/export, compatibility |
| `/datasets`, `/fs` | dataset analysis, upload, combine, split, k-fold, export, category rename/delete |
| `/filesystem` | directory and file browsing for the UI |

### 1.2 Depth

- Model chosen with `DEPTH_MODEL` (`DA3-SMALL`, `DA3-BASE` default,
  `DA3-LARGE`, `DA3-GIANT`; the two larger ones are non-commercial licensed).
- Endpoints: `GET /health`, `GET /info`, `POST /estimate`,
  `POST /estimate-batch`, `POST /depth/estimate`.
- The depth map (0 = near, 1 = far) drives object scaling, the Beer-Lambert
  attenuation, depth-aware caustics and depth-based occlusion.

### 1.3 Segmentation

- SAM3 is loaded in the background (`GET /model-status`); a Hugging Face token
  with access to the model is required (`HF_TOKEN`).
- Scene analysis (`POST /analyze`) returns a region map with ids
  `0 unknown, 1 open_water, 2 seafloor, 3 surface, 4 vegetation, 5 rocky,
  6 sandy, 7 murky`. `POST /check-compatibility` and `POST /suggest-placement`
  score object/region compatibility.
- Object extraction: `POST /extract/analyze-dataset`, `/extract/objects`,
  `/extract/custom-objects`, `/extract/single-object`, `/extract/imagenet`,
  with job status under `/extract/jobs`.
- SAM3 tools: `POST /sam3/segment-image`, `/sam3/convert-dataset` (bounding
  boxes to masks), `POST /segment-text`.
- Auto-labeling endpoints (`/labeling/*`) exist on the service itself but are
  neither proxied by the gateway nor used by the UI.

### 1.4 Augmentor

- Composition (`POST /compose`, `POST /compose-batch`), job management
  (`/jobs`, `/jobs/{id}/resume|retry|regenerate-dataset|logs`), generated
  dataset access (`/datasets`), `POST /validate`, `POST /lighting`, and
  object-size configuration.
- Main modules in `services/augmentor/app/`:

| Module | Responsibility |
|---|---|
| `composer.py` | Backward-compatible facade: `from app.composer import ImageComposer` |
| `composition/` | `ImageComposer` implementation, split into mixins (`core.py` assembles them): `object_placement.py` (position selection), `transform.py` (scale/rotation, depth-aware scaling, boxes), `effects.py` (harmonization and realism effects), `validation.py` (rejection checks), `single_image.py` (`compose`), `batch.py` / `batch_parallel.py` (batch planning), `output.py` (COCO and reports), `debug_viz.py`, `scene_analysis.py` (depth, lighting), plus `geometry.py` and `results.py` (pure helpers and result types) |
| `placement.py` | Semantic placement: maps each class's buoyancy behavior (`sink`/`float`/`hover`, from the domain config) to compatible scene regions |
| `effects_underwater.py` | Beer-Lambert attenuation, caustics, depth occlusion |
| `lighting_engine.py` | Light-source detection and shadow parameters |
| `validators.py` | Quality (LPIPS, Isolation Forest, composition) and physics checks |
| `synth_profile.py`, `faithful_scale.py`, `copy_paste.py`, `object_planner.py`, ... | Synthesis profiles (see SYNTHESIS_PROFILE.md) |
| `caches.py` | LRU caches for depth maps, scene analyses and decoded objects |

### 1.5 Domain gap

- Reference sets: `POST /references/upload|create|from-directory`,
  `.../add-batch`, `.../finalize`, `GET/DELETE /references[/{id}]`.
- Metrics (`POST /metrics/compute`, `/metrics/compare`): embeddings from
  C-RADIOv4-H (RADIO-MMD, FD-RADIO, PRDC), Inception features (FID, KID), CLIP
  (CMMD) and LAB color distribution; combined into a 0-100 gap score and level.
- `POST /analyze` returns issues and parameter suggestions; the rule-based
  advisor lives in `engines/advisor_engine.py`.
- Reduction techniques: `/randomize/*`, `/style-transfer/*`, `/diffusion/*`
  (refinement, LoRA training and validation), `POST /optimize`.
- `/ml-optimize/*`: XGBoost predictor plus Optuna search (ML_AUTO_TUNING.md).

## 2. Generation pipeline

Executed by the augmentor for every output image of a batch job:

1. **Preparation.** Pick a background and the objects to place according to the
   per-class targets and `max_objects_per_image`.
2. **Depth and scene analysis.** Depth map from the depth service; region map
   from the segmentation service (cached per background).
3. **Lighting estimation.** Detect up to `max_light_sources`, dominant
   direction, color temperature and ambient intensity (`lighting_engine.py`).
4. **Placement and scaling.** Positions come from semantic placement when
   `semantic_placement` is on, otherwise random; overlap is limited by
   `overlap_threshold` with up to `try_count` attempts. The depth-aware scale
   converts the normalized depth to a distance (0.5 m to the assumed visibility
   limit), applies a 1/distance perspective factor relative to a reference
   capture distance and uses the real-world size of the class. Synthesis
   profiles can replace this with the empirical size distribution of real data.
5. **Shadows.** Silhouette-based shadow projected opposite to the detected
   light, softened with `shadow_blur`.
6. **Effects**, in this order when enabled: color correction (LAB transfer),
   blur matching, lighting, underwater (flat tint, or per-channel Beer-Lambert
   with `underwater_physical`), caustics (seeded per image, depth-aware with
   `caustics_depth_aware`), motion blur, edge smoothing, Poisson blend.
   Objects are composited with alpha by default; `depth_occlusion` removes
   object pixels hidden by nearer geometry and retries objects that remain
   less than 25% visible.
7. **Validation.** Quality: LPIPS-based perceptual score, Isolation Forest
   anomaly score and composition score against the thresholds in
   `configs/config.yaml`. Physics: gravity, buoyancy, scale plausibility and
   occlusion. With `reject_invalid` a failing image is discarded and the
   attempt is repeated.
8. **Output.** Image written to `output_dir`, COCO annotations accumulated
   (boxes are recomputed after global effects), per-image `quality` scores and
   `depth_source` recorded, and the job checkpointed so it can be resumed.

Underwater effect model (per BGR channel, `apply_beer_lambert`):

```
I(d) = J * exp(-c * d) + B * (1 - exp(-c * d))
```

with `J` the unattenuated radiance, `d` the normalized depth, `c` the channel
attenuation for the selected `water_clarity` (`clear`, `murky`, `very_murky`)
and `B` the veil light (`water_color`).

## 3. Configuration

- `configs/config.yaml`: default augmentation, lighting and validation
  parameters and thresholds.
- `config/domains/*.json`: domain definitions (regions, objects with
  real-world size and physics properties, effects, presets), validated against
  `domain_schema.json`. Shipped domains: `underwater`, `aerial_birds`,
  `fire_smoke`; `_template.json` is a starting point.
- `config/object_sizes.json`: default real-world object sizes.
- Effect defaults are the `EffectsConfig` schema in
  `services/augmentor/app/models/schemas.py`; the realism options
  `underwater_physical`, `caustics_depth_aware` and `depth_occlusion` default to
  off.
- Environment variables are listed in `.env.microservices.example`.

## 4. Technology

| Component | Technology |
|---|---|
| Services | FastAPI, Python 3.11-3.12 (per service image) |
| GPU | PyTorch, CUDA 12.x base images |
| Depth | Depth Anything V3 |
| Segmentation | SAM3 |
| Quality | LPIPS, Isolation Forest (scikit-learn) |
| Domain gap | C-RADIOv4-H, Inception, CLIP, Optuna, XGBoost |
| Persistence | SQLite (jobs), shared Docker volumes |
| Frontend | Vue 3, TypeScript, Pinia, Tailwind CSS, vue-i18n, Vite |
| Orchestration | Docker Compose |
| Export formats | COCO, YOLO, Pascal VOC |
