# User Guide

How to run the generator and use the web UI. For the service layout and the
generation pipeline see [ARCHITECTURE.md](../ARCHITECTURE.md); for automatic
effect tuning see [ML_AUTO_TUNING.md](ML_AUTO_TUNING.md); for synthesis profiles
see [SYNTHESIS_PROFILE.md](SYNTHESIS_PROFILE.md).

## 1. Requirements

- Docker and Docker Compose.
- NVIDIA GPU with CUDA drivers and the NVIDIA container runtime (depth,
  segmentation, augmentor and domain_gap use the GPU). Plan for at least 8 GB
  of VRAM and 16 GB of RAM (32 GB recommended).
- A Hugging Face token with access to the SAM3 model.
- A modern web browser.

## 2. Setup and start

1. Copy `.env.microservices.example` to `.env` and set
   `DATASET_PATH` (your data root), `HF_TOKEN` and optionally `DEPTH_MODEL`.
2. Organize the data root as expected by the services:

   ```
   <DATASET_PATH>/
     Backgrounds_filtered/    background images
     Objects/<class>/         object crops (PNG with alpha), one folder per class
   ```

   Both locations can also be chosen in the UI. Generated datasets are written
   under `./output`.
3. Start the stack:

   ```bash
   docker compose -f docker-compose.microservices.yml up -d
   docker compose -f docker-compose.microservices.yml ps
   ```

4. Open the UI at **http://localhost:3001**. The API is at
   http://localhost:8000 (Swagger UI at `/docs`).
5. Stop with `docker compose -f docker-compose.microservices.yml down`.

For UI development, run `npm install && npm run dev` in `frontend-vue/`
(Vite dev server on port 3000).

The UI is available in English and Spanish (language switcher in the header).
`Ctrl+K` opens page search.

## 3. UI map

Sidebar sections and routes:

| Section | Page | Route |
|---|---|---|
| Dashboard | Home: system metrics, active domain, recent jobs | `/` |
| Domains | Domain Manager, details, editor | `/domains` |
| Workflow | Prepare, Configure, Generate, Export & Split | `/prepare`, `/configure`, `/generate`, `/export` |
| Tools | Object Extraction, Domain Gap | `/tools/object-extraction`, `/tools/domain-gap` |
| System | Jobs and Services tabs | `/tools/system` |
| (viewer) | Dataset Viewer for a finished job | `/datasets/:jobId` |

## 4. Domains

A domain describes the environment being simulated: scene regions (with the
SAM3 prompts that detect them), object types (real-world size, density,
buoyancy behavior), effects, physics, presets and a region/object
compatibility matrix. The repository ships `underwater`, `aerial_birds` and
`fire_smoke`.

- **Domain Manager** lists domains. Create, edit, delete, import or export
  (JSON) them, and **activate** one; the active domain is the base
  configuration for generation.
- Built-in domains cannot be modified in place; saving a change stores an
  override that can be reverted.

## 5. Workflow

### 5.1 Prepare

- **Analyze:** upload a COCO JSON (or choose an existing one) and review image,
  annotation and category counts and the per-category distribution. Alternatively
  skip to *custom objects* mode and generate without a COCO dataset.
- **Sources:** choose the backgrounds directory (empty means solid-color
  backgrounds), the object directory (each subfolder is a class; empty means
  objects come from the analyzed dataset), which classes to include, and the
  output directory.

### 5.2 Configure

- **Effects & Placement:** presets (underwater, urban, minimal, aggressive and
  your own saved presets) and four groups of settings.

  | Group | Controls |
  |---|---|
  | Basic augmentation | blur, noise, brightness, contrast, rotation, scale, flips |
  | Underwater & realism | color correction, blur matching, lighting (ambient, spotlight, gradient), underwater tint and water clarity, shadows, caustics, motion blur, perspective warp |
  | Blending & edges | binary alpha, alpha feather, edge smoothing, blur budget |
  | Object placement | objects per image, size and area limits, overlap threshold, semantic placement |

  A preview button renders a sample composition with the current settings.
- **Object Sizes:** real-world size per class used for depth-aware scaling;
  each field is saved as you edit it.

### 5.3 Generate

Tabs:

| Tab | Content |
|---|---|
| Targets | Number of images to generate per category |
| Validation | Identity, perceptual-quality and anomaly thresholds, physics validation, *reject invalid images* |
| Processing | Concurrency limit, VRAM threshold, debug output |
| Lighting | Maximum light sources, intensity threshold, HDR estimation, water attenuation, depth category (shallow 0-5 m, mid 5-20 m, deep 20 m+) |
| Metadata | Dataset name, version, description, contributor, license |

Options also include depth estimation, segmentation and depth-aware placement.

Press **Start Generation** to launch the job; the page shows progress, counts
and job logs and survives a page reload. Jobs can be cancelled.

**Auto-Tune + Generate** first tunes the effect parameters against a reference
image set to reduce the domain gap, then generates the full dataset (see
[ML_AUTO_TUNING.md](ML_AUTO_TUNING.md)). It needs a reference set (section 6.2).

When the job finishes, open the **Dataset Viewer** to browse images with their
annotations overlaid, quality scores and depth source.

### 5.4 Export & Split

- **Export:** COCO JSON, YOLO (`.txt` per image plus `data.yaml`) or Pascal VOC
  (`.xml` per image), optionally copying the images.
- **Combine:** merge two or more datasets, optionally merging categories of the
  same name.
- **Split:** train/validation/test by percentage (must total 100%, optional
  stratification and seed) or k-fold.
- **Balance:** class-distribution analysis, class weights, under- and
  oversampling.

## 6. Tools

### 6.1 Object Extraction

Cuts objects out of an existing COCO dataset into per-class transparent crops
for use as generation sources. Inputs: COCO JSON, images directory, output
directory. Options: minimum object size, include masks, use SAM3 for precise
masks (slower), remove visually similar duplicates, padding around the box.
Progress is shown in the page and in the Jobs tab.

### 6.2 Domain Gap

Measures and reduces the difference between synthetic and real images.

| Tab | Purpose |
|---|---|
| References | Upload or register sets of real images (20-50 or more recommended) with name, description and domain |
| Validation | Compare a synthetic directory with a reference set; shows gap score (0-100), FID and KID, gap level, detected issues and parameter suggestions that can be applied to the pipeline |
| Randomization | Produce randomized variants (color jitter, noise, histogram matching) while preserving annotations |
| Style Transfer | Transfer the appearance of the reference set to synthetic images |
| Diffusion | Diffusion-based refinement (ControlNet, IP-Adapter, LoRA) |

Gap score bands used by the UI: below 30 low, 30-60 moderate, above 60 high.
The Generate page also shows a collapsible validation panel after a job
completes.

### 6.3 System

- **Jobs:** all jobs, filterable by status and source. Actions depend on the
  job: cancel, resume or retry (augmentor jobs), regenerate, delete. The list
  refreshes automatically.
- **Services:** health, port and latency of each service, with manual refresh.

## 7. Recommended practice

1. Start with a small batch (20-50 images) and inspect it in the Dataset Viewer
   before launching a full run.
2. Use backgrounds from the environment where the model will be deployed.
3. Validate against real reference images before generating a large dataset;
   iterate on the suggested parameters, or let Auto-Tune do it.
4. Check class balance (Export & Split, Balance) and use stratified splits.
5. Check the Services tab before long jobs.
