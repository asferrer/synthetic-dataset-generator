# Synthesis profiles

The augmentor composes synthetic images by pasting object crops onto background images. A *synthesis profile*
(`services/augmentor/app/synth_profile.py`, class `SynthProfile`) groups the options that control how faithful the
pasted objects are to the statistics of real images. Every option is off by default, so a request without a profile
reproduces the original (`legacy`) behaviour.

A profile is selected per request (`synth_profile`) and any field can be overridden (`synth_overrides`).

| Profile | Content |
|---|---|
| `legacy` | No option active. |
| `v10` | Class-faithful scale and object count, per-paste augmentation, crop reuse cap, per-image seeds and trace, output resolution and JPEG recompression, round-robin backgrounds, haze limited from statistics, seabed placement per class. |
| `v11` | `v10` without per-paste augmentation and with the rectangular-crop filter. Further options are enabled through the override sets below. |

## Options

Statistics files live in `services/augmentor/app/data/` and are computed from the **training split only**
(`scripts/build_*_stats.py`).

| Group | Options | Effect |
|---|---|---|
| Scale and count | `faithful_scale`, `faithful_count`, `min_object_px`, `max_crop_upscale`, `max_objects_cap` | Object size per class and objects per image follow the empirical distribution (`real_stats_train.json`). |
| Placement | `seabed_fraction`, `water_column_behavior`, `position_from_stats`, `position_margin`, `border_touch_fraction` | Per-class probability of resting on the seabed; positions follow the real position histogram, with a margin to the image border for boxes that do not touch it (`position_stats_train.json`). |
| Crop selection | `crop_rect_filter`, `crop_scale_range`, `max_crop_reuse`, `extra_objects_dirs` | Discards crops whose mask fills almost its whole box, picks the crop whose native resolution keeps the paste scale in range, caps reuse, adds extra crop libraries. |
| Real backgrounds | `copy_paste_real`, `real_annotations_path`, `real_overlap_max` | Pastes onto real training images and keeps their real boxes (`origin = "real"` in the COCO output). |
| Colour and medium | `harmonize*`, `medium_transfer*`, `haze_from_stats`, `copy_paste_haze_max`, `copy_paste_blur_max` | Local colour transfer in a ring around the box, ring-to-ring medium transfer in linear RGB, caps on global haze and blur when the background is a real photo. |
| Edges | `edge_matting*`, `single_resample`, `match_sharpness*` | Trimap matting (`pymatting`), single-warp resampling, sharpness and grain matched to real objects of the class (`real_sharpness_train.json`). |
| Visibility | `visibility_filter`, `visibility_min_contrast`, `visibility_percentile`, `visibility_ring` | Rejects pasted objects whose local contrast against the surrounding ring falls below a threshold taken from low percentiles of real objects (`visibility_percentiles_train.json`). |
| Background matching | `background_match`, `background_match_*`, `local_patch_match` | Chooses crops whose source image has a global LAB colour close to the background (`background_match_train.npz`). |
| Annotation | `tight_bbox`, `save_masks` | Box fitted to the pasted mask instead of the union with shadows and global effects; optional RLE mask per object. |
| Output | `output_resolution`, `jpeg_recompress`, `trace`, `per_image_seed` | Output size and compression follow real images; per-object trace in the COCO; reproducible per-image seeds. |

## Configuration used for the thesis batch

The final batch is the variant `s_drop` of `tools/run_variants.py`: profile `v11` with

- `copy_paste_real`, `harmonize`, the visibility filter at per-class thresholds, `single_resample`,
  `match_sharpness`, `crop_scale_range = (0.6, 1.5)`, `position_from_stats` (set `V11_POSITION_OVERRIDES` in
  `synth_profile.py`) and `tight_bbox`;
- crop libraries `Extracted_objects_train_drop` and `crops_poly_drop`, which hold only the crops that pass the
  library filter described below (original masks, anomalous crops removed).

Request parameters (effects, per-class targets, validation) are in `configs/batch_request.json`.

## Inputs

All inputs come from the training split of the detection dataset; none from validation or test.

| Input | Description |
|---|---|
| `Backgrounds_filtered/` | Background images (underwater scenes), restricted to training-split images. |
| `real_train/` + `annotations.json` | Real training images with their COCO boxes, used as backgrounds for copy-paste. Built by `scripts/build_real_train.py`; files are named `<image id:05d>_<original name>`. |
| `Extracted_objects_train/` | RGBA crops of annotated training objects, produced by the segmentation service (`/extract/objects`, SAM 3 with the class name as text prompt, matched to the annotation by box IoU). |
| `crops_poly/` | Crops cut from annotated polygons (annotations that carry a polygon). |

## Crop library filter (`tools/clean_crops.py`)

For every crop the script locates it in its source image by template matching on the alpha channel, relates it to its
annotation, and decides whether to keep it:

1. The alpha is clipped to the annotated box enlarged by 4 px, small components (< 2 % of the area) are removed.
2. The crop is discarded if more than 50 % of the mask lay outside the box plus margin, if the remaining area is
   below 100 px^2, or if the final coverage of the box is outside [0.10, 0.97] ([0.03, 0.97] for fishing nets).

`--mode clean` writes the clipped masks; `--mode drop` applies the same decisions but copies the retained crops
unchanged. The batch uses `--mode drop`.

## Reproducing the batch

```bash
# 1. Real-background folder and statistics of the training split (CPU)
python scripts/build_real_train.py --train <dataset>/annotations/train.json --images-root <dataset> --out gen_inputs/real_train
python scripts/measure_real_visibility.py --out out   # percentiles behind visibility_percentiles_train.json
python scripts/build_real_stats.py --train <train.json> --out services/augmentor/app/data/real_stats_train.json
python scripts/build_position_stats.py --train gen_inputs/real_train/annotations.json --out services/augmentor/app/data/position_stats_train.json
python scripts/build_sharpness_stats.py --train gen_inputs/real_train/annotations.json --out services/augmentor/app/data/real_sharpness_train.json
python scripts/build_background_match.py     # needs GROUPS_JSON, the group assignment of the training images

# 2. Crop libraries (CPU)
python tools/clean_crops.py --mode drop --workers 8

# 3. Generation inside the augmentor container (needs the depth and segmentation services, GPU)
docker cp tools/run_variants.py synth-augmentor:/tmp/run_variants.py
docker cp configs/batch_request.json synth-augmentor:/tmp/batch_request.json
docker exec synth-augmentor python /tmp/run_variants.py --request /tmp/batch_request.json \
    --variants s_drop --n 2000 --seed 21 --out /app/output/synthetic_batch/chunk_1
```

The batch of 12,000 images was generated as six chunks of 2,000 images with seeds 21 to 26. Each chunk writes
`<out>/s_drop/synthetic_dataset.json` and its images; chunks are merged into a single COCO file by the benchmark
repository.

## Tests

```bash
cd services/augmentor && python -m pytest tests -q      # no GPU or services required
```

## Bundled statistics

`services/augmentor/app/data/` holds the statistics and descriptors computed from the training split:
`real_stats_train.json`, `position_stats_train.json`, `real_sharpness_train.json`, `visibility_percentiles_train.json`
and `background_match_train.npz`. The descriptor table lists, for each training image, its id, group, global LAB colour
and coarse histogram, and for each crop its source image and ring colour. Image ids and group numbers are those of the
public `annotations/train.json` of the dataset release; the image names are `<id:05d>_<original file name>`.
