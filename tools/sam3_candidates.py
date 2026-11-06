"""Phase 1 of crop refinement: SAM3 mask candidates for every annotated object (standalone, no repository imports).

Reads a `jobs.json` (written by `refine_crops_sam3.py plan`) and, for each object, saves in `<out>/cand/<ann>.npz` the
candidate masks SAM3 returns for three prompt types:
  - `box`      : the annotated box as a positive example (no text).
  - `boxtext`  : the annotated box + a text concept (the class name).
  - `text`     : the concept only (once per image; used to check that the mask belongs to the class).
The image is encoded ONCE and reused for all objects of the same image (`vision_embeds`).
Masks are stored cropped to the region (the box grown by `--margin`) and bit-packed, together with the full area of
each mask so the overflow beyond the region can be measured.

Run it inside the segmentation container (`synth-segmentation`: it has transformers and the cached `facebook/sam3`
weights), without touching the running service: the process loads its own copy of the model.

CPU, lowest priority, 8 threads:
    docker exec -i -e CUDA_VISIBLE_DEVICES= synth-segmentation nice -n 19 python - --jobs /app/datasets/crops_sam3/jobs.json \
        --out /app/datasets/crops_sam3 --device cpu --threads 8 --seed 21 < tools/sam3_candidates.py
GPU: same command with `-e CUDA_VISIBLE_DEVICES=0`, `--device cuda` and without `nice`.
Resumable: objects whose .npz already exists are skipped.
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

CONCEPTS = {"Bottle": "bottle", "Can": "can", "Fishing_Net": "fishing net", "Glove": "glove", "Mask": "face mask",
            "Metal_Debris": "metal debris", "Plastic_Debris": "plastic debris", "Tire": "tire"}


def region_for(box, size, margin):
    """Region (x0, y0, x1, y1) in pixels: box [x, y, w, h] grown by `margin` per side and clipped to the image."""
    x, y, w, h = box
    w_img, h_img = size
    x0, y0 = max(0, int(x - margin * w)), max(0, int(y - margin * h))
    x1, y1 = min(w_img, int(x + w * (1 + margin)) + 1), min(h_img, int(y + h * (1 + margin)) + 1)
    return x0, y0, x1, y1


def pack_candidates(items, region):
    """items = [(mode, score, HxW bool mask)] -> dict for np.savez_compressed (masks cropped to the region)."""
    import numpy as np
    x0, y0, x1, y1 = region
    rw, rh = x1 - x0, y1 - y0
    packed, meta = [], []
    for mode, score, m in items:
        crop = np.zeros((rh, rw), bool)
        crop[:] = m[y0:y1, x0:x1]
        packed.append(np.packbits(crop.ravel()))  # same length for all: same region
        meta.append((mode, float(score), int(m.sum()), int(crop.sum())))
    return {"region": np.array(region), "shape": np.array([rh, rw]),
            "modes": np.array([t[0] for t in meta]), "scores": np.array([t[1] for t in meta], np.float32),
            "area_full": np.array([t[2] for t in meta], np.int64), "area_region": np.array([t[3] for t in meta], np.int64),
            "bits": np.array(packed, np.uint8).reshape(len(packed), -1)}


class Sam3Runner:
    def __init__(self, device, threshold):
        import torch
        from transformers import Sam3Model, Sam3Processor
        self.torch, self.device, self.threshold = torch, device, threshold
        self.proc = Sam3Processor.from_pretrained("facebook/sam3")
        self.model = Sam3Model.from_pretrained("facebook/sam3", low_cpu_mem_usage=True).to(device).eval()

    def encode(self, img):
        inp = self.proc(images=img, return_tensors="pt")
        with self.torch.no_grad():
            return self.model.get_vision_features(pixel_values=inp["pixel_values"].to(self.device))

    def run(self, img, vision, text=None, box=None):
        """Instances [(score, HxW bool mask)] for one prompt (text and/or xyxy box)."""
        kw = {}
        if text:
            kw["text"] = text
        if box is not None:
            kw.update(input_boxes=[[list(box)]], input_boxes_labels=[[1]])
        inp = self.proc(images=img, return_tensors="pt", **kw).to(self.device)
        sizes = inp["original_sizes"].tolist()
        fwd = {k: v for k, v in inp.items() if k not in ("pixel_values", "original_sizes", "reshaped_input_sizes")}
        with self.torch.no_grad():
            out = self.model(vision_embeds=vision, **fwd)
        res = self.proc.post_process_instance_segmentation(out, threshold=self.threshold, mask_threshold=0.5,
                                                           target_sizes=sizes)[0]
        return [(float(s), m.cpu().numpy().astype(bool)) for s, m in zip(res["scores"], res["masks"])]


def process_image(runner, img, objs, out_dir, modes, margin):
    import numpy as np
    vision = runner.encode(img)
    text_items = []
    classes = {o["cls"] for o in objs}
    if "text" in modes:
        text_by_cls = {c: runner.run(img, vision, text=CONCEPTS.get(c, c.lower())) for c in classes}
    for o in objs:
        x, y, w, h = o["box"]
        xyxy = (x, y, x + w, y + h)
        items = []
        if "box" in modes:
            items += [("box", s, m) for s, m in runner.run(img, vision, box=xyxy)]
        if "boxtext" in modes:
            items += [("boxtext", s, m) for s, m in runner.run(img, vision, CONCEPTS.get(o["cls"], o["cls"].lower()), xyxy)]
        if "text" in modes:
            items += [("text", s, m) for s, m in text_by_cls[o["cls"]]]
        region = region_for(o["box"], img.size, margin)
        np.savez_compressed(out_dir / "cand" / f"{o['ann']}.npz", **pack_candidates(items, region))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--modes", default="box,boxtext,text")
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    jobs = json.loads(Path(args.jobs).read_text(encoding="utf8"))
    out = Path(args.out)
    (out / "cand").mkdir(parents=True, exist_ok=True)
    modes = set(args.modes.split(","))
    by_img = defaultdict(list)
    for o in jobs["objects"]:
        if not (out / "cand" / f"{o['ann']}.npz").exists():
            by_img[o["image"]].append(o)
    runner = Sam3Runner(args.device, args.threshold)
    t0 = time.time()
    for i, (path, objs) in enumerate(sorted(by_img.items())):
        img = Image.open(Path(jobs["images_root"]) / path).convert("RGB")
        process_image(runner, img, objs, out, modes, args.margin)
        print(f"[{i + 1}/{len(by_img)}] {path} objects={len(objs)} t={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
