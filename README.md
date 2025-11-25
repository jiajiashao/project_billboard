Project Billboard – Run Guide
=============================

Clone & Setup (60s)
-------------------
```bash
# clone & env
git clone <YOUR_REPO_URL> project_billboard
cd project_billboard
conda env create -f envs/environment.yml
conda activate billboard   # or the name defined in environment.yml

# sanity: python and (optional) CUDA in THIS env
python -c "import sys; print(sys.version)"
python -c "import torch; print('cuda=', torch.cuda.is_available(), torch.version.cuda if torch.cuda.is_available() else None)"

# (Windows) if you hit cv2 import errors
pip install --upgrade opencv-python-headless

# ffmpeg present?
ffmpeg -version
```

This repo contains eight entry-point pipelines (SAM-2 and XMem variants) plus supporting prompt/shot-detection tooling. Paths below are repo‑relative unless stated.

Environments
------------
- Conda/venv specs: `envs/environment.yml`, `envs/requirements.txt`.
- Create/activate an env that matches your runner (CUDA for XMem; CPU/MPS is fine for SAM-2 seeding only).
- Example setup: `conda env create -f envs/environment.yml` or `python -m venv .venv && pip install -r envs/requirements.txt`; then `conda activate <env>` or `source .venv/bin/activate`.

Data layout (relative to `--data-root` or `--root`)
---------------------------------------------------
- `data/clips/<clip_id>.mp4` (input videos)
- `data/frames/<clip_id>/*.jpg` (pre-extracted frames, if present)
- `data/gt_frames/<clip_id>/frame_*.json` (LabelMe-style GT masks; used for fallback seeding/metrics)
- Prompts list: `prompts_list.txt`

Frame extraction (optional)
```bash
# place your video at: data/clips/clip_corner.mp4
# extract frames for faster debugging
ffmpeg -i data/clips/clip_corner.mp4 -qscale:v 2 data/frames/clip_corner/%06d.jpg
```

Models / weights
----------------
**Place or auto‑download weights exactly to these paths:**

- **SAM‑2** (download per upstream SAM‑2 instructions)
  - `sam2/**/models/<sam2_weights>.pth`

- **XMem checkpoint** (CUDA only)
  - Download the model from: [Google Drive](https://drive.google.com/drive/folders/1rPmJGm8JVt5ewECwvu5V3vtIOFyaUlKE?usp=sharing), and save it under `project_billboard/xmem/model/xmem/`  ← from XMem releases

- **OWL‑ViT** (auto‑prompt)
  - Pulled automatically by HuggingFace on first run: `google/owlv2-base-patch16-ensemble`

- **Moondream** (auto‑prompt)
  - Pulled on first run. If you hit Pillow conflicts, pin `pillow==10.4.*` in your env.

- **YOLO11 finetuned (optional)**
  - `sam2/sam2_yolo11/weights/best.pt`
  - `xmem/xmem_yolo11/weights/best.pt`

Outputs
-------
- SAM-2: under `sam2/runs/<config>/<run_id>/` (overlay MP4, per-shot annotated JPGs, `re_prompts_*.csv`, `pilot_*.log`, summary CSVs).
- XMem: under `xmem/outputs/<run_id>/` (or `xmem/outputs/G_<clip>/runs/<run_id>/` for YOLO/XMem). You get masks PNGs, overlay MP4 (when XMem runs), `re_prompts_*.csv`, per-shot seed JPGs, pilot log, summary CSV.

Example: after a SAM‑2 GT baseline on `clip_corner`, look under `sam2/runs/*/<run_id>/clip_corner/overlay.mp4` and `.../pilot_*.log`.

Known requirements / pitfalls
-----------------------------
- XMem **requires a CUDA-capable GPU**; on CPU/Mac it will exit early. Use `--skip-xmem` to debug seeds only.
- `cv2` import errors on some Python builds; reinstall OpenCV (`pip install opencv-python-headless`) in the active env.
- HuggingFace model downloads may warn about symlinks; use `HF_HUB_DISABLE_SYMLINKS_WARNING=1` if needed.

Entry points and key CLI flags
------------------------------
- Prompts precedence: `--prompts` (CLI) > `--prompts-file` > built‑in defaults.
- Path args differ: SAM‑2 uses `--data-root`, XMem uses `--root`.

1) `sam2/sam2_gt/main.py`
   - Args: `--data-root`, `--clips <id[,id2...]>`, `--device {cuda,cpu,mps}`, `--stride`, `--run-id`.
2) `sam2/sam2_gt_shotdetector/main.py`
   - Args: same as above plus shot detection baked into pipeline.
3) `sam2/sam2_moondream/main.py`
   - Args: `--auto-prompt`, `--prompts "p1; p2"`, `--moondream-threshold`, `--clips`, `--device`.
4) `sam2/sam2_owlvit/main.py`
   - Args: `--auto-prompt`, `--prompts "p1; p2"` or `--prompts-file`, `--clips`, `--device`, `--owlvit-score-thr`.
5) `sam2/sam2_yolo11/main.py`
   - Args: `--yolo-model <path>`, `--source <mp4>`, `--device`, `--yolo-conf`.
6) `xmem/xmem_gt/main_gt.py`
   - Args: `--root`, `--clip`, `--device cuda`, `--stride`, `--width`.
7) `xmem/xmem_owlvit/main_owlvit.py`
   - Args: `--root`, `--clip`, `--device {cuda,cpu}`, `--shot-detect`, `--auto-prompt`, `--prompts "p1; p2"` or `--prompts-file`, `--owlvit-score-thr`, `--autoprompt-fallback {gt,skip}`, `--seed-erosion`, `--owlvit-debug`, `--skip-xmem` (to stop before CUDA-only XMem eval).
8) `xmem/xmem_yolo11/main_yolo.py`
   - Args: `--root`, `--clip`, `--device {cuda,cpu}`, `--width`, `--shot-detect`, `--auto-prompt`, `--yolo-model`, `--yolo-conf`, `--yolo-max-objects`, `--yolo-imgsz`, `--autoprompt-fallback {gt,skip}`, `--yolo-debug`, `--skip-xmem` (seed-only/debug), `--run-id`.

Quickstart commands (copy/paste, adjust clip ids)
-------------------------------------------------
First run (works on most machines)
```bash
# Baseline SAM‑2 with GT (CPU/Mac friendly)
python sam2/sam2_gt/main.py --data-root sam2/data --clips clip_corner --device cpu

# OWL‑ViT auto‑prompt seeding (GPU recommended; seeds work on CPU)
python sam2/sam2_owlvit/main.py --data-root sam2/data --clips clip_corner \
  --auto-prompt --prompts "perimeter billboard; stadium advertising board" \
  --owlvit-score-thr 0.08 --device cuda
```
- SAM-2 GT: `python sam2/sam2_gt/main.py --data-root sam2/data --clips clip_gentle --device cuda`
- SAM-2 GT + shot detection: `python sam2/sam2_gt_shotdetector/main.py --data-root sam2/data --clips clip_gentle --device cuda`
- SAM-2 OWL-ViT: `python sam2/sam2_owlvit/main.py --auto-prompt --prompts "perimeter billboard; sideline banner" --clips clip_occlusion --device cuda`
- SAM-2 Moondream: `python sam2/sam2_moondream/main.py --auto-prompt --prompts "perimeter billboard; sideline banner" --moondream-threshold 0.05 --clips clip_occlusion --device cuda`
- SAM-2 YOLO11: `python sam2/sam2_yolo11/main.py --yolo-model sam2/sam2_yolo11/weights/best.pt --source data/clips/clip_occlusion.mp4 --device cuda`
- XMem GT: `python xmem/xmem_gt/main_gt.py --root xmem --device cuda --clip clip_gentle`
- XMem OWL-ViT (seeds only on CPU/Mac): `python xmem/xmem_owlvit/main_owlvit.py --root xmem --device cuda --clip clip_corner --shot-detect --auto-prompt --prompts "perimeter billboard; sideline banner" --owlvit-debug --skip-xmem`
- XMem YOLO11 (seeds only on CPU/Mac): `python xmem/xmem_yolo11/main_yolo.py --root xmem --device cpu --width 1280 --shot-detect --auto-prompt --yolo-model xmem/xmem_yolo11/weights/best.pt --yolo-conf 0.05 --yolo-max-objects 2 --clip clip_glare --skip-xmem`
- Full XMem runs require CUDA; drop `--skip-xmem` and set `--device cuda` on a GPU host.

Troubleshooting (common in this repo)
-------------------------------------
- **“No clips were processed”** → check `--data-root/--root` and the `--clips/--clip` name exists under `data/clips/` (or `data/frames/` if running from frames).
- **CUDA available but script falls back to CPU** → verify in the same env: `python -c "import torch; print(torch.cuda.is_available())"`. If `False`, you’re not on the right conda/venv.
- **HuggingFace symlink warning on Windows** → set once and reopen shell: `setx HF_HUB_DISABLE_SYMLINKS_WARNING 1`.
- **OpenMP conflict during YOLO training** (`libomp` vs `libiomp5md`) → temporary workaround: `setx KMP_DUPLICATE_LIB_OK TRUE` then restart shell.
- **`cv2` import errors** → reinstall OpenCV in the active env: `pip install --upgrade opencv-python-headless`.

Tested / hardware
-----------------
- XMem eval: CUDA GPU required (fails fast if `torch.cuda.is_available()` is False).
- OWL-ViT / YOLO seeding: runs on CPU if CUDA is absent (but masks require CUDA).

Notes
-----
- Ensure video clips are present under `data/clips/` before running.
- Prompts: keep a reusable list in `prompts_list.txt` and pass via `--prompts-file` when supported.
- For seed-only debugging on non-CUDA hosts, use `--skip-xmem` in the XMem OWL-ViT/YOLO wrappers; inspect `re_prompts_*.csv` and `shot_XXX_seed.jpg`.
