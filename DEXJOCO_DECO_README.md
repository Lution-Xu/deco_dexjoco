# DexJoCo + DECO Training and Evaluation

This is a separate DexJoCo path. It does not modify the original `train.py`,
`dataset.py`, `libero_dataset.py`, `eval_libero.py`, or `models/deco/deco.py`.

## Environment

Install the original DECO requirements, then add an mp4 backend:

```bash
pip install -r requirements.txt
pip install "imageio[ffmpeg]"
```

Install DexJoCo in its own environment as described by `/home/xz/deco/dexjoco-main/README.md`
for simulation evaluation.

## Stats

Stats JSON files are already generated under `config/`:

```text
config/dexjoco_stats_single_rand_obj.json
config/dexjoco_stats_single_rand_full.json
config/dexjoco_stats_dual_rand_obj.json
config/dexjoco_stats_dual_rand_full.json
```

Regenerate one if the dataset changes:

```bash
python dexjoco_stats.py \
  --task-group single \
  --regime rand_obj \
  --output config/dexjoco_stats_single_rand_obj.json
```

## Training

Single-arm rand_obj:

```bash
python train_dexjoco.py \
  --config config/deco_dexjoco_single_rand_obj.yaml \
  --device_id 0 \
  --batch-size 64 \
  --num-workers 8 \
  --epochs 200 \
  --amp \
  --logs logs/deco_dexjoco_single_rand_obj
```

Dual-arm rand_full:

```bash
python train_dexjoco.py \
  --config config/deco_dexjoco_dual_rand_full.yaml \
  --device_id 0 \
  --batch-size 32 \
  --num-workers 8 \
  --epochs 200 \
  --amp \
  --logs logs/deco_dexjoco_dual_rand_full
```

Run four configs for the full plan:

```text
config/deco_dexjoco_single_rand_obj.yaml
config/deco_dexjoco_single_rand_full.yaml
config/deco_dexjoco_dual_rand_obj.yaml
config/deco_dexjoco_dual_rand_full.yaml
```

## Evaluation

Evaluate all single-arm rand_obj tasks:

```bash
python eval_dexjoco.py \
  --config config/deco_dexjoco_single_rand_obj.yaml \
  --checkpoint logs/deco_dexjoco_single_rand_obj/best.pth \
  --tasks all \
  --episodes 50 \
  --save-video
```

Evaluate one dual-arm smoke task:

```bash
python eval_dexjoco.py \
  --config config/deco_dexjoco_dual_rand_obj.yaml \
  --checkpoint logs/deco_dexjoco_dual_rand_obj/best.pth \
  --tasks bimanual_hanoi \
  --episodes 1 \
  --save-video
```

Outputs are written to `outputs/dexjoco_deco_eval/<regime>/<task>/` with per-task
success marker files and a JSON summary.
