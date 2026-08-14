# DECO: Decoupled Multimodal Diffusion Transformer for Bimanual Dexterous Manipulation with a Plugin Tactile Adapter
<p>
  <a href="https://arxiv.org/pdf/2602.05513">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv">
  </a>
  <a href="https://baai-humanoid.github.io/DECO-webpage/">
    <img src="https://img.shields.io/badge/Project-Website-blue">
  </a>
  <a href="https://huggingface.co/datasets/BAAI-Humanoid/DECO-50">
    <img src="https://img.shields.io/badge/HuggingFace-Data-yellow?logo=huggingface">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache--2.0-green">
  </a>
</p>

# Quick start Example 
## 1. Installation
The code is tested on python3.10 and pytorch2.6, cuda12.4.
```
pip install -r requirements.txt
```

## 2. Hyperparameters
- Configuration files for DECO are provided in the [`config/`](https://github.com/1343744768/IL_training_codebase/tree/master/config) directory.

```yaml
model:
  action_dim: 7               # DOF of action
  chunk_size: 16              # chunk size
  n_view: 2                   # Number of camera views
  obs_state: True             # Whether to use proprioceptive states
  obs_dim: 8                  # DOF of proprioception states
  use_task_condition: True    # Whether to enable one-hot task embedding
  num_tasks: 40               # Number of tasks for one-hot embedding
  use_tactile: False          # Whether to use tactile observations
  plugin: False               # Enable plugin tactile adapter for vision-based pretrained policy
  num_attn_blocks: 6          # Number of transformer attention blocks
  inf_step: 5                 # Number of flow-matching inference steps
  heads: 8                    # Number of attention heads
  dim: 512                    # Model hidden dimension
  rope_axes_dim: [256, 256]   # Rotary positional embedding dimensions for [height, width], same to img_size
  pretrain_model_path: False  # Path to vision-based pretrained policy checkpoint (for deployment)
  adapter_model_path: False   # Path to plugin tactile adapter (for deployment)

data:
  chunk_size: 16              # DOF of action
  n_view: 2                   # Number of camera views
  tac_left_max: 3486.0        # Maximum value of left tactile sensor
  tac_right_max: 4050.0       # Maximum value of right tactile sensor
  norm_type: 'mean_std'       # Normalization type: mean_std or min_max
  observation_mean: [...]     # Observation mean (computed via utils/cal_mean_std.py)
  observation_std: [...]      # Observation standard deviation
  observation_min: [...]      # Observation minimum 
  observation_max: [...]      # Observation maximum
  action_mean: [...]          # Action mean
  action_std: [...]           # Action standard deviation
  action_min: [...]           # Action minimum
  action_max: [...]           # Action maximum

img:
  img_size: [256, 256]        # [height, width]. Resize input images to this resolution
  img_mean: [0.485, 0.456, 0.406]  # ImageNet mean
  img_std: [0.229, 0.224, 0.225]   # ImageNet standard deviation
```

### LIBERO Config Notes
- Two pretrained model sizes are provided: `config/deco_libero_80m.yaml` (80M params) and `config/deco_libero_0.5b.yaml` (0.5B params).

## 3. Dataset Preparation

### Prepare LIBERO Dataset

- Download the LIBERO dataset from HuggingFace:
```bash
hf download HuggingFaceVLA/libero --repo-type dataset --local-dir ./
```

- Convert the LIBERO data format:
```bash
cd utils
python convert_libero.py --data_path /path/to/libero/data/chunk-000 --save_path /path/to/output
python task2json.py --input /path/to/libero/meta/tasks.parquet --output /path/to/tasks.json
```

- The processed data structure:
```
datasets/
├── episode_0000/
│   ├── img1/              # External camera
│   │   └── frame_*.png
│   ├── img2/              # Wrist camera
│   │   └── frame_*.png
│   └── episode_info.pkl   # Contains state, action, task index
└── tasks.json              # prompt → index mapping
```

## 4. Training

- **Multi-GPU training**
```bash
torchrun --nproc_per_node 8 \
  train.py \
  --distributed True \
  --amp True \
  --device_id '0,1,2,3,4,5,6,7' \
  --config ./config/deco_libero_80m.yaml \
  --data /path/to/libero/data \
  --lr 1e-4 \
  --lr_f 1e-6 \
  --batch_size 1024 \
  --num-workers 16 \
  --epochs 300 \
  --logs ./logs/libero_deco/ \
  --save_period 50
```
- **Single-GPU training**
```bash
python train.py \
  --distributed False \
  --amp False \
  --device_id '0' \
  --config ./config/deco_libero_80m.yaml \
  --data /path/to/libero/data \
  --lr 1e-4 \
  --lr_f 1e-6 \
  --batch_size 128 \
  --num-workers 16 \
  --epochs 300 \
  --logs ./logs/libero_deco/ \
  --save_period 50
```

## 5. Evaluation

### Eval LIBERO

- Install LIBERO:
```bash
pip install -r requirements_libero.txt # NOTE: mujoco version must be 3.3.2
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e .
```

- Patch LIBERO for high PyTorch versions (`LIBERO/libero/libero/benchmark/__init__.py` line 164):
```python
init_states = torch.load(init_states_path, weights_only=False)
```

- Configure `eval_libero.py`: set `yaml_path`, `task_suite_name`, `task_config` and the checkpoint path in [`yaml`](https://github.com/1343744768/IL_training_codebase/blob/main/config/deco_libero_0.5b.yaml#L16).
- Run evaluation:
```bash
export MUJOCO_GL=egl
python eval_libero.py
```

- LIBERO Results: Success rates (50 trials per task):

| Model | Spatial | Object | Goal | Long | **Average** |
|:-----:|:--------------:|:-------------:|:-----------:|:---------:|:-----------:|
| DECO-80M | 94.2% | 99.4% | 96.8% | 86.4% | **94.2%** |
| DECO-0.5B | 96.4% | 98.6% | 96.6% | 89.8% | **95.4%** |

# Citation
If you find this project useful for your research, please consider citing our paper:

```bibtex
@article{li2026deco,
  title={DECO: Decoupled Multimodal Diffusion Transformer for Bimanual Dexterous Manipulation with a Plugin Tactile Adapter},
  author={Xukun Li and Yu Sun and Lei Zhang and Bosheng Huang and Yibo Peng and Yuan Meng and Haojun Jiang and Shaoxuan Xie and Guacai Yao and Alois Knoll and Zhenshan Bing and Xinlong Wang and Zhenguo Sun},
  journal={arXiv preprint arXiv:2602.05513},
  year={2026}
}
```
# License
This project is released under the Apache 2.0 license.

## Contact
We are hiring!!! Full-time researchers, engineers, interns and PhD students are all open for recruitment. If you are interested in working with us on whole-body mobile manipulation for humanoid robots, please contact hitsunzhenguo@gmail.com.
