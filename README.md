# Open-Vocabulary Gaze Object Prediction

Official repository for **[Open-Vocabulary Gaze Object Prediction: Benchmark and Method](https://arxiv.org/abs/2607.18827)**, accepted by **ACM Multimedia 2026**.

We introduce **DiSG (Diverse Scenes for Gaze Object Prediction)**, a real-image benchmark for open-vocabulary gaze object prediction, with diverse human-centric scenes, object categories, and fine-grained body-part categories. Together with an novel **OVGOP framework** for localizing and recognizing gaze targets under a free-form category vocabulary. This project provides:

- Data of DiSG.
- Code and model weights of our OVGOP framework.
- Model weights of various previous GOP methods for DiSG.

<p align="center">
  <a href="https://arxiv.org/abs/2607.18827"> Paper</a> ·
  <a href="#disg-dataset"> DiSG Dataset</a> ·
  <a href="#ovgop-framework"> OVGOP Framework</a> ·
  <a href="#citation"> Citation</a>
</p>

<p align="center">
  <img src="assets/IntroductionFigure.png" alt="Comparison between conventional gaze object prediction and open-vocabulary gaze object prediction" width="100%">
</p>

<p align="center"><em>Conventional GOP operates on fixed categories, while OVGOP supports gaze-object prediction with unseen category prompts.</em></p>

## News and Todo List

- [x] **2026-07-10** — Our paper was accepted by ACM Multimedia 2026.
- [x] **2026-07-16** — The DiSG dataset was made available.
- [x] **2026-10-08** - Code of our OVGOP framework is available in `ovgop/`.
---

## DiSG Dataset
<p align="center">
  <img src="assets/VisuDataset.png" alt="Examples of gaze-object annotations in the DiSG benchmark" width="100%">
</p>

<p align="center"><em>DiSG gaze-object annotations across diverse scenes, common objects, and fine-grained body parts.</em></p>

### Download

The DiSG package is available from three download mirrors:

- 🌐 **Google Drive:** [Download DiSG](https://drive.google.com/file/d/1oH4tr66ZQLQvWkJnlaCosyetDs1zOxma/view?usp=sharing)
- ☁️ **Baidu Netdisk:** [Download DiSG](https://pan.baidu.com/s/1aD86VRsDN9ki22WyIT8i5Q?pwd=disg) — access code: `disg`
- 🤗 **Hugging Face:** [Download DiSG](https://huggingface.co/datasets/sensniu/DiSG)

### Dataset Statistics
The released validation annotation contains **2,781 images**, **3,700 gaze-object annotations**, and all **86 categories**.

| Statistic | Value |
| --- | ---: |
| Images | 13,041 |
| Gaze-object annotations | 17,447 |
| Categories | 86 |
| Base categories | 60 |
| Novel categories | 26 |

DiSG defines a hybrid **Base/Novel** split over all 86 categories:

| Category source | Base (`seen`) | Novel (`unseen`) | Total |
| --- | ---: | ---: | ---: |
| COCO object categories | 57 | 23 | 80 |
| Human body-part categories | 3 | 3 | 6 |
| **Total** | **60** | **26** | **86** |

The six body-part categories are split as follows:

- Base: `head`, `hand`, `leg`
- Novel: `foot`, `arm`, `torso`

### Data Preparation

1. Download the DiSG package from the link above.
2. Extract the package into a local data directory.
3. Keep the provided training and validation annotation files unchanged when reproducing the official benchmark split.
4. Use the `ov_setting` field in `categories` to distinguish Base (`seen`) and Novel (`unseen`) categories.

### Annotation Format

DiSG uses a COCO-compatible JSON structure, and the training and validation annotations share the same schema:

- `images`: image IDs, file names, and dimensions
- `annotations`: standard COCO object-detection and segmentation annotations
- `annotations_gaze`: gaze-object instances, including the gaze point, target box, head box, person box, target category, and source metadata
- `categories`: category names, and `ov_setting` (`seen` for Base and `unseen` for Novel)

Bounding boxes use the COCO convention `[x, y, width, height]`, and all coordinates are expressed in image pixels.

---

## OVGOP Framework

### Environment Setup

The model requires an NVIDIA GPU, Python 3.7.11, PyTorch 1.10.0, torchvision 0.11.1, and CUDA Toolkit 11.3. Run all installation, compilation, training, and evaluation commands from the `ovgop/` subdirectory.

Starting from the repository root:

```bash
cd ovgop

conda create -n ovgop python=3.7.11 -y
conda activate ovgop
conda install pytorch=1.10.0 torchvision=0.11.1 cudatoolkit=11.3 -c pytorch -c conda-forge
python -m pip install -r ../requirements.txt
```

Compile the deformable-attention CUDA extension. A CUDA Toolkit installation containing `nvcc` is required; set `CUDA_HOME` to its actual location:

```bash
export CUDA_HOME=/path/to/cuda-11.3
python setup_groundingdino_ext.py build_ext --inplace
python -c "from models.dino import _C; print('extension OK')"
```

### Data Preparation

Download DiSG from the [dataset download links above](#download) and extract it with the following layout:

```text
/path/to/DiSG/
├── train/
├── val/
└── annotations/
    ├── ovgop_train.json
    └── ovgop_val.json
```

> Replace `/path/to/DiSG` in the commands below with the absolute path to your dataset.

### Model Weights

Download the OVGOP checkpoint from [Google Drive](https://drive.google.com/file/d/1iGAaGDinqIQSXhXdhskqU14RCJPGyYh4/view?usp=drive_link).

Create a checkpoint directory from within `ovgop/`:

```bash
mkdir -p checkpoints
```

Save the downloaded checkpoint as `checkpoints/ovgop_disg.pth` (that is, `ovgop/checkpoints/ovgop_disg.pth` relative to the repository root).

### Evaluation on DiSG

Evaluate the released OVGOP checkpoint on the complete DiSG validation set, including Base and Novel categories.

```bash
python main.py -c configs/disg_ovgop.py \
  --eval \
  --amp \
  --resume checkpoints/ovgop_disg.pth \
  --data-root /path/to/DiSG \
  --output-dir outputs/disg_eval
```

Evaluation results are written to `outputs/disg_eval/` inside `ovgop/`.

### Training

To train OVGOP on DiSG:

```bash
bash start_train.sh \
  --amp \
  --data-root /path/to/DiSG \
  --output-dir outputs/disg_ovgop
```

## Citation

```
@inproceedings{wang2026open,
  author = {Binglu Wang and Sensen Niu and Ying Chen and Guangyu Guo},
  title = {Open-Vocabulary Gaze Object Prediction: Benchmark and Method},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year = {2026}
}

```

## Awesome-list of GOP
- [Awesome list of Human-Centered Relationship Understanding](https://github.com/yangyang9912/Awesome-Human-Centered-Relationship), including [Methods](https://github.com/yangyang9912/Awesome-Human-Centered-Relationship#c-gaze-object-prediction-gop) and [Benchmarks](https://github.com/yangyang9912/Awesome-Human-Centered-Relationship#gaze-object-prediction) of gaze object prediction.
