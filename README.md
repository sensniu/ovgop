# Open-Vocabulary Gaze Object Prediction

Official repository for **[Open-Vocabulary Gaze Object Prediction: Benchmark and Method](https://arxiv.org/abs/2607.18827)**, accepted by **ACM Multimedia 2026**.

We introduce **DiSG (Diverse Scenes for Gaze Object Prediction)**, a real-image benchmark for open-vocabulary gaze object prediction, with diverse human-centric scenes, object categories, and fine-grained body-part categories. Together with an novel **OVGOP framework** for localizing and recognizing gaze targets under a free-form category vocabulary. This project provides:

- Data of DiSG.
- Code and model weights of our OVGOP framework.
- Model weights of various previous GOP methods for DiSG.

<p align="center">
  <a href="https://arxiv.org/abs/2607.18827"> Paper</a> ·
  <a href="#disg-dataset"> DiSG Dataset</a> ·
  <a href="#code-and-model-weights"> Code and Model Weights</a> ·
  <a href="#citation"> Citation</a> ·
</p>

<p align="center">
  <img src="assets/IntroductionFigure.png" alt="Comparison between conventional gaze object prediction and open-vocabulary gaze object prediction" width="100%">
</p>

<p align="center"><em>Conventional GOP operates on fixed categories, while OVGOP supports gaze-object prediction with unseen category prompts.</em></p>

## News and Todo List

- [x] **2026-07-10** — Our paper was accepted by ACM Multimedia 2026.
- [x] **2026-07-16** — The DiSG dataset was made available.
- [ ] Uploading code and weights of compared methods for DiSG
- [ ] Uploading code of our OVGOP framework
---

## DiSG Dataset
<p align="center">
  <img src="assets/VisuDataset.png" alt="Examples of gaze-object annotations in the DiSG benchmark" width="100%">
</p>

<p align="center"><em>DiSG gaze-object annotations across diverse scenes, common objects, and fine-grained body parts.</em></p>

### Download

The DiSG package is available from two download mirrors:

- 🌐 **Google Drive:** [Download DiSG](https://drive.google.com/file/d/1oH4tr66ZQLQvWkJnlaCosyetDs1zOxma/view?usp=sharing)
- ☁️ **Baidu Netdisk:** [Download DiSG](https://pan.baidu.com/s/1aD86VRsDN9ki22WyIT8i5Q?pwd=disg) — access code: `disg`

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

## Code and Model Weights

The following components will be released as soon as possible:

- Code and model weights of our OVGOP 
- Model weights of previous GOP methods

## Citation

```
@inproceedings{wang2026open,
  author = {Binglu Wang and Sensen Niu and Ying Chen and Guangyu Guo},
  title = {Open-Vocabulary Gaze Object Prediction: Benchmark and Method},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year = {2026}
}

```
