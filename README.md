# P114-Annotation-Free-Segmentation

This project explores annotation-free tumor detection by framing tumors as anomalies in medical images. It combines robust subspace methods and unsupervised deep learning to separate normal tissue from abnormal regions automatically.

The repository contains several model families for anomaly detection via robust reconstruction-based segmentation. Most of them are implementations of published methods, some with sight adaptations.

## Implemented Models

| Model | Type | Summary |
| ---- | --- | --- |
| RPCA | Classical low-rank + sparse decomposition | Performs robust principal component analysis with an augmented Lagrange multiplier solver. The low-rank term is treated as background and the sparse term as anomaly signal. |
| RDA | Deep autoencoder | Uses a convolutional encoder, a tied-weight linear bottleneck, and a pixel-shuffle decoder to reconstruct normal structure while exposing anomalous residuals. |
| ceVAE | Context-encoding variational autoencoder | A fully convolutional VAE with coordinate convolutions and attribution-based anomaly generation. Training is supported via both standard stochastic methods and context-encoding techniques. |
| RVAE | Robust variational autoencoder | A convolutional VAE that reconstructs images from a compact latent space  using Beta Divergence. |
| RDDPM | Denoising diffusion model | A lightweight diffusion-based model that corrupts and iteratively denoises inputs to produce reconstructions and anomaly maps. |
| **R&#8209;ceVAE** | Context-encoding variational autoencoder | A novel combination of the ceVAE base architecture and the RDA's ADMM trainng loop. |

## Attribution And Credits

The implementations below are based on prior work.

| Model | Credit  |
| :--- | :--- |
| RPCA | Candès, E. J., Li, X., Ma, Y., & Wright, J. ["Robust Principal Component Analysis?"](https://people.eecs.berkeley.edu/~yima/psfile/JACM11.pdf), Journal of the ACM (JACM), 2011 |
| RDA | Zhou, C., & Paffenroth, R. C. ["Anomaly Detection with Robust Deep Autoencoders"](https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p665.pdf), ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2017 |
| ceVAE | Zimmerer, D., Kohl, S. A., Petersen, J., Isensee, F., & Maier-Hein, K. H. ["Context-encoding Variational Autoencoder for Unsupervised Anomaly Detection"](https://arxiv.org/abs/1812.05941), Medical Image Computing and Computer Assisted Intervention (MICCAI) Workshop / arXiv, 2018 |
| RVAE | Akrami, H., Joshi, A. A., Li, J., Aydore, S., & Leahy, R. M. ["A Robust Variational Autoencoder Using Beta Divergence"](https://pmc.ncbi.nlm.nih.gov/articles/PMC9881733/), Knowledge-Based Systems / PMC Manuscript, 2019/2022 |
| RDDPM | Moradi, M., & Paynabar, K. ["RDDPM: Robust Denoising Diffusion Probabilistic Model for Unsupervised Anomaly Segmentation"](https://arxiv.org/abs/2508.02903), ICCV Workshop (VISION) / arXiv, 2025 |

## Project Structure

- `models/` contains the model definitions and the shared model factory.
- `trainers/` contains the training loops for the supported model families.
- `evaluation/` contains metrics and visualization helpers.
- `data/` contains dataset wrappers and augmentation utilities.
<!-- - `gen_res.py` is the main inference / evaluation entry point for running experiments across datasets and augmentation strengths. -->

## Quick Start

The code is organized around two helper factories:

```python
from models import get_model
from trainers import get_trainer

model = get_model('ceVAE', latent_channels=1024, with_r=True)
trainer = get_trainer('ceVAE', model=model, loader=data_loader)
```

<!-- To run the end-to-end evaluation pipeline, use `gen_res.py` from the project root:

```bash
python gen_res.py --model ceVAE --param dense_noise --dataset CDNet
``` -->

<!-- Supported model choices in the current pipeline are `ceVAE`, `RPCA`, `RVAE`, `RDA`, `RDDPM`, and `Opus`. -->

## Notes

- Inputs are typically resized to `128 x 128` and converted to single-channel images.
<!-- - Several notebooks in the repository show training and evaluation workflows for individual datasets and models.
- Result files and saved checkpoints are written under `results/` and `saved_models/` respectively. -->
