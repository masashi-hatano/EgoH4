# The Invisible EgoHand: 3D Hand Forecasting through EgoBody Pose Estimation

<div align="center">

[![Paper](https://img.shields.io/badge/arXiv-2504.08654-red)](https://arxiv.org/abs/2504.08654)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://masashi-hatano.github.io/EgoH4/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

**This is the official code releasse for "The Invisible EgoHand: 3D Hand Forecasting through EgoBody Pose Estimation".**

## 🔨 Installation

```bash
git clone https://github.com/masashi-hatano/EgoH4.git
cd EgoH4
python3 -m venv egoh4
source egoh4/bin/activate
pip install -r requirements.txt
```

## 🔥 Training
Please run the following command to train the EgoH4 model. The checkpoint can be downloaded from [here](https://huggingface.co/masashi-hatano/EgoH4/tree/main).
```bash
python3 lit_main.py train=True test=False
```

## 🔍 Evaluation
To evaluate the model, please run the following command.
```bash
python3 lit_main.py train=False test=True devices=[0] strategy=auto
```

## ✍️ Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{Hatano2025EgoH4,
  title={The Invisible EgoHand: 3D Hand Forecasting through EgoBody Pose Estimation},
  author = {Hatano, Masashi and Zhu, Zhifan and Saito, Hideo and Damen, Dima},
  journal={arXiv preprint arXiv:2504.08654},
  year={2025}
}
```
