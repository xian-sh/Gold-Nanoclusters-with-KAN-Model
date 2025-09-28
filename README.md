# Prediction of Frontier Orbital Energy of Atomically Precise Gold Nanoclusters with KAN Model

Tingting Jiang, Qiqi Zhang, Zhan Si, Jingjing Hu, Ying Lv, Tingting Wang, Haizhu Yu*

-------------

## Workflow

A machine learning model using interpretable automated feature engineering (Kolmogorov-Arnold Networks) identifies charge number and Au-Au coordination as key structural descriptors, enabling accurate prediction of gold nanoclusters' electronic properties (HOMO, LUMO, gap, oxidation potential) with low error, offering a cost-efficient strategy for catalytic design.

![a34d044ee64218e5ee877520c9912196_720](https://github.com/user-attachments/assets/eee7ba84-50a1-4090-89ec-d31d2aabacb2)

-----------------

## Installation

```
  # Core scientific stack & visualization
  pip install numpy pandas statsmodels scikit-learn matplotlib seaborn

  # PyTorch (CPU build shown; for GPU see https://pytorch.org/get-started/locally/)
  pip install torch

  # Symbolic math
  pip install sympy

  # KAN network (pykan)
  pip install git+https://github.com/KindXiaoming/pykan.git
```

## Acknowledgements

We would like to extend our sincere thanks to the authors of the **KAN network** for their pioneering work and for making their code publicly available. This project benefits significantly from their contributions.  🔗 [Kaggle](https://www.kaggle.com/code/seyidcemkarakas/kan-regression-graduate-admissions/notebook)
