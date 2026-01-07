[![Version](https://img.shields.io/github/v/release/bokertof/crystal_density_prediction)](https://github.com/bokertof/crystal_density_prediction/releases)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/github/license/bokertof/crystal_density_prediction)](LICENSE)

# Crystal Density Prediction Using a GRU-based Model
**Stanislav Melnikov**

**Abstract:**

Accurate prediction of crystal density is a key challenge in the design of energetic materials, as density critically influences the performance and safety of these materials. Here, we present a fast and accurate deep-learning approach for crystal density prediction based solely on molecular SMILES representations, avoiding explicit quantum-chemical calculations. A bidirectional gated recurrent unit (GRU) neural network was trained on large datasets derived from the Cambridge Structural Database (CCDC), comprising over 100,000 molecular crystals, as well as several benchmark datasets of energetic compounds serving as test datasets. To enhance model robustness and accuracy, randomized SMILES representations were employed together with test-time augmentation (TTA), where predictions are averaged over multiple randomized SMILES inputs of the same molecule.
We further estimate the intrinsic lower bound of achievable prediction error by analyzing density variations among experimentally observed polymorphs, showing that mean uncertainties of ~0.02–0.03 g cm⁻³ derived previously in literature might represent a fundamental limit for structure-based density prediction due to polymorphism. The proposed model achieves mean absolute errors as low as 0.015 g cm⁻³ on external test sets and ~0.03 g cm⁻³ on diverse CCDC datasets, approaching this estimated limit and outperforming or matching state-of-the-art QSAR and quantum-chemistry-based methods at a fraction of the computational cost. Cross-dataset validation demonstrates good generalization, while bootstrap aggregation provides no significant improvement over TTA alone.
Overall, this work establishes randomized-SMILES deep learning with test-time augmentation as a simple, computationally efficient, and highly accurate strategy for crystal density prediction, enabling large-scale virtual screening of energetic materials prior to synthesis.


The environment of the project can be readily built from the density.yml file:

```
conda env create -f density.yml
```

Then, you can go to the environment by command:

```
conda activate density
```
