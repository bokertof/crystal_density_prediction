[![Version](https://img.shields.io/github/v/release/bokertof/crystal_density_prediction?color=orange)](https://github.com/bokertof/crystal_density_prediction/releases)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/github/license/bokertof/crystal_density_prediction?color=green)](LICENSE)

# Crystal Density Prediction Using a RNN-based Model
**Stanislav Melnikov**

**Abstract:**

Accurate prediction of crystal density is a key challenge in the design of energetic materials, as density critically influences the performance and safety of these materials. Here, we present a fast and accurate deep-learning approach for crystal density prediction based solely on molecular SMILES representations, avoiding explicit quantum-chemical calculations. A bidirectional gated recurrent unit (GRU) neural network was trained on large datasets derived from the Cambridge Structural Database (CCDC), comprising over 100,000 molecular crystals, as well as several benchmark datasets of energetic compounds serving as test datasets. To enhance model robustness and accuracy, randomized SMILES representations were employed together with test-time augmentation (TTA), where predictions are averaged over multiple randomized SMILES inputs of the same molecule.
We further estimate the intrinsic lower bound of achievable prediction error by analyzing density variations among experimentally observed polymorphs, showing that mean uncertainties of ~0.02–0.03 g cm⁻³ derived previously in literature might represent a fundamental limit for structure-based density prediction due to polymorphism. The proposed model achieves mean absolute errors as low as 0.015 g cm⁻³ on external test sets and ~0.03 g cm⁻³ on diverse CCDC datasets, approaching this estimated limit and outperforming or matching state-of-the-art QSAR and quantum-chemistry-based methods at a fraction of the computational cost. Cross-dataset validation demonstrates good generalization, while bootstrap aggregation provides no significant improvement over TTA alone.
Overall, this work establishes randomized-SMILES deep learning with test-time augmentation as a simple, computationally efficient, and highly accurate strategy for crystal density prediction, enabling large-scale virtual screening of energetic materials prior to synthesis.

# The Problem of Density Prediction

# Dataset
We used several datasets in this work. The largest, CCDC, was manually curated from the Cambridge Structural Database: structures had at least one carbon atom, no errors or disorders, R-factor < 5%, and were collected under ambient conditions. Rare atoms (occurring < 200 times) were removed, and molecules were converted to SMILES using RDKit. Duplicates (structures with <1.5% difference in reduced unit parameters) were removed, while polymorphs (same SMILES, different crystal packing) were retained randomly to avoid density biases. The resulting CCDC dataset contains 100,717 molecules (train:test ≈ 90:10). A CHNO-only subset, CCDC CHNO, contains 36,895 molecules (train:test ≈ 90:10).

The Casey dataset (26,265 CHNO molecules) was extracted from GDB with an Oxygen Balance criterion for energetic candidates (train:test = 21,012:5,253).

For additional testing, we used three smaller literature datasets: Huang & Massa (109 molecules), Rice (34 high-nitrogen molecules), and Mathieu (307 molecules).

| Dataset | CCDC | CCDC CHNO | Casey | Rice + Huang&Massa + Mathieu |
|:--------|------:|----------:|------:|----------------------------:|
| Number of samples | 100,717 | 36,895 | 26,265 | 450 |
| Size of vocabulary | 111 | 33 | 22 | –** |
| Min SMILES length* | 3 | 5 | 15 | 8 |
| Max SMILES length* | 178 | 150 | 50 | 207 |
| Mean SMILES length* | 66 | 56 | 30 | 75 |
| Median SMILES length* | 59 | 50 | 29 | 69 |
| Min density (g/cm³) | 0.412 | 0.412 | 1.377 | 1.288 |
| Max density (g/cm³) | 8.432 | 2.694 | 1.916 | 2.184 |
| Mean density (g/cm³) | 1.4918 | 1.3087 | 1.6399 | 1.7593 |
| Median density (g/cm³) | 1.403 | 1.291 | 1.638 | 1.762 |


**Summary of the datasets' statistics.**

<img width="440" height="439" alt="image" src="https://github.com/user-attachments/assets/7144194a-1ba4-4a91-a8b8-7e477c9fc1b0" />

**Paired density vs SMILES length distributions for CCDC dataset.**

# Estimation of the Lowest Possible Error
It is well known that a single molecule can crystallize in multiple polymorphic forms with different packing arrangements and symmetries, which generally leads to different crystal densities. As a result, crystal density cannot be predicted with arbitrarily high precision. To assess the intrinsic lower bound of prediction accuracy, we analyzed density variations among polymorphs extracted from the Cambridge Structural Database (CCDC).
By identifying structures with identical canonical SMILES but reduced lattice parameters differing by more than 1.5%, we estimated the irreducible uncertainty arising from polymorphism. The resulting mean and median density differences (~0.02–0.03 g cm⁻³) are comparable to the “excellent” accuracy threshold proposed by Kim et al. and to the best errors reported by state-of-the-art methods. This suggests that mean absolute errors in this range are close to the fundamental limit for crystal density prediction.
| Calculated parameter | All atoms | Only CHNO |
|:--------------------|----------:|----------:|
| Number of polymorph groups | 1607 | 807 |
| Max (max–min) difference (g/cm³) | 0.1920 | 0.1370 |
| Mean (max–min) difference (g/cm³) | 0.0302 | 0.0235 |
| Mean of all pairwise differences (g/cm³) | 0.0317 | 0.0277 |
| Median of all pairwise differences (g/cm³) | 0.0240 | 0.0230 |
| Mean standard deviation (g/cm³) | 0.0216 | 0.0200 |

**Statistics of crystal density variations among polymorphs in the CCDC dataset.**
Values were calculated within each polymorph group and then averaged over all groups. Erroneous polymorph assignments with unusually large differences were manually excluded.


The largest observed density differences among polymorphs reach 0.19 g cm⁻³ for all elements and 0.14 g cm⁻³ for CHNO-only compounds, highlighting the importance of polymorphism-aware error interpretation. Notably, errors in predicted density may reflect the existence of unobserved polymorphs rather than deficiencies of the model itself. All analyses were performed on chemically diverse and noisy datasets, and the derived limits may not directly extrapolate to more homogeneous systems.


The environment of the project can be readily built from the density.yml file:

```
conda env create -f density.yml
```

Then, you can go to the environment by command:

```
conda activate density
```
