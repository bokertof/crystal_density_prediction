[![Version](https://img.shields.io/github/v/release/bokertof/crystal_density_prediction?color=orange)](https://github.com/bokertof/crystal_density_prediction/releases)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/github/license/bokertof/crystal_density_prediction?color=green)](LICENSE)

# Crystal Density Prediction Using a RNN-based Model
*Stanislav Melnikov*


> 🔬 **Web demo available**  
> Try the crystal density predictor online — no code or installation needed:  
> 👉 https://crystaldensityprediction.streamlit.app


**Crystal density prediction:**

Accurate prediction of crystal density is a key challenge in the design of energetic materials, as density critically influences the performance and safety of these materials. Here, we present a fast and accurate deep-learning approach for crystal density prediction based solely on molecular SMILES representations, avoiding explicit quantum-chemical calculations. A bidirectional gated recurrent unit (GRU) neural network was trained on large datasets derived from the Cambridge Structural Database (CCDC), comprising over 100,000 molecular crystals, as well as several benchmark datasets of energetic compounds serving as test datasets. To enhance model robustness and accuracy, randomized SMILES representations were employed together with test-time augmentation (TTA), where predictions are averaged over multiple randomized SMILES inputs of the same molecule.
We further estimate the intrinsic lower bound of achievable prediction error by analyzing density variations among experimentally observed polymorphs, showing that mean uncertainties of ~0.02–0.03 g cm⁻³ derived previously in literature might represent a fundamental limit for structure-based density prediction due to polymorphism. The proposed model achieves mean absolute errors as low as 0.015 g cm⁻³ on external test sets and ~0.03 g cm⁻³ on diverse CCDC datasets, approaching this estimated limit and outperforming or matching state-of-the-art QSAR and quantum-chemistry-based methods at a fraction of the computational cost. Cross-dataset validation demonstrates good generalization, while bootstrap aggregation provides no significant improvement over TTA alone.
Overall, this work establishes randomized-SMILES deep learning with test-time augmentation as a simple, computationally efficient, and highly accurate strategy for crystal density prediction, enabling large-scale virtual screening of energetic materials prior to synthesis.

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


*Summary of the datasets' statistics.*


 <img width="693" height="458" alt="image" src="https://github.com/user-attachments/assets/63a2b2c5-802b-43d1-bf3e-f4a47a7ab529" />

*Violin plots of paired Tanimoto similarity distributions among 2,000 random picked samples over 4 selected datasets. For Rice, Huang&Massa and Mathieu datasets all molecules were used for calculations.*

# Model
Molecules were converted into canonical and randomized SMILES using RDKit, without standardization, to assess model performance under realistic conditions. Up to 30 randomized SMILES per molecule were generated. Special tokens <sos>, and <eos> were added, and SMILES were one-hot encoded into sparse matrices (size: 111 for CCDC, 33 for CCDC CHNO, 22 for Casey) using a pack_sequence approach to save memory.

Matrices were fed into a 4-layer bidirectional GRU (hidden size = 128), with the final hidden states concatenated and passed through a feedforward layer to predict crystal density. Training used 256-sized mini-batches for 500 epochs (250 for canonical SMILES) with Adam optimizer (lr = 0.003, weight decay = 0) and gradient clipping (norm = 50). The learning rate decayed by 0.9 every 10 (5 for canonical) epochs.

During inference, test-time augmentation (TTA) averaged predictions over 30 randomized SMILES per molecule. Bootstrap experiments were performed by resampling datasets of equal size for ensemble evaluation.

<img width="2002" height="1350" alt="Picture5" src="https://github.com/user-attachments/assets/e8655a08-d772-4aea-b7f2-5b25dd950eb0" />

*The model architecture. Molecules in RandomSMILES representations are fed by neural network. It includes 4-layer bidirectional GRU with attached feed-forward layer. The size of one-hot vectors depends on the dataset used for training. In the case of CCDC all size = 111, size = 33 for CCDC CHNO and size = 22 for Casey*

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

*Statistics of crystal density variations among polymorphs in the CCDC dataset.*
Values were calculated within each polymorph group and then averaged over all groups. Erroneous polymorph assignments with unusually large differences were manually excluded.


The largest observed density differences among polymorphs reach 0.19 g cm⁻³ for all elements and 0.14 g cm⁻³ for CHNO-only compounds, highlighting the importance of polymorphism-aware error interpretation. Notably, errors in predicted density may reflect the existence of unobserved polymorphs rather than deficiencies of the model itself. All analyses were performed on chemically diverse and noisy datasets, and the derived limits may not directly extrapolate to more homogeneous systems.

# Results
To assess the benefit of randomized SMILES, we first trained our GRU-based model on canonical SMILES from the CCDC dataset. Even with canonical representations, the model achieved strong performance on crystal density prediction for the test set. For this experiment, we adjusted the training hyperparameters, as the default settings (n_epochs = 500, step_size = 10) led to unstable training and poor convergence. Reducing them to n_epochs = 250 and step_size = 5 stabilized learning and prevented gradient issues. The training curve (see the report file) shows pronounced overfitting, consistent with previously reported observations for models trained on canonical SMILES.
Errors below 0.03 g cm⁻³ are considered “excellent” according to Kim et. al. Using this criterion, our model achieves excellent accuracy for 74% of the training set and 61% of the val CCDC set.

<img width="757" height="485" alt="image" src="https://github.com/user-attachments/assets/14b548fc-4310-42ea-b545-2b8c250aebd4" />

*The average Mean Absolute Error vs number of test-time augmentations. Calculations were performed 5 times on CCDC test subset.*

| Metric           | Casey (val) | Casey (train) | CCDC CHNO (val) | CCDC CHNO (train) | CCDC (val)  | CCDC (train) |
| ---------------- | ----------- | ------------- | --------------- | ----------------- | ----------- | ------------ |
| **Dataset size** | 5,253       | 21,012        | 3,895           | 33,000            | 10,717      | 90,000       |
| **R²**           | 0.8557 (33) | 0.9732 (21)   | 0.9272 (34)     | 0.9235 (19)       | 0.9697 (32) | 0.9866 (16)  |
| **Pearson r**    | 0.9261 (3)  | 0.9865 (0)    | 0.9632 (1)      | 0.9614 (1)        | 0.9849 (1)  | 0.9933 (0)   |
| **Max |Δρ|**     | 0.1435 (51) | 0.1471 (27)   | 0.1956 (33)     | 1.3381 (24)       | 2.1324 (27) | 2.9750 (69)  |
| **MAE**          | 0.0153 (1)  | 0.0069 (0)    | 0.0249 (1)      | 0.0231 (0)        | 0.0295 (1)  | 0.0221 (1)   |
| **RMSE**         | 0.0126 (1)  | 0.0068 (0)    | 0.0251 (1)      | 0.0302 (0)        | 0.0299 (2)  | 0.0283 (1)   |
| **Error < 0.01** | 44%         | 80%           | 27%             | 30%               | 27%         | 34%          |
| **Error < 0.02** | 74%         | 94%           | 51%             | 54%               | 50%         | 59%          |
| **Error < 0.03** | 88%         | 98%           | 68%             | 73%               | 67%         | 76%          |
| **Error > 0.1**  | 0.2%        | 0.05%         | 0.8%            | 0.7%              | 3%          | 1%           |

*Performance on Training and Validation Sets (Randomized SMILES). Performance on Training and Validation Sets (Randomized SMILES)*

| Training dataset → <br> Test dataset ↓ | Casey                                    | CCDC CHNO                               | CCDC                                    | Rice                                      | Huang & Massa                           | Mathieu                                 |
| -------------------------------------- | ---------------------------------------- | --------------------------------------- | --------------------------------------- | ----------------------------------------- | --------------------------------------- | --------------------------------------- |
| **Train on Casey**                     | 0.0153 (1)<br>0.0126 (1)<br>0.8557 (33)  | –                                       | –                                       | –                                         | –                                       | –                                       |
| **Train on CCDC CHNO**                 | 0.0630 (1)<br>0.0444 (2)<br>–0.8221 (23) | 0.0249 (0)<br>0.0251 (0)<br>0.9272 (34) | –                                       | 0.0366 (38)<br>0.0333 (17)<br>0.8828 (43) | –                                       | –                                       |
| **Train on CCDC**                      | 0.0367 (0)<br>0.0276 (1)<br>0.4023 (35)  | 0.0231 (1)<br>0.0232 (3)<br>0.9378 (21) | 0.0295 (1)<br>0.0299 (2)<br>0.9697 (32) | 0.0340 (14)<br>0.0327 (3)<br>0.8883 (28)  | 0.0397 (2)<br>0.0261 (1)<br>0.8609 (36) | 0.0420 (2)<br>0.0287 (1)<br>0.7227 (24) |
| **Literature SOTA**                    | MAE = 0.0113                             | –                                       | –                                       | MAE = 0.0352                              | MAE = 0.0625                            | R² = 0.798<br>RMSE = 0.0623             |


*Results averaged over 3 runs, each using 30-fold test-time augmentation (TTA).Results averaged over 3 runs, each using 30-fold TTA
For model results, values are reported as MAE / RMSE / R² (top → bottom)*

# Accessibility
The environment of the project can be readily built from the density.yml file:

```
conda env create -f density.yml
```

Then, you can go to the environment by command:

```
conda activate density
```
