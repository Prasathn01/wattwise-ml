# WattWise: Household Energy Consumption Analysis and Peak-Load Prediction

Machine-learning analysis of one household's electricity use from the UCI *Individual Household Electric Power Consumption* dataset: exploratory analysis, next-hour load regression, peak-hour classification, clustering of daily load profiles, sequential models (Markov chain, HMM, Kalman filter) and tree ensembles.

| Notebook | Open in Google Colab |
|---|---|
| PBL 1: Units 1 and 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Prasathn01/wattwise-ml/blob/main/notebooks/PBL1_Units1_2.ipynb) |
| PBL 2: Units 3 and 4 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Prasathn01/wattwise-ml/blob/main/notebooks/PBL2_Units3_4.ipynb) |
| PBL 3: Unit 5 | *to be added* |

## Team and course

| Member | Register number |
|---|---|
| Amit Kumar Singh | RA2411026011138 |
| N Ram Prasath | RA2411026011139 |
| G Viswa Deepak Raga | RA2411026011140 |
| Suriya Narayanan S | RA2411026011141 |
| Sidharth S | RA2411026011142 |

- **Course:** 21CSC305P Machine Learning
- **Faculty:** Dr. Mohammad Ishtiyaq Qureshi, Assistant Professor, Department of Computational Intelligence
- **Institution:** SRM Institute of Science and Technology, Kattankulathur
- **Team:** Team 4

## Problem

The project uses four years of minute-level electricity readings from a single house to answer questions from each unit of the course:

- **Unit 1 (analysis):** how the household's power is distributed, how it varies over the day, week and year, how it splits across sub-meters, and how probability rules (expectation, Bayes' rule, independence) and polynomial curve fitting describe it.
- **Unit 2 (supervised learning):** predict the next hour's average power (regression) and whether the next hour will be a *peak* hour (classification), using only information available before the hour starts.
- **Unit 3 (unsupervised learning):** find the kinds of day the household has, by clustering 24-hour daily load profiles and reducing their dimensionality.
- **Unit 4 (sequential models):** model the hourly load as a sequence with a Markov chain on load levels, a hidden Markov model and a linear dynamical system (Kalman filter), and compare their next-hour predictions.
- **Unit 5 (tree ensembles):** revisit the peak-hour classification task of Unit 2 with CART, bagging, random forests, boosting, GAMs, Bayesian model averaging and stacking.

## Dataset

**UCI Machine Learning Repository, *Individual Household Electric Power Consumption* (dataset ID 235).** Donors: Georges Hebrail and Alice Berard.
Page: <https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption>
Download (zip, about 20 MB): <https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip>. It unzips to `household_power_consumption.txt` (about 127 MB).

- One house in Sceaux, about 7 km from Paris, France.
- One reading per minute from 16 December 2006 17:24 to 26 November 2010: **2,075,259 rows** over 47 months.
- Field separator `;`. Missing readings are written as `?`. 25,979 rows (1.25%) are missing, always as whole rows, in 71 gaps; the longest gap is about 5 days.
- First data row: `16/12/2006;17:24:00;4.216;0.418;234.840;18.400;0.000;1.000;17.000`

| Column | Meaning | Unit |
|---|---|---|
| Date | Day, dd/mm/yyyy | – |
| Time | Minute, hh:mm:ss | – |
| Global_active_power | Average power the whole house used in that minute. **The main variable of the project.** | kW |
| Global_reactive_power | Power that flows back and forth and is not consumed | kW |
| Voltage | Supply voltage | V |
| Global_intensity | Current drawn | A |
| Sub_metering_1 | Kitchen: dishwasher, oven, microwave | Wh |
| Sub_metering_2 | Laundry room: washing machine, tumble dryer, fridge, a light | Wh |
| Sub_metering_3 | Electric water heater and air conditioner | Wh |

Energy used in the rest of the house (Wh per minute) = Global_active_power × 1000 / 60 − Sub_metering_1 − Sub_metering_2 − Sub_metering_3.

**The dataset is not stored in this repository.** The text file is larger than GitHub's 100 MB file limit. Each notebook downloads and unzips the UCI archive itself when `household_power_consumption.txt` is not present in the working folder, and `.gitignore` excludes the data file and zip archives.

## Repository structure

```
wattwise-ml/
├── README.md
├── LICENSE                     MIT, copyright 2026 Team 4, SRM IST
├── .gitignore                  Python template + household_power_consumption.txt, *.zip, .ipynb_checkpoints/
├── requirements.txt            packages needed to run the notebooks
├── requirements-test.txt       extra packages needed only by tests/
├── notebooks/
│   ├── PBL1_Units1_2.ipynb     PBL 1 (Units 1-2), outputs kept
│   └── PBL2_Units3_4.ipynb     PBL 2 (Units 3-4), outputs kept
├── tests/
│   └── verify_pbl2.py          checks the from-scratch PBL 2 code against standard libraries
├── results/
│   └── pbl2_results.json       numbers printed by the PBL 2 run
├── figures/
│   └── pbl2/                   PNG figures from the PBL 2 run
└── docs/                       PBL 1 and PBL 2 reports and review slides (PDF)
```

The PBL 3 notebook (`notebooks/PBL3_Unit5.ipynb`), its results (`results/pbl3_results.json`) and figures (`figures/pbl3/`) will be added when the Unit 5 review is complete.

## How to run

**In Google Colab.** Click a badge above, then choose *Runtime → Run all*. The notebook downloads the dataset (about 20 MB) on its first run.

**Locally.**

```bash
pip install -r requirements.txt
jupyter notebook
```

Then open a notebook from `notebooks/` and run all cells. The dataset is downloaded into the notebook's working folder.

**Tests.** The PBL 2 algorithms are written from scratch; `tests/verify_pbl2.py` recomputes each result with a standard library and reports PASS or FAIL per check.

```bash
pip install -r requirements-test.txt
python tests/verify_pbl2.py notebooks/PBL2_Units3_4.ipynb .
```

## PBL 1 summary: Units 1 and 2

Notebook: [`notebooks/PBL1_Units1_2.ipynb`](notebooks/PBL1_Units1_2.ipynb). Report: [`docs/PBL1_Report.pdf`](docs/PBL1_Report.pdf).

**Preprocessing**

- Minute readings are resampled to **34,589 hourly means**. Gaps of 3 hours or less are time-interpolated; 399 outage hours remain missing and are dropped.
- **Peak label:** an hour is a peak if its mean power is at least **1.596 kW**, the 75th percentile of the 2006–2009 hours.
- **Chronological split:** train December 2006 – December 2009 (26,191 hours), test January – November 2010 (7,264 hours). Features are standardised with training statistics only.
- **37 features**, all known before the hour starts: load lags of 1, 2, 3, 24 and 168 hours; mean of the previous 24 hours; previous-hour voltage, reactive power and sub-meters; hour-of-day one-hot; a weekend flag; month as sine and cosine.
- **Global_intensity is excluded** as a feature: its correlation with active power is 0.999, so it would leak the target.

**Unit 1 results**

- **Distributions:** active power is right-skewed and bimodal (mean 1.09 kW, median 0.60 kW). Voltage is close to N(240.8, 3.24²).
- **Expectation:** two daily peaks, about 1.5 kW at 07–08 h and about 1.9 kW at 19–21 h. Weekends average 1.23 kW against 1.04 kW on weekdays.
- **Sub-metering:** the water heater and air conditioner take 35.5% of the energy, and 51% of the energy is not sub-metered.
- **Bayes' rule:** P(peak | evening) = P(evening | peak) × P(peak) / P(evening) = 0.403 × 0.245 / 0.209 = 0.473.
- **Independence:** peak and weekend are dependent (χ² = 290.5, p ≈ 10⁻⁶⁵).
- **Polynomial curve fitting** of the hourly consumption curve: test error is lowest at order M = 6 (E_RMS 0.326); M = 16 with regularisation λ = 0.01 gives 0.318.

**Unit 2 regression: next-hour Global_active_power, test year 2010**

| Model | RMSE (kW) | MAE (kW) | R² |
|---|---|---|---|
| Persistence (y[t−1]) | 0.618 | 0.407 | 0.371 |
| Ordinary least squares | 0.504 | 0.362 | 0.581 |
| Huber (robust) | 0.508 | 0.353 | 0.575 |
| Ridge (λ by time-series CV) | 0.504 | 0.362 | 0.581 |
| Bayesian linear regression | 0.504 | 0.362 | 0.581 |

**Unit 2 classification: peak / off-peak hour, test year 2010**

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Gaussian Naive Bayes | 0.696 | 0.566 | 0.851 |
| LDA | 0.850 | 0.586 | 0.900 |
| Bayesian logistic regression (Laplace) | 0.851 | 0.599 | 0.902 |
| Linear SVM | 0.849 | 0.579 | 0.902 |
| RBF-kernel SVM | 0.854 | 0.613 | 0.897 |

## PBL 2 summary: Units 3 and 4

Notebook: [`notebooks/PBL2_Units3_4.ipynb`](notebooks/PBL2_Units3_4.ipynb). Report: [`docs/PBL2_Report.pdf`](docs/PBL2_Report.pdf). Numbers: [`results/pbl2_results.json`](results/pbl2_results.json). Figures: [`figures/pbl2/`](figures/pbl2/).

**Implementation rule.** Every algorithm in this notebook is written from its equations in NumPy. pandas is used only to read the file and average minutes into hours, and matplotlib only to plot. No scikit-learn, SciPy, hmmlearn or statsmodels appears in the notebook; those libraries are used only in `tests/verify_pbl2.py`, which checks that the code is correct (see *Implementation and verification* below).

**Preprocessing**

- **Hourly load:** an hour is kept only if at least 45 of its 60 minutes exist, giving **34,151 hours**.
- **Unit 3 input:** **1,411 complete days**, each a row of 24 hourly values, with every hour column standardised.
- **Unit 4 input:** gaps of 2 hours or less are interpolated; longer gaps split the series into segments.
- **Unit 4 split:** chronological; the first 80% is training (27,492 hours), the last 20% is test (6,681 hours, from 11 February 2010).
- **Unit 4 levels:** Low < 0.40 kW ≤ Medium < 1.40 kW ≤ High (training terciles).

**Exploratory analysis**

- Hourly mean 1.09 kW, median 0.80 kW.
- On weekdays the busiest hour is 21:00 (1.86 kW) and the quietest is 03:00 (0.41 kW), with a second, smaller peak in the morning around 07:00.
- December averages 1.49 kW and August 0.57 kW.
- Autocorrelation is 0.71 at 1 hour, 0.44 at 24 hours and 0.46 at 168 hours.

**Unit 3: unsupervised learning on daily profiles**

- **K-means** (k-means++ initialisation, 20 restarts): K = 2 is chosen because it has the best silhouette (0.20).
  - *Everyday* days: 916 days, 20.7 kWh per day, 19% weekends, 12% in December–February.
  - *Heavy* days: 495 days, 36.4 kWh per day, 45% weekends, 46% in December–February.
- **Gaussian mixture fitted with EM:** BIC chooses full covariance. Silhouette 0.05, Davies–Bouldin 3.49, adjusted Rand index 0.09 against K-means.
- **Hierarchical clustering** (single, complete, average and Ward linkage, with the Lance–Williams update): Ward gives 887 / 524 days (silhouette 0.11, ARI 0.17 against K-means). Single and average linkage chain 1,410 days into one cluster.
- **PCA:** PC1 explains 29.3% of the variance and PC2 14.6%. 14 components are needed for 90% of the variance, and 6 have an eigenvalue above 1.
- **Factor analysis (EM):** 13 factors are best by held-out likelihood.

**Unit 4: sequential models on the hourly series**

- **Markov chain** on the three levels: next-hour level accuracy 63.1%.
- **Hidden Markov model** (3 states, Gaussian emissions of log kW, Baum–Welch with scaled forward–backward, 48 iterations):
  - States: Low 0.29 kW, Medium 0.74 kW, High 1.93 kW.
  - Stay probabilities 0.77 / 0.50 / 0.81, i.e. average stays of about 4.4 h, 2.0 h and 5.3 h.
- **Viterbi decoding** agrees with the hour-by-hour posterior state on 98.0% of test hours.
- **Kalman filter / linear dynamical system:** a hidden "unusual part" on top of the usual load for each hour of the day; EM learned a = 0.70, Q = 0.32, R = 0.02.

**Unit 4 next-hour comparison (6,677 test forecasts)**

| Model | Level accuracy | MAE (kW) | RMSE (kW) |
|---|---|---|---|
| Persistence | 63.1% | 0.395 | 0.600 |
| Hour-of-day average | 51.2% | 0.508 | 0.668 |
| Markov chain on levels | 63.1% | 0.446 | 0.586 |
| HMM (forward filter + A) | 60.0% | 0.438 | 0.586 |
| Kalman filter (AR(1) LDS) | 61.1% | **0.366** | **0.514** |

PBL 1 and PBL 2 use different preprocessing (hour-completeness rules, splits and test periods), so their numbers are not comparable with each other.

## PBL 3 summary: Unit 5

To be added when the Unit 5 notebook is complete. It will revisit the PBL 1 peak-hour task (same features, threshold and 2010 test year) with CART, bagging, random forests, AdaBoost, gradient boosting, a GAM, Bayesian model averaging and stacking.

## Implementation and verification

The PBL 2 notebook implements every algorithm from scratch in NumPy. `tests/verify_pbl2.py` loads the notebook's code, reruns it on the same data and compares each result with a standard library implementation:

| From-scratch function | Checked against |
|---|---|
| K-means, silhouette, Davies–Bouldin, adjusted Rand index, Gaussian mixture (EM), PCA, factor analysis | scikit-learn |
| Hierarchical clustering (single, complete, average, Ward) | SciPy |
| Baum–Welch, forward–backward, Viterbi | hmmlearn |
| Kalman filter log-likelihood | statsmodels |

EM finds a local maximum, so for the Gaussian mixture the test checks that scikit-learn, started from our solution, stays there (the same fixed point) rather than expecting the same number from a random start.

## Limitations

- The data come from one household, so the findings do not generalise to other homes without further work.
- Hourly averages hide which appliance caused a spike within the hour.
- No weather data are used, although temperature drives the heating and cooling load.
- PBL 1 and PBL 2 use different preprocessing, so their results cannot be compared directly.

## Contributions

Roles from the PBL 1 report:

- **Amit Kumar Singh:** data loading and preprocessing, summary statistics and EDA plots.
- **N Ram Prasath:** continuous-random-variable analysis, leakage check, polynomial curve fitting; PBL 2 team lead.
- **G Viswa Deepak Raga:** peak label, Bayes' rule, independence tests.
- **Suriya Narayanan S:** feature engineering, regression models; PBL 1 team lead.
- **Sidharth S:** classification models and their comparison.

## Citation

> G. Hebrail and A. Berard, "Individual Household Electric Power Consumption," UCI Machine Learning Repository, 2012. [Online]. Available: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

## License

MIT. See [LICENSE](LICENSE).
