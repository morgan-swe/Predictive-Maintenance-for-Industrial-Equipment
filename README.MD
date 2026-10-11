# Predictive Maintenance for Rotating Industrial Equipment

Predicting how many operating cycles a machine has left before it fails, using sensor telemetry and machine learning.

Built on the NASA C-MAPSS FD001 turbofan dataset, but the problem is the same one energy infrastructure companies deal with every day: compressors, pumps, and turbines running continuously along a pipeline or inside a plant, instrumented with dozens of sensors, where an unplanned failure is expensive and can carry real safety and environmental risk.

The alternative to prediction is either fixed maintenance schedules, which service healthy equipment for no reason, or running to failure, which is worse. This project asks whether continuous sensor data can tell you which units actually need attention and when.

**Final result: the model predicts remaining life within about 10 operating cycles on average, roughly 5% of a typical engine's lifespan, at an R² of 0.865.**

## Results

Four models, all trained on 80 engines and tested on 20 engines they had never seen. Error is in operating cycles, against an average engine lifetime of 206 cycles.

| Model                        | MAE (cycles) | RMSE (cycles) | R²        |
| ---------------------------- | ------------ | ------------- | --------- |
| Linear Regression (baseline) | 14.17        | 17.65         | 0.821     |
| Decision Tree                | 11.63        | 17.11         | 0.832     |
| **Random Forest**            | **10.29**    | **15.34**     | **0.865** |
| Random Forest (tuned)        | 10.23        | 15.23         | 0.867     |

A 108 combination GridSearchCV run with 5 fold cross validation improved MAE by only 0.06 cycles. Since the two forests were effectively tied, the untuned Random Forest was chosen as the production model: same accuracy, cheaper to train and serve. Tuning was still worth running, because knowing a model has hit its ceiling is a result.

![Model comparison](reports/figures/model_comparison.png)

## What the analysis actually found

The parts of this project I would want to talk about in an interview.

**Only one kind of feature earned its keep.** I generated 63 engineered features across three families and then grouped them by type to see which style of feature engineering paid off:

| Feature type                    | Count | Total importance |
| ------------------------------- | ----- | ---------------- |
| Rolling mean                    | 21    | 0.840            |
| Time and operational settings   | 4     | 0.145            |
| Raw sensor                      | 21    | 0.013            |
| Rolling standard deviation      | 21    | 0.002            |
| Trend (cycle over cycle change) | 21    | 0.0001           |

Rolling means carried 84% of the predictive signal. Rolling standard deviation and trend features contributed almost nothing, and the reason matters: both are built to detect erratic, jumpy behaviour, and degradation in this dataset is a slow steady drift. Rolling std is the right tool for a bearing that starts vibrating unevenly before its average shifts. That failure mode just isn't in this data. A feature is a question you ask the data, and it only pays off when the question matches the real failure mechanism.

**The model is accurate exactly where the decision gets made.** Broken down by actual remaining life, error is lowest in the 0 to 25 cycle band at 3.9 cycles MAE, and peaks at 18.1 cycles in the middle of an engine's life. Being sharpest when a unit is close to failure is the behaviour you want out of a maintenance tool.

**Direction of error matters more than size.** Predicting too little life means servicing something early, which wastes money but is safe. Predicting too much promises life that isn't there. MAE scores both the same, a maintenance schedule does not. Digging into the worst predictions, all ten belonged to a single engine (unit 84) that the model consistently called half dead while it was nearly new, so the fix is to rank engines rather than rows.

**Where I would and wouldn't trust it.** This model is solid as a second opinion for ranking which units need attention first. I would not schedule maintenance directly off the number, because it is systematically optimistic about some of the engines that fail early. That distinction is the difference between a model that works in a notebook and one you can put in front of an operations team.

## How it was built

**Exploration.** Profiled 100 engine units, 21 sensors, and 20,631 cycle records. Screened sensors by variance to find flat, uninformative channels, and used a correlation matrix to catch redundant sensors measuring the same physical process (sensors 9 and 14 turned out to be perfectly correlated).

**Feature engineering.** Built the remaining useful life target from cycle to failure data, then generated rolling means, rolling standard deviations, and cycle over cycle deltas for all 21 sensors, giving 63 engineered features on top of the raw inputs.

**Preventing leakage.** The train/test split is by engine unit using `GroupShuffleSplit`, not by row. A random row split would put cycles from the same engine on both sides of the boundary and produce accuracy that would collapse in production. Scaling is fit on the training set only. Remaining useful life is capped at 125 cycles, since sensors carry little degradation signal early in a unit's life.

**Modelling and evaluation.** Trained linear regression, decision tree, and random forest regressors, scored each on MAE, RMSE, and R², then ran feature importance analysis, hyperparameter tuning, and a full error breakdown by engine and by remaining life band.

**Delivery.** Trained models are serialized with joblib and served two ways: a Streamlit app for interactive per engine predictions, and a Power BI report for the reporting side.

## Applications

**Streamlit app** (`app/`) loads any of the four saved models and predicts remaining life for a selected engine at its most recent cycle, alongside that model's recorded metrics. Features are reindexed against `feature_names_in_` on every prediction, because passing 88 columns in the wrong order gives silently wrong answers rather than an error.

```bash
streamlit run app/app.py
```

**Power BI report** (`dashboard/`) covers predicted versus actual remaining life, model comparison, feature importance, and a risk breakdown that buckets every engine into Critical (under 30 cycles), Maintenance Soon (30 to 75), or Healthy, with a DAX measure tracking the critical count. Across the 20 test engines, 560 of 3,990 predicted cycles land in Critical.

![Dashboard](dashboard/images/Predictive%20Maintenance%20Dashboard.png)

## Project structure

```
data/raw/         Original NASA C-MAPSS FD001 files
data/interim/     Parsed and validated CSVs
data/processed/   Feature engineered dataset, 91 columns
notebooks/        Analysis, versioned stage by stage
models/           Serialized trained models
reports/          Feature importance, error analysis, figures
app/              Streamlit application
dashboard/        Power BI report and its exported data
src/              Data loading and config
```

Notebooks run in order:

1. `01_data_overview.ipynb` Dataset profiling, sensor variance, correlation analysis
2. `02_feature_engineering.ipynb` Target construction, rolling and trend features, leakage safe split
3. `03_model_training.ipynb` Training, tuning, feature importance, error analysis, dashboard exports

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/load_data.py        # parse raw files into data/interim
```

Then run the notebooks in order, or launch the Streamlit app against the saved models directly.

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Streamlit, Power BI, Jupyter, Git

## Next steps

- Move the notebook logic into dedicated `src/` modules (feature engineering, preprocessing, training, evaluation) so the whole pipeline runs end to end from the command line. Right now `src/load_data.py` handles raw loading and validation, and the rest of the analysis lives in the notebooks.
- Score models with an asymmetric metric that penalises over prediction harder than under prediction, since the two carry very different operational costs.
- Rank at the engine level rather than the row level, which the error analysis showed is the more useful unit of comparison.
- Validate against FD002 to FD004, which add multiple operating conditions and fault modes and are closer to how real equipment behaves.
