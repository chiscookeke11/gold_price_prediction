# Gold Price Prediction

An exploratory machine-learning project that estimates the price of the **SPDR Gold Shares ETF (`GLD`)** from same-day market indicators. The complete workflow lives in a Jupyter notebook and covers data inspection, correlation analysis, a random-forest regression model, and a chart comparing held-out observations with predictions.

> **Important:** despite the repository name, this is not a production gold-price forecasting system. The model uses contemporaneous values for the input markets, and the notebook uses a random train/test split. It therefore demonstrates regression on this historical dataset rather than a time-ordered, forward-looking forecast.

## Project layout

```text
.
├── Gold_Price_prediction.ipynb       # Exploratory analysis, training, evaluation, and plots
├── README.md                         # Project documentation
└── sample_data/
    └── gld_price_data.csv            # Historical market dataset used by the notebook
```

## What the notebook does

1. Imports NumPy, pandas, Matplotlib, seaborn, and scikit-learn.
2. Loads `sample_data/gld_price_data.csv` into a pandas `DataFrame`.
3. Reviews the rows, schema, descriptive statistics, and missing-value counts.
4. Computes a numeric correlation matrix and visualizes it with a heatmap.
5. Plots the distribution of the `GLD` target.
6. Removes `Date` and `GLD` from the feature matrix.
7. Splits the data into 80% training and 20% test sets with `random_state=2`.
8. Fits a `RandomForestRegressor` with 100 trees.
9. Evaluates predictions with the coefficient of determination (R²) and plots actual and predicted test-set values.

## Dataset

The included CSV contains **2,290 records** spanning **January 2, 2008 through May 16, 2018**. It has no missing values in its six columns.

| Column | Role | Description |
| --- | --- | --- |
| `Date` | Excluded from model | Observation date, stored as text in the CSV. |
| `SPX` | Feature | S&P 500 index value. |
| `GLD` | Target | SPDR Gold Shares ETF price to estimate. |
| `USO` | Feature | United States Oil Fund price. |
| `SLV` | Feature | iShares Silver Trust price. |
| `EUR/USD` | Feature | Euro/U.S. dollar exchange rate. |

The model input is therefore `SPX`, `USO`, `SLV`, and `EUR/USD`; `GLD` is the regression target. In the notebook's correlation output, `SLV` has the strongest positive linear correlation with `GLD` (about 0.87). Correlation is descriptive only and does not establish causation.

## Quick start

### 1. Clone and enter the repository

```bash
git clone <repository-url>
cd gold_price_prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\Activate.ps1     # Windows PowerShell
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 4. Launch the notebook

Run this command from the repository root so the relative CSV path resolves correctly:

```bash
jupyter notebook Gold_Price_prediction.ipynb
```

Open the notebook in Jupyter and use **Run All**. The final cells train the model, print its R² score, and display the actual-versus-predicted plot.

## Reproducing the model in Python

The notebook is the source of truth. The following compact script reproduces its data split, feature selection, estimator configuration, and metric calculation:

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

gold_data = pd.read_csv("sample_data/gld_price_data.csv")

X = gold_data.drop(["Date", "GLD"], axis=1)
y = gold_data["GLD"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)
print(f"R²: {r2_score(y_test, predictions):.4f}")
```

The notebook's saved output reports an R² of approximately **0.989**. `RandomForestRegressor` does not set a `random_state` in the current notebook, so fitting again can produce slightly different predictions and scores. Set `random_state` on the estimator if exact repeatability is required.

## Interpreting the evaluation

R² measures the fraction of target variance explained relative to predicting the mean target value; a value nearer to 1 is better on the selected test data. This metric should be interpreted in the context of the experiment:

- The test observations are randomly sampled from the same historical period as the training observations.
- Features are same-day market values, so they may not be available before the target price is known.
- The notebook does not report error in price units (for example, MAE or RMSE), prediction intervals, or performance on a later untouched time period.
- Historical relationships can change, and a strong result on this dataset is not evidence of future investment performance.

## Suggested next steps

- Parse `Date`, sort chronologically, and evaluate with a walk-forward or time-series split.
- Use lagged values and features that would be known at prediction time.
- Add MAE and RMSE alongside R², and compare with simple baselines.
- Set model and split seeds, pin dependency versions, and add an automated training/evaluation script.
- Tune the forest with cross-validation that respects temporal ordering.
- Inspect feature importance and validate results on a more recent out-of-sample period.

## Notes

- The notebook currently uses `sns.distplot`, which seaborn has deprecated. A future cleanup can replace it with `sns.histplot` or `sns.displot`.
- This repository does not currently include a license file. Confirm the intended license and the provenance/terms of any dataset before redistributing or using the project beyond experimentation.
