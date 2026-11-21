# Gold Volatility Prediction Using an Artificial Neural Network

An end-to-end deep learning project for **forecasting gold-market risk** by predicting the standard deviation of gold's daily percentage returns over the **next five trading days**. This is a **regression** task: the model estimates volatility, **not** the future gold price or whether it will rise or fall.

## 1. Problem and objective

Gold prices can move sharply during periods of financial uncertainty. Predicting the *magnitude* of future price movements can help describe market risk even when the direction of the next move is difficult to predict.

**Objective:** develop, tune, evaluate, save, and reload an Artificial Neural Network (ANN) that forecasts five-day-ahead gold volatility from historical gold and related market indicators.

## 2. Data sources

The notebook downloads historical daily data beginning **January 1, 2005** from:

| Source | Variables |
|---|---|
| Yahoo Finance | Gold futures (`GC=F`), silver futures (`SI=F`), US Dollar Index (`DX-Y.NYB`), oil futures (`CL=F`), US 10-year Treasury yield (`^TNX`), VIX (`^VIX`) |
| FRED | 10-year real yield (`DFII10`) and 10-year breakeven inflation expectations (`T10YIE`) |

The notebook shifts FRED observations by one day to account for publication delay, aligns series by date, forward-fills available observations, and removes remaining missing rows. Data availability and results may change when the notebook is rerun.

**Recorded notebook run:** 5,502 aligned observations, January 4, 2005–November 20, 2026. Feature engineering produced 5,477 labeled rows with 22 input features.

## 3. End-to-end workflow

1. **Collect and explore data:** inspect missing values, distributions, returns, correlations, and rolling gold volatility.
2. **Engineer features:** five recent gold returns and their absolute values; 10-day and 20-day gold volatility; VIX level; other-market changes; real-yield and inflation-expectation levels and changes.
3. **Create target:** standard deviation of daily gold percentage returns over the next five trading days.
4. **Split chronologically:** approximately 70% training, 15% validation, and 15% testing, without shuffling. Leave a five-trading-day gap at the split boundaries to reduce forward-target overlap.
5. **Scale features:** fit `StandardScaler` on training data only, then transform validation and test data.
6. **Train and tune ANN:** compare four configurations using validation loss.
7. **Evaluate:** report MAE, RMSE, and R² on the held-out test period; compare against simple volatility baselines.
8. **Save, reload, and infer:** persist the trained model and preprocessing metadata, verify reload consistency, and expose a prediction function.

## 4. ANN architecture

The model is implemented in **TensorFlow / Keras**:

```text
Input: 22 engineered features
  → Dense(units, ReLU, L2)
  → Dropout(dropout)
  → Dense(units // 2, ReLU, L2)
  → Dropout(dropout)
  → Dense(1)  [predicted volatility]
```

- **Optimizer:** Adam
- **Loss:** Mean Squared Error (MSE)
- **Regularization:** L2 penalties, Dropout, and Early Stopping
- **Early Stopping:** monitor validation loss, patience of 10 epochs, restore best weights
- **Tuning runs:** up to 100 epochs each, batch size 32
- **Final training:** up to 200 epochs, batch size 32, with Early Stopping

## 5. Hyperparameter tuning

Four configurations were tested manually. **Validation loss** was used to select the best configuration; the test set was reserved for final evaluation.

| Trial | First hidden-layer units | Dropout | L2 strength | Learning rate | Best validation loss |
|---|---:|---:|---:|---:|---:|
| 1 | 32 | 0.10 | 0.0001 | 0.0010 | 0.2346 |
| **2 — selected** | **64** | **0.20** | **0.0001** | **0.0010** | **0.2242** |
| 3 | 128 | 0.30 | 0.0010 | 0.0010 | 0.2847 |
| 4 | 64 | 0.20 | 0.0001 | 0.0003 | 0.2386 |

The selected model uses **64 and 32 neurons** in its two hidden layers. These are results recorded in the supplied notebook, not averages across repeated runs.

## 6. Test-set performance

The final ANN was compared with a naive prediction using the last 10-day volatility, another using the last 20-day volatility, and a constant prediction equal to the training-set target average.

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| **ANN** | **0.408** | **0.623** | **0.223** |
| Last 10-day volatility | 0.461 | 0.708 | -0.002 |
| Last 20-day volatility | 0.447 | 0.684 | 0.065 |
| Training-set average | 0.483 | 0.726 | -0.054 |

**Interpretation:** on this recorded test run, the ANN achieved lower MAE and RMSE and higher R² than all three baselines. An R² of 0.223 indicates that predictive performance remains limited; the model is not a guarantee of future market behavior. MAE and RMSE are expressed in percentage-point units of the daily-return volatility target.

## 7. Limitations

- Forecasts **volatility, not price direction or future price**.
- Sudden news, policy decisions, and market shocks may not be captured by historical indicators.
- Results depend on data availability, market regime, and the particular time-based split.
- The recorded tuning results are from a single notebook run; additional robustness testing would be useful.
- This is an educational risk-forecasting project, **not financial advice**.

## 8. Technologies

Python · TensorFlow/Keras · pandas · NumPy · scikit-learn · Matplotlib · Requests · Joblib

---

**Course:** Final Project — Deep Learning Unit  
**GitHub:** [RaghadQi1/Deep-learning-project](https://github.com/RaghadQi1/Deep-learning-project)
