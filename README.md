# Day-Ahead PJM Load Forecasting with a Time-Series Foundation Model

Forecasting hourly PJM East electricity demand 24 hours ahead with Chronos-Bolt, a pretrained time-series foundation model, used zero-shot and fine-tuned, and evaluated with a rolling-origin backtest against classical baselines.

## Overview

Time-series foundation models are pretrained on very large collections of time series and can forecast new series without task-specific training. This project asks a practical question for grid operations: how well does such a model forecast day-ahead electricity demand out of the box, and how much does it gain from fine-tuning on the target system?

Four models are compared:

1. **Chronos-Bolt (small), zero-shot**: the pretrained model applied directly, with no training on PJM data.
2. **Chronos-Bolt (small), fine-tuned**: the same model further trained on the PJM training period.
3. **Seasonal naive**: the load at the same hour on the previous day, a strong reference for demand with a stable daily shape.
4. **LightGBM (AutoGluon DirectTabular)**: gradient-boosted trees on lagged load and calendar features.

## Evaluation

The last 14 days of the data are held out. For each test day, every model receives all load data up to midnight and forecasts the next 24 hours, so no forecast uses information from the future (rolling-origin backtest). Accuracy is measured with MAPE and RMSE, and the calibration of the fine-tuned model's 10-90% prediction interval is checked against the ideal 80% coverage.

## Results

Mean over the 14 test days (24-hour-ahead forecasts, 22 April to 5 May 2017):

| Model | MAPE (%) | RMSE (MW) | MAPE reduction vs seasonal naive |
|-------|----------|-----------|----------------------------------|
| Chronos-Bolt zero-shot | **4.02** | **1,315** | **34.0%** |
| Chronos-Bolt fine-tuned | 4.43 | 1,454 | 27.3% |
| LightGBM (DirectTabular) | 4.67 | 1,556 | 23.3% |
| Seasonal naive | 6.09 | 1,935 | - |

**What the results show**

- The pretrained foundation model, used zero-shot with no training on PJM data, gave the most accurate forecasts: 34% lower MAPE than the seasonal naive forecast and 14% lower than the LightGBM model trained on this series.
- Fine-tuning did not help here. With only 2,664 hours (about 16 weeks) of training data, further training moved the model away from what it learned in pretraining and slightly overfit the training period. Fine-tuning is likely to pay off with several years of data, and that is the natural next experiment.
- The fine-tuned model's 10-90% prediction intervals contained only 41% of the actual hourly loads, against an ideal 80%. Its uncertainty bands are too narrow, which is consistent with overfitting the short training period and shows why calibration should be checked alongside point accuracy.
- Both Chronos variants still beat the classical baselines, which suggests that pretraining on a large and diverse collection of time series transfers well to electricity demand.

## Data

Hourly PJM East (PJME) demand, January to May 2017 (3,000 hours), from the PJM hourly energy consumption series published by PJM Interconnection. The same demand series drives the [battery arbitrage project](https://github.com/ekahorsu/battery-arbitrage-reinforcement-learning). The single missing hour at the daylight saving time change is filled by time interpolation.

## How to run

The easiest way is Google Colab with a free GPU (Runtime > Change runtime type > T4 GPU). Open `pjm_load_forecasting_chronos.ipynb` and run all cells; installation and training take about 15 to 20 minutes.

To run locally:

```bash
git clone https://github.com/ekahorsu/pjm-load-forecasting-chronos.git
cd pjm-load-forecasting-chronos
pip install autogluon.timeseries
jupyter notebook pjm_load_forecasting_chronos.ipynb
```

## Tech stack

- **AutoGluon TimeSeries** for a common interface to all models
- **Chronos-Bolt** (Amazon) pretrained time-series foundation model, with **PyTorch** for fine-tuning
- **LightGBM** for the tabular baseline
- **pandas, NumPy, Matplotlib** for data handling, evaluation and plots

## Repository contents

- `pjm_load_forecasting_chronos.ipynb` - the full analysis notebook
- `pjm_east_hourly_load.csv` - hourly PJM East load (MW)

## Next steps

- Add temperature forecasts as covariates, since weather drives most load forecast error
- Extend the data to several years to capture seasonal effects and test whether fine-tuning then improves on zero-shot accuracy
- Fine-tune larger Chronos models on a GPU cluster and compare accuracy with model size
