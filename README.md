# Day-Ahead PJM Load Forecasting with a Time-Series Foundation Model

Fine-tuning Chronos-Bolt, a pretrained time-series foundation model, to forecast hourly electricity demand in the PJM East region 24 hours ahead, evaluated with a rolling-origin backtest against zero-shot and classical baselines.

## Overview

Time-series foundation models are pretrained on very large collections of time series and can forecast new series without task-specific training. This project asks a practical question for grid operations: how well does such a model forecast day-ahead electricity demand out of the box, and how much does it gain from fine-tuning on the target system?

Four models are compared:

1. **Chronos-Bolt (small), zero-shot**: the pretrained model applied directly, with no training on PJM data.
2. **Chronos-Bolt (small), fine-tuned**: the same model further trained on the PJM training period.
3. **Seasonal naive**: the load at the same hour on the previous day, a strong reference for demand with a stable daily shape.
4. **LightGBM (AutoGluon DirectTabular)**: gradient-boosted trees on lagged load and calendar features.

## Evaluation

The last 14 days of the data are held out. For each test day, every model receives all load data up to midnight and forecasts the next 24 hours, so no forecast uses information from the future (rolling-origin backtest). Accuracy is measured with MAPE and RMSE, and the calibration of the fine-tuned model's 10-90% prediction interval is checked against the ideal 80% coverage.

## Selected results

Mean over the 14 test days (24-hour-ahead forecasts):

| Model | MAPE (%) | RMSE (MW) |
|-------|----------|-----------|
| Chronos-Bolt fine-tuned | | |
| Chronos-Bolt zero-shot | | |
| LightGBM (DirectTabular) | | |
| Seasonal naive | | |

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
- Extend the data to several years to capture seasonal effects
- Fine-tune larger Chronos models on a GPU cluster and compare accuracy with model size
