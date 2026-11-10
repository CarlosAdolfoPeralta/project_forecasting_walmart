🇪🇸 [Leer en español](README.es.md)

# Retail Demand Forecasting (M5, Walmart): Does Machine Learning Beat a Simple Model?

28-day daily sales forecasts for 600 product-store series from Walmart's M5 dataset. A ladder of models, from baselines up to LightGBM and a neural network, is compared under an evaluation protocol fixed in advance, with two metrics (RMSSE and MAE) and bootstrap confidence intervals across series to tell which differences are real. All training was done on CPU.

**Author:** Carlos Adolfo Peralta · [LinkedIn](https://www.linkedin.com/in/carlosperalta1993/) · [GitHub](https://github.com/CarlosAdolfoPeralta)

## Summary

- **Main result:** on the test set, LightGBM (RMSSE 0.768; MAE 1.011) and a simple statistical model, exponential smoothing with a weekly index (0.769; 1.008), are practically tied. The neural network (0.772; 1.023) and the 28-day moving average (0.782; 1.059) follow closely.
- **Machine learning does not reliably improve on the simple model.** According to the bootstrap, LightGBM's MAE advantage over the statistical model (validation) disappears on test, and on RMSSE there is no difference distinguishable from noise. The neural network performs worse than LightGBM on validation.
- **The only thing that holds across both windows:** the statistical model with a weekly index beats the moving average on RMSSE, by 1.2% and 1.7%.
- **Demand is very sporadic:** 68% of daily observations are zero and 91% of the series have intermittent or lumpy demand. An RMSSE below 1 is not a good result on its own: always forecasting zero gives 0.94 and 0.99.
- **RMSSE and MAE do not always rank the models the same way**, which is why they are reported together.

## Data

| | |
|---|---|
| Source | *M5 Forecasting - Accuracy*, the competition organized by the University of Nicosia; copy on Zenodo (record 12636070), CC BY 4.0 license |
| Series | 30,490: 3,049 products across 10 stores in 3 US states (CA, TX and WI) |
| Period | January 29, 2011 to May 22, 2016 (1,941 days) |
| Files | daily sales (`sales_train_evaluation.csv`), calendar with events and SNAP days, and weekly prices |
| Sample | 600 series: 60 at random per store, seed 42. It reproduces the composition of the total (FOODS 48.0% vs. 47.1%) and the intermittency (68.2% of days without sales vs. 68.0%) |

The competition evaluated 42,840 hierarchical series; here only the most disaggregated level (product and store) is used.

**What the exploration showed** (notebook 01):
- **Strong weekly seasonality:** Saturday (1.21) and Sunday (1.20) sell about 20% more than average, and Tuesday through Thursday, 12% less. Annual seasonality is much smaller (0.96 to 1.04).
- **Christmas closure:** on December 25 the stores practically do not sell. Other events, such as Thanksgiving (0.73) and New Year's Day (0.82), sell less than a normal day. SNAP days sell more in all three states.
- **Intermittency:** 72.7% of the series are intermittent and 18.3% lumpy (Syntetos, Boylan and Croston classification). Those series account for 57.6% of units.
- **Late-starting series:** 77.1% have their first sale after the first day and 44.0% have at least one year of leading zeros. Those zeros are not zero demand, which is why the metrics count the history from the first sale.
- **Prices:** they change in only 1.2% of weeks.

## Evaluation protocol

Fixed in notebook 01, before training any model.

- **Horizon:** 28 days, from a single origin and without using any sales after it. The calendar and prices for the forecast period are assumed to be known in advance, as in M5.
- **Windows:** validation from March 28 to April 24, 2016 (days 1,886 to 1,913) and test from April 25 to May 22 (days 1,914 to 1,941). Hyperparameters are chosen on validation; the chosen model is retrained with data up to the test origin and evaluated on test only once. The validation window has no events; the test window has four minor ones. Neither includes the highest-impact holidays.
- **RMSSE (scaled error):** the mean squared error of the forecast divided by that of a naive forecast (repeat the previous day) over the same series' history, with a square root. It is counted from the first sale and averaged unweighted across series.
- **MAE:** mean absolute error in units per series per day.

## Models

| Model | Description |
|---|---|
| Zero | Always forecasts 0 (a reference for understanding the metrics) |
| Naive | Repeats the last sale |
| Seasonal naive | Repeats what was sold on the same day of the previous week |
| 28-day moving average | Average of the last 28 days |
| 4-week seasonal average | Average of the same weekday over the last 4 weeks |
| **SES + weekly index** | Divides each sale by the index of its day of the week (estimated per category using all series together), estimates the level with simple exponential smoothing (with alpha chosen per series, using only data before the origin) and reapplies the index |
| **LightGBM** | A global model with a direct strategy: each row is a series, an origin and a horizon day. 34 features: recent sales, averages, intermittency, series age, same weekday, calendar, SNAP, price and series attributes. Trained on 52 weekly origins; chosen on validation among six configurations (Tweedie loss with exponent 1.5 and 15 leaves, 509 trees) |
| **Neural network (Keras)** | Multilayer perceptron of 256, 128 and 64 units on the same rows and features, with embeddings for the categorical variables, a positive output and Tweedie loss. 3 networks with different seeds are averaged |

An automated test checks that the model's features do not change when sales after the origin are altered, to rule out information leakage.

## Results

| Model | RMSSE validation | RMSSE test | MAE validation | MAE test |
|---|---|---|---|---|
| **LightGBM** | **0.740** | **0.768** | **0.983** | 1.011 |
| 50/50 average of LightGBM and the network | 0.743 | 0.769 | 0.994 | 1.013 |
| **SES + weekly index** | 0.742 | 0.769 | 1.036 | **1.008** |
| Neural network | 0.749 | 0.772 | 1.015 | 1.023 |
| 28-day moving average | 0.751 | 0.782 | 1.020 | 1.059 |
| 4-week seasonal average | 0.811 | 0.849 | 1.045 | 1.101 |
| Zero | 0.940 | 0.990 | 1.392 | 1.477 |
| Naive | 0.936 | 1.007 | 1.361 | 1.306 |
| Seasonal naive | 0.995 | 1.029 | 1.195 | 1.233 |

The 50/50 average is a fixed rule defined in advance, not tuned to the results.

### Are the differences real?

The top four models are within 0.5% of each other on RMSSE, so the 95% confidence interval of the error difference between pairs of models was computed with a paired bootstrap across series (5,000 resamples). A negative difference means the first model makes fewer errors; an asterisk marks the intervals that exclude zero.

| Comparison (A − B) | RMSSE validation | RMSSE test | MAE validation | MAE test |
|---|---|---|---|---|
| LightGBM − SES | −0.0018 (−0.0079 to 0.0046) | −0.0009 (−0.0056 to 0.0039) | −0.0527 (−0.1164 to −0.0051) * | +0.0035 (−0.0120 to 0.0196) |
| Neural network − LightGBM | +0.0093 (0.0037 to 0.0158) * | +0.0038 (−0.0004 to 0.0088) | +0.0316 (0.0115 to 0.0554) * | +0.0116 (−0.0076 to 0.0386) |
| Neural network − SES | +0.0076 (−0.0012 to 0.0183) | +0.0029 (−0.0019 to 0.0080) | −0.0211 (−0.0823 to 0.0296) | +0.0150 (−0.0042 to 0.0391) |
| 50/50 average − SES | +0.0010 (−0.0058 to 0.0090) | −0.0003 (−0.0044 to 0.0038) | −0.0415 (−0.1050 to 0.0090) | +0.0056 (−0.0077 to 0.0206) |
| SES − moving average | −0.0093 (−0.0154 to −0.0036) * | −0.0131 (−0.0195 to −0.0078) * | +0.0157 (−0.0197 to 0.0619) | −0.0512 (−0.1056 to −0.0084) * |

- **Only 6 of the 20 comparisons exclude zero**, and with 20 intervals about one is expected by chance.
- **SES beats the moving average on RMSSE in both windows.** It is the only difference that holds in both.
- **LightGBM does not reliably beat SES:** its MAE advantage on validation (−0.053) disappears on test. In addition, its configuration was chosen among six on that same window, so the validation result is somewhat optimistic.
- **The neural network is worse than LightGBM** on validation and has the same sign on test, although there the intervals include zero. It does not beat SES either. Its RMSSE varies by up to 0.009 depending on the seed, more than the difference between LightGBM and SES.
- **The MAE intervals are much wider than the RMSSE ones:** MAE is in units, and a few high-volume series dominate it.
- **Forecasting zero is the worst model on MAE,** even though 60% of the series have a median of zero over the last 56 days. The 45 erratic and smooth series account for about 31% of its absolute error.

**Overall reading (unverified hypothesis):** at the product, store and day level, demand is so sporadic and noisy that almost everything predictable lies in the recent level of sales. Additional features and more flexible models do not produce a reliable improvement in these windows.

These results are not comparable with those of the competition: here 600 series from the most disaggregated level are evaluated, on a single window, with unweighted RMSSE and with far less feature engineering and tuning. In the competition, the best results were obtained with machine learning (the winner combined several LightGBM models).

## Limitations

- A single window per stage: the bootstrap measures variability across series, but not across time windows. There is no multiple-comparison correction.
- Only 600 of the 30,490 series; a validation window without events and a test window with minor events.
- LightGBM and the network were tuned only to a limited extent (for the network, the architecture, learning rate and regularization were not tuned; only two losses and 3 seeds were tried). The network is a multilayer perceptron, so the results do not extend to sequential architectures such as LSTMs or Transformers.
- RMSSE unweighted by revenue, unlike the competition's WRMSSE.
- The exploration statistics use all days, including the evaluation windows; they are not used to tune any model, and the models only see data before each origin.
- The event, SNAP and price tables are descriptive and do not establish causal relationships.

## Future work

- Evaluate with several consecutive origins to estimate variability across windows.
- Use all 30,490 series and evaluate WRMSSE.
- Tune hyperparameters more thoroughly, try sequential architectures and add features aggregated by store or product.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_exploracion.ipynb
│   ├── 02_baselines_estadistico.ipynb
│   ├── 03_lightgbm.ipynb
│   └── 04_red_neuronal_comparacion.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py            # M5 data loading and series sampling
│   ├── metrics.py         # RMSSE and MAE
│   ├── forecasting.py     # baselines, statistical model and demand classification
│   ├── gbm.py             # LightGBM features and training
│   ├── nn.py              # preprocessing and neural network (Keras)
│   └── comparacion.py     # per-series metrics and paired bootstrap
├── reports/               # result tables (.csv) and predictions (.npz)
├── models/                # trained models (not versioned)
└── data/                  # downloaded data (not versioned)
```

## How to reproduce

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Get the M5 data from the [Zenodo page](https://zenodo.org/records/12636070) and put the zip (or the CSVs) in `data/raw/`. `src/data.py` unzips it and tries to download it if it is missing; if the automatic download fails, just save the zip by hand in that folder. `calendar.csv`, `sell_prices.csv` and `sales_train_evaluation.csv` are needed.
3. Run the notebooks in order from `notebooks/`. Notebook 03 uses the results of 02, and 04 uses the predictions saved by 03. On a desktop CPU, the models in notebooks 02 and 03 take seconds per configuration and the training of the networks in 04 takes a few minutes.

Developed with Python 3.13. Seeds are fixed, but network training may vary slightly between runs and machines.

## References and licenses

- Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. *International Journal of Forecasting*, 38(4), 1346-1364. https://doi.org/10.1016/j.ijforecast.2021.11.013
- Syntetos, A. A., Boylan, J. E. and Croston, J. D. (2005). On the categorization of demand patterns. *Journal of the Operational Research Society*, 56(5), 495-503. https://doi.org/10.1057/palgrave.jors.2601841
- Hyndman, R. J. and Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679-688.
- Data: M5 Forecasting - Accuracy (copy on Zenodo, record 12636070, CC BY 4.0 license). The data is not included in this repository.
