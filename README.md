# BTC Price Prediction with Chronos-2 (Zero-Shot)

This repository contains a script to predict the price of Bitcoin (BTC) using Amazon's Chronos-2 time series foundation model in a zero-shot manner.

## Setup

1.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    Note: `chronos-forecasting` requires Python 3.10+.

## Usage

1.  Run the prediction script:
    ```bash
    python predict_btc.py
    ```

## Description

The script `predict_btc.py` performs the following steps:
1.  Fetches historical BTC-USD daily price data for the last year using `yfinance`.
2.  Loads the pre-trained `amazon/chronos-2` model from Hugging Face.
3.  Prepares the data for zero-shot forecasting.
4.  Predicts the next 30 days of BTC prices.
5.  Outputs the forecast quantiles (10%, 50%, 90%) to the console.
6.  Generates a plot `btc_prediction.png` visualizing the historical data and the forecast with confidence intervals.

## Results

The model generates a probabilistic forecast. The script outputs the median forecast (0.5 quantile) along with the 10% and 90% quantiles, representing the uncertainty.
