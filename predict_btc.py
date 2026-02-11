import pandas as pd
import yfinance as yf
import torch
from chronos import Chronos2Pipeline
import matplotlib.pyplot as plt

def main():
    # 1. Fetch BTC data
    print("Fetching BTC-USD data...")
    btc = yf.download("BTC-USD", period="1y", interval="1d")

    if btc.empty:
        print("Error: No data fetched.")
        return

    # yfinance returns a MultiIndex columns if not flattened in newer versions,
    # but let's check and just take the 'Close' price.
    # Also handle the case where yfinance might return different structures.
    if isinstance(btc.columns, pd.MultiIndex):
        btc = btc.xs('Close', level=0, axis=1)
        # If it's still a dataframe with tickers as columns, we might need to select the ticker
        if "BTC-USD" in btc.columns:
             btc_close = btc["BTC-USD"]
        else:
             # Just take the first column if unsure
             btc_close = btc.iloc[:, 0]
    elif 'Close' in btc.columns:
        btc_close = btc['Close']
    else:
        print("Error: Could not find 'Close' column.")
        print(btc.columns)
        return

    # Ensure it's a Series with a DatetimeIndex
    btc_close.index = pd.to_datetime(btc_close.index)

    # Drop NaNs
    btc_close = btc_close.dropna()

    print(f"Data fetched: {len(btc_close)} points.")
    print(btc_close.tail())

    # 2. Load Chronos-2 model
    model_name = "amazon/chronos-2"
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device_map}...")

    pipeline = Chronos2Pipeline.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if device_map == "cuda" else torch.float32,
    )

    # 3. Prepare data for prediction
    # Chronos expects a torch tensor or list of tensors for context
    # But Chronos2Pipeline.predict uses tensors.
    # Chronos2Pipeline.predict_df uses pandas dataframe.

    # Let's use predict_df as it handles dates nicely.
    # We need to construct a context dataframe.

    context_df = btc_close.reset_index()
    context_df.columns = ["timestamp", "target"]
    context_df["id"] = "BTC-USD" # Single time series

    prediction_length = 30

    print(f"Predicting next {prediction_length} days...")

    # 4. Run prediction
    # We don't have future covariates, so we just pass context_df
    # We need to specify future_df if we had covariates, but for univariate zero-shot with no covariates
    # we might need to see how predict_df handles it.
    # The example showed:
    # pipeline.predict_df(context_df, future_df=future_df, ...)
    # If we don't have covariates, maybe we can pass a future_df with just timestamps and id?

    last_date = context_df["timestamp"].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_length, freq="D")

    future_df = pd.DataFrame({
        "timestamp": future_dates,
        "id": "BTC-USD"
    })

    # The pipeline might expect 'target' column in future_df if we are evaluating,
    # but here we are predicting.

    forecast_df = pipeline.predict_df(
        context_df,
        future_df=future_df,
        prediction_length=prediction_length,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="id",
        timestamp_column="timestamp",
        target="target"
    )

    print("Prediction complete.")
    print(forecast_df.head())

    # 5. Visualize
    plt.figure(figsize=(10, 6))

    # Plot history (last 60 days for clarity)
    history_plot = context_df.iloc[-60:]
    plt.plot(history_plot["timestamp"], history_plot["target"], label="History", color="black")

    # Plot forecast
    # forecast_df usually has columns like 'id', 'timestamp', 'quantile_0.1', 'quantile_0.5', 'quantile_0.9'
    # Wait, check the output structure.
    # If predict_df returns quantiles, they might be in a specific format.
    # Let's assume the columns are named as such or similar.

    # Actually, let's inspect the columns in the output first in the print above.
    # But for plotting:

    # Check if columns are floats or strings
    q01 = 0.1 if 0.1 in forecast_df.columns else "0.1"
    q05 = 0.5 if 0.5 in forecast_df.columns else "0.5"
    q09 = 0.9 if 0.9 in forecast_df.columns else "0.9"

    # If simple float access fails, iterate to find matching columns
    if q05 not in forecast_df.columns:
        # Check for string representation like '0.1'
        # The print output showed '0.1', '0.5', '0.9' which implies they might be strings or just printed that way.
        # Let's try to find them.
        cols = forecast_df.columns
        if '0.1' in cols: q01 = '0.1'
        if '0.5' in cols: q05 = '0.5'
        if '0.9' in cols: q09 = '0.9'

    if q05 in forecast_df.columns:
        plt.plot(forecast_df["timestamp"], forecast_df[q05], label="Median Forecast", color="blue")
        plt.fill_between(
            forecast_df["timestamp"],
            forecast_df[q01],
            forecast_df[q09],
            color="blue",
            alpha=0.2,
            label="10%-90% Confidence Interval"
        )
    else:
        print("Could not find quantile columns for plotting.")
        print("Columns found:", forecast_df.columns)

    plt.title("BTC-USD Price Prediction using Chronos-2 (Zero-Shot)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)

    output_image = "btc_prediction.png"
    plt.savefig(output_image)
    print(f"Plot saved to {output_image}")

if __name__ == "__main__":
    main()
