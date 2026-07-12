import pandas as pd 
import numpy as np

def build_features(ticker):
    raw_data = pd.read_csv(f"app/data/raw/{ticker}_raw.csv", index_col=0)
    feature_df = pd.DataFrame(index=raw_data.index)

    # Calc log return
    raw_data["log_return"] = np.log(raw_data["Close"] / raw_data["Close"].shift(1))

    # Trend
    feature_df["trend_20d"] = np.log(raw_data["Close"] / raw_data["Close"].shift(20))

    sma_20 = raw_data["Close"].rolling(20).mean()

    feature_df["price_vs_sma20d"] = (raw_data["Close"] - sma_20) / sma_20

    # Mean-reversion 

    std_dev = raw_data["Close"].rolling(20).std()
    feature_df["mean_reversion_20d"] = (raw_data["Close"] - sma_20)/std_dev


    # Volatility 
    volatility = raw_data["log_return"].rolling(10).std()
    feature_df["volatility_10_annualized"] = volatility * np.sqrt(252)

    # Momentum 
    feature_df["momentum_12_1"] = np.log(raw_data["Close"].shift(21) / raw_data["Close"].shift(252))


    # Liquidity 
    dollar_volume = raw_data["Close"] * raw_data["Volume"]
    dollar_volume = dollar_volume.rolling(20).mean()
    amihud_daily = np.abs(raw_data["log_return"]) / ((dollar_volume / 1e6))
    feature_df["amihud_20d"] = np.log(amihud_daily.rolling(20).mean())

    print(feature_df.isna().sum())
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).dropna()
    feature_df["Close"] = raw_data["Close"]

    feature_df.to_csv(f"app/data/features/{ticker}_features.csv")
    return feature_df


def main():
    ticker = "^GSPC"

    feature_df = build_features(ticker)

if __name__ == "__main__":
    main()