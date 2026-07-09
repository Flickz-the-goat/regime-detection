import yfinance as yf
import pandas as pd

def get_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, group_by="ticker")[ticker]

    data_df = pd.DataFrame(data=data, index=data.index)

    data_df.to_csv(f"app/data/raw/{ticker}_raw.csv")

    return data

def main():
    ticker = "^GSPC"
    start_date = "2016-01-01"
    end_date = "2026-01-01"
    data = get_data(ticker, start_date, end_date)

    print("Successfully got data: \n", data[:5])

if __name__ == "__main__":
    main()

