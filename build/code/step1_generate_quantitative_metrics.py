import polars as pl
from pathlib import Path
from config import DATA_RAW, DATA_TEMP


sp500 = pl.read_csv(DATA_RAW / "sp500_fundamental_quarterly.csv")
sp500.schema

# rename columns for mapping

FIELD_MAP = {
    "ticker": "tic",
    "capital_expenditure": "capxy",
    "depreciation_and_amortization": "dpq",  # Depreciation & Amortization – Total
    "net_income": "niq",  # Net Income (Loss)
    "outstanding_shares": "cshoq",  # Common Shares Outstanding – Total
    "total_assets": "atq",  # Assets – Total
    "total_liabilities": "ltq",  # Liabilities – Total
    "shareholders_equity": "teqq",  # Stockholders Equity – Total
    "dividends_and_other_cash_distributions": "dvpy",  # Cash Dividends Declared on Common Stock
    "issuance_or_purchase_of_equity_shares": "prstkcy",  # Purchase of Common & Preferred Stock
    "revenue": "revtq",  # Revenue (Net Sales)
    "cost_of_goods": "cogsq",  # Cost of Goods Sold
    "operating_net_cash_flow": "oancfy",  # Operating net cash flow
    "quarter": "fqtr",  # Fiscal Quarter
    "year": "fyearq",  # Fiscal Year
    "price_close": "prccq",  # Price Close
    "price_high": "prchq",  # Price High
    "price_low": "prclq",  # Price Low
}

REVERSED_FIELD_MAP = {comp: py for py, comp in FIELD_MAP.items()}
sp500 = sp500.rename(REVERSED_FIELD_MAP)


#
sp500 = (
    sp500.drop_nulls(
        subset=[
            "price_close",
            "price_high",
            "price_low",
            "total_assets",
            "total_liabilities",
        ]
    )
    .with_columns(
        [
            (pl.col("revenue") - pl.col("cost_of_goods")).alias("gross_profit"),
            (pl.col("revenue") / pl.col("cost_of_goods")).alias("gross_margin"),
            (pl.col("operating_net_cash_flow") - pl.col("capital_expenditure")).alias(
                "free_cash_flow"
            ),
            ((pl.col("price_close") + pl.col("price_low")) / 2).alias("price"),
            (pl.col("total_assets") - pl.col("total_liabilities")).alias(
                "shareholders_equity"
            ),
        ]
    )
    .with_columns(
        [
            (pl.col("price") * pl.col("outstanding_shares")).alias("market_capital"),
        ]
    )
)
output_path = (
    DATA_TEMP / "sp500_fundamental_quarterly_quantitative_metrics.parquet.gzip"
)
sp500.write_parquet(
    output_path,
    compression="gzip",
)

# check the AAPL market capital
# the unit of outstanding shares is 1m
# sp500_aa = sp500.filter(pl.col("ticker") == "AAPL")[
#     [
#         "market_capital",
#         "price",
#         "outstanding_shares",
#         "year",
#         "quarter",
#         "ticker",
#     ]
# ]

# sp500_aa.write_csv("sp500_aa.csv")
