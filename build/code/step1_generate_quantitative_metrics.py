import polars as pl
from pathlib import Path
from config import DATA_RAW,DATA_TEMP


sp500 = pl.read_csv(Path(DATA_RAW) / "sp500_fundamental_quarterly.csv")







