# <center><font size=6>Empirical Research Template</font></center>
<p align=right> <font size=2>Yiyi Wang<br>20241011</font></p>
_________________________________________________________________






The main purpose is to provide a useful data folder structure for version control across time and between local laptop and high performance cluster (HPC). This folder structure follows [Guide written by Matthew Gentzkow and Jesse M. Shapiro](https://web.stanford.edu/~gentzkow/research/CodeAndData.pdf) and contains useful `config.do` and `config.py` that set up path and packages in stata and python. 


Section `1. file Tree` presents the file structure. Then in section `2 raw data` , `3 code`will briefly summarize the contents in the folder.  `4. Results` will compare replicated results with original paper's.



## <font size=5>1. file tree</font>

```python



```




## <font size=5>2. raw data</font>

Business-line-level and firm-level data come from the [Insurance Statutory Financials](https://www.capitaliq.spglobal.com/web/client?auth=inherit#office/screener?perspective=287). Macro factors comes from [USA | Economic & Demographic Data](https://www.capitaliq.spglobal.com/web/client?auth=inherit#country/economicDemographic?keycountry=US). All data can be found in Capital IQ Pro. Detailed variable definition can be found in [Variable definitions.xlsx](./Build/data/raw/Variable%20definitions.xlsx).

## <font size=5>3. code</font>

### <font size=5>step_1_clean_and_build_data.py  </font>
I highly recommend to use data from 2000 because of data quality as [corp_missing_value_by_year.xlsx](./Build/data/temp/corp_1_missing_value_by_year.csv) shows. And I find any firm-level ratio suffer the similar problem. The alternative way is to build ratio by ourselves.
input:
```python
├── .git
├── analysis
│   ├── code
│   │   └── config.do
│   └── data
│       ├── input
│       └── output
│           └── config.do
├── build
│   ├── code
│   │   ├── config.do
│   │   ├── config.py
│   │   ├── data_merge_template.py
│   │   ├── toolkit.py
│   │   └── variables_record.py
│   └── data
│       ├── processed
│       ├── raw
│       └── temp
├── README.md
├── README.py
└── resource

```



output:
```python
├── Build
│   └── data
│       └── temp
│           ├── {business_line}_database.csv
│           ├── {business_line}_summary.csv
├── Build
│   └── data
│       ├── output
│       │   ├── table_4_stat_summary.csv
```

`