# <center><font size=6>Empirical Research Template</font></center>
<p align=right> <font size=2>Yiyi Wang<br>20241011</font></p>
___________________________________________________________________________________________________________________________






The main purpose is to provide a useful data folder structure for version control across time and between local laptop and high performance cluster (HPC). This folder structure follows [Guide written by Matthew Gentzkow and Jesse M. Shapiro](https://web.stanford.edu/~gentzkow/research/CodeAndData.pdf) and contains useful `config.do` and `config.py` that set up path and packages in stata and python. 


- Decouple code and data, build and analysis


## <font size=5>file tree</font>

```python
├── analysis
│   ├── code: 
│   └── data
│       ├── input:  	panel for descriptive and regressions, usually only .do file here
│       └── output: 	tables and figures
├── build
│   ├── code  
│   └── data
│       ├── processed	database that usually without any constrain
│       ├── raw:    	only raw data 
│       └── temp:   	temp bin, place merge keys and other stuff
├── README.md
├── README.py			code to generate this file tree (I will )
└── resource			related paper or other material


```




## <font size=5>2. raw data</font>

Business-line-level and firm-level data come from the [Insurance Statutory Financials](https://www.capitaliq.spglobal.com/web/client?auth=inherit#office/screener?perspective=287). Macro factors comes from [USA | Economic & Demographic Data](https://www.capitaliq.spglobal.com/web/client?auth=inherit#country/economicDemographic?keycountry=US). All data can be found in Capital IQ Pro. Detailed variable definition can be found in [Variable definitions.xlsx](./Build/data/raw/Variable%20definitions.xlsx).

## <font size=5>3. code</font>

### <font size=5>step_1_clean_and_build_data.py  </font>

```python


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