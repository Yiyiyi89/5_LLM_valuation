# <center><font size=6>Replicate Machine learning improves accounting estimates:evidence from insurance payments</font></center>
<p align=right> <font size=2>Yiyi Wang<br>20241011</font></p>
_________________________________________________________________






The main purpose is to replicate works from [Ding er al(2010)](https://link.springer.com/article/10.1007/s11142-020-09546-9) 


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
├── Build
│   └── data
│       ├── raw
│       │   ├── {business_line}1.xls
│       │   ├── {business_line}2.xls
│       │   ├── {business_line}3.xls
│       │   ├── corp1.xls
│       │   ├── Macro factors.xls
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

`{business_line}1.xls` should **only** contains 'ACTUAL_LOSS' variable to ensure code running.`{business_line}2.xls` and `{business_line}3.xls` contain other operational variables.`corp1.xls` and `Macro factors.xls` are corresponding firm-level,macro-level variables.

![image info](./Resource/table_10_1.png)

{business_line}_summary.csv is stat summary for corresponding businessline. 
### <font size=5>step_2_train_ML_models.py  </font>
**This part should be run in High Performance Cluster(HPC)**

input:
```python
├── Build
│   └── data
│       └── temp
│           ├── {business_line}_database.csv
```

output:
```python
├── Build
│   └── data
│       └── temp
│           ├── {business_line}_models
│           │   ├── {business_line}_with_pred.csv
│           │   ├── grid_{start_year}_{end_year}_{var}_{model_name}.pkl
```
Since the $start\_year$ is always 1996 here, training models will be models in such combinations of parameters of
$$[business\_line]*[end\_year]*[var]*[model\_name]$$
$[var]$ is either'ml_manager' or 'ml', which means whether or not to contains 'MANAGER_ESTIMATION' into the model training

### <font size=5>step_3_get_model_results.py  </font>

input:
```python
├── Build
│   └── data
│       └── temp
│           ├── {business_line}_database.csv
│           ├── {business_line}_models
│           │   ├── grid_{start_year}_{end_year}_{var}_{model_name}.pkl
```

output:
```python
├── Build
│   └── data
│       └── temp
│           ├── lines_database_without_corp2.csv
├── Build
│   └── data
│       └── output
│           ├── table_5_best_cv_results.csv
│           ├── table_6_best_pred_results.csv
│           ├── table_11_cv_results_4_models.csv
│           ├── table_12_pred_results_4_modrls.csv
```

I write a Class `TrainedModel` to extract results of trained models ,which need two input to build `Model=TrainedModel(<filepath>,<dataframe>)`. `<filepath>`  is **a filepath(not pkl file)** for **one** `grid_{start_year}_{end_year}_{var}_{model_name}.pkl`.`<dataframe>` is the database which was used to training models. Here it is `{business_line}_database.csv`

### <font size=5> step_4_var_corp2.py  </font>

input:
```python
├── Build
│   └── data
│       ├── raw
│       │   ├── corp2.xls
```

output:
```python
├── Build
│   └── data
│       └── temp
│           ├── corp2.csv
```
Get other firm-level variables to build manager's incentive. I calculate $ROA_t =\frac{Net\_Income_t}{Total\_Asset_t}$ **rather than** directly use `ROAA`in [Insurance Statutory Financials](https://www.capitaliq.spglobal.com/web/client?auth=inherit#office/screener?perspective=287) because there are too many missing values in `ROAA` their dataset.
![image info](./Resource/table_10_2.png)


### <font size=5> step_5_var_violation.py  </font>

input:
```python
├── Build
│   └── data
│       ├── raw
│       │   ├── IRIS ratios
│       │   │   ├── {year}.xlsm
```

output:
```python
├── Build
│   └── data
│       └── temp
│           ├── violation.csv
```

Get `violation` variables. I gold [IRIS ratios](./Build/data/raw/IRIS%20ratios) by asking online agent in Capital IQ.And I recommend to do so to skip issues about the macro in excel files.

### <font size=5> step_6_merge_data_and_gene_vars.py  </font>

input:
```python
├── Build
│   └── data
│       └── temp
│           ├── corp2.csv
│           ├── lines_database_without_corp2.csv
│           ├── violation.csv
```

output:
```python
├── Build
│   └── data
│       └── temp
│           ├── regression_database.csv
│           ├── table_8_error_analysis.csv
```
Generate final database for regression analysis

### <font size=5> step_7_regression.py  </font>

input:
```python
├── Analysis
│   └── data
│       ├── output
│       └── temp
│           ├── regression_database.csv
```

output:
```python
├── Analysis
│   └── data
│       ├── output
│           ├── regression_results.xlsx
```
regression to follow table 9 in [Ding er al(2010)](https://link.springer.com/article/10.1007/s11142-020-09546-9) 



## <font size=5>4. Results</font>


### <font size=5>4.1 Best models for the prediction </font>

Refer to [table_12_pred_results_4_modrls.xlsx](./Analysis/data/output/table_12_pred_results_4_models.xlsx), **random forest** model perform better most of time. This result is consistent with original paper's result.
![image info](./Resource/table_12_1.png)

And the primary difference between the MAE criteria and the RMSE criteria is : **MAE** is preferable if you have a lot of outliers or if you value consistent performance across all predictions.**RMSE** is preferable if large errors carry more weight and you need to minimize those.

Since I used model with best MAE in this project([step 2](./Build/code/step_2_train_ML_models.py)). We will more focus on MAE score here. But we can always change `refit='MAE'` to`refit='RMSE'`
```python
grid = GridSearchCV(model, params, scoring=scoring, cv=5, refit='MAE')
```


### <font size=5>4.2 Best var for the prediction </font>

Refer to [table_6_pred_results_4_models.xlsx](./Analysis/data/output/table_6_best_pred_results.xlsx). In most business lines, machine learning shows more prediction power than manager estimation except `homeowner/farmerowner` line,which is the same as original paper's result.
![image info](./Resource/table_6.png)

### <font size=5>4.3 Is prediction biased by common earnings management incentives? </font>
Since the cleaning step is different from Ding' work. The `Group` and `SMALL_LOSS` is always zero in our final sample. So I dropped these two variables.

**after controlling for two way fixed effect**
Refer to [regression_results_1.xlsx](./Analysis/data/output/regression_results_1.xlsx). The patterns learned by machine learning model is not influenced by common manager's bias like `taxshield`,`smooth` etc,which is the same as original paper's result.
![image info](./Resource/regression_result.png)

[regression_results_1.xlsx](./Analysis/data/output/regression_results_1.xlsx) includes results for models without two-way fixed effects, with individual fixed effects, with time fixed effects, and with both two-way fixed effects.


<!-- - ***patent count***: the number of patents applied for by a given inventor in a given year (patent application year, patents that are eventually granted)
- ***citations***: the total number of citations obtained on all patents that the inventor applies for in a given year
- ***Citations per patent***: the average number of forward citations (counted until 2022) per patent for all patents that a given inventor applies for in a given year
- ***age***: year - birthyear
- ***tenure***: number of years an inventor has worked at a company
- ***transition***: if an inventor has changed companies in the next year, we define it as 1, otherwise 0
- ***generality***: take the average of all patents' generalities in a given year for an inventor to generate inventor-level generality metrics
	-	Compile a list of 3-digit IPC classes corresponding to all patents in the patent_id list.
	-	Calculate the generality of the patent using the formula to obtain the patent-generality correspondence.
	-	Compile a list of patents at the inventor-year level.
	- Calculate the generality at the inventor-year level based on the patent-generality correspondence
- ***inventor general human capital***: take the average of all cumulative patents' generalities up to a given year for a certain inventor









- ***Productivity (scaled cite) around transition***

<p align="center">
  <img src="stata/output/Fig10_scaled_cite_arount_transition.png" alt="Fig10_scaled_cite_arount_transition.png" width="60%" style="border: 1px solid black"  />
</p>
<p align="center"><strong>Fig 10: Productivity (scaled cite) arount transition</strong></p> -->