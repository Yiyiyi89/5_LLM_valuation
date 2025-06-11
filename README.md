# <center><font size=6>Empirical Research Template</font></center>
<p align=right> <font size=2>易翊翼<br>20241011</font></p>
___________________________________________________________________________________________________________________________

## <font size=5>项目说明</font>

这是一个用于实证研究的项目模板，主要目的是提供一个标准化的项目结构，以便于：
1. 版本控制
2. 本地电脑和高性能计算集群(HPC)之间的代码同步
3. 代码和数据的解耦
4. 构建(build)和分析(analysis)的分离

该模板结构遵循 [Matthew Gentzkow 和 Jesse M. Shapiro 的指南](https://web.stanford.edu/~gentzkow/research/CodeAndData.pdf)，并包含了实用的 `config.do` 和 `config.py` 文件，用于设置 Stata 和 Python 的路径和包。

## <font size=5>目录结构</font>

```python
├── analysis          # 分析阶段
│   ├── code         # 分析代码
│   └── data
│       ├── input    # 用于描述性统计和回归的面板数据，通常只包含.do文件
│       └── output   # 生成的表格和图形
├── build            # 数据构建阶段
│   ├── code        # 数据处理代码
│   └── data
│       ├── processed # 处理后的数据库，通常没有约束
│       ├── raw      # 原始数据
│       └── temp     # 临时文件，存放合并键值和其他中间文件
├── README.md        # 项目说明文档
├── README.py        # 生成目录树的代码
└── resource         # 相关论文和其他材料
```

## <font size=5>使用说明</font>

### 1. 数据管理
- `build/data/raw`: 存放原始数据，建议只读
- `build/data/processed`: 存放处理后的数据
- `build/data/temp`: 存放中间文件，如合并键值等
- `analysis/data/input`: 存放用于分析的数据
- `analysis/data/output`: 存放分析结果，包括表格和图形

### 2. 代码组织
- `build/code`: 包含数据清洗和构建的代码
- `analysis/code`: 包含分析代码
- 每个代码文件都应该有清晰的注释和文档

### 3. 版本控制
- 使用 `.gitignore` 忽略大型数据文件
- 使用 `.gitkeep` 保持空目录结构
- 定期提交代码更改

### 4. 配置管理
- 使用 `config.py` 管理 Python 路径和包
- 使用 `config.do` 管理 Stata 路径和包

## <font size=5>注意事项</font>

1. 数据安全
   - 不要将敏感数据提交到版本控制系统
   - 使用 `.gitignore` 排除大型数据文件

2. 代码规范
   - 保持代码整洁和文档完整
   - 使用有意义的变量名和函数名
   - 添加适当的注释

3. 性能优化
   - 对于大型数据集，考虑使用分块处理
   - 使用适当的数据结构和算法

4. 协作建议
   - 定期同步代码
   - 保持沟通和文档更新
   - 遵循项目的代码规范

## <font size=5>维护说明</font>

- 定期更新依赖包
- 保持文档的及时更新
- 定期清理临时文件
- 备份重要数据

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