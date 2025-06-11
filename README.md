# <center><font size=6>Empirical Research Template</font></center>
<p align=right> <font size=2>Yi Yi<br>20241011</font></p>
___________________________________________________________________________________________________________________________

## <font size=5>Project Description</font>

This is an empirical research project template designed to provide a standardized project structure for:
1. Version control
2. Code synchronization between local machine and HPC
3. Decoupling of code and data
4. Separation of build and analysis phases

The template structure follows the [Guide by Matthew Gentzkow and Jesse M. Shapiro](https://web.stanford.edu/~gentzkow/research/CodeAndData.pdf) and includes useful `config.do` and `config.py` files for setting up paths and packages in Stata and Python.

## <font size=5>Directory Structure</font>

```python
├── analysis          # Analysis phase
│   ├── code         # Analysis code
│   └── data
│       ├── input    # Panel data for descriptive stats and regressions
│       └── output   # Generated tables and figures
├── build            # Data construction phase
│   ├── code        # Data processing code
│   └── data
│       ├── processed # Processed databases
│       ├── raw      # Raw data
│       └── temp     # Temporary files, merge keys, etc.
├── README.md        # Project documentation
├── README.py        # Directory tree generator
└── resource         # Related papers and materials
```

## <font size=5>Usage Guide</font>

### 1. Data Management
- `build/data/raw`: Store raw data (read-only)
- `build/data/processed`: Store processed data
- `build/data/temp`: Store intermediate files
- `analysis/data/input`: Store analysis-ready data
- `analysis/data/output`: Store analysis results

### 2. Code Organization
- `build/code`: Data cleaning and construction code
- `analysis/code`: Analysis code
- Each code file should have clear documentation

### 3. Version Control
- Use `.gitignore` for large data files
- Use `.gitkeep` to maintain empty directories
- Regular code commits

### 4. Configuration
- Use `config.py` for Python paths and packages
- Use `config.do` for Stata paths and packages

## <font size=5>Best Practices</font>

1. Data Security
   - Don't commit sensitive data
   - Use `.gitignore` for large files

2. Code Standards
   - Keep code clean and documented
   - Use meaningful names
   - Add appropriate comments

3. Performance
   - Use chunking for large datasets
   - Choose appropriate data structures

4. Collaboration
   - Regular code sync
   - Keep documentation updated
   - Follow project standards

## <font size=5>Maintenance</font>

- Regular dependency updates
- Documentation maintenance
- Temporary file cleanup
- Data backup

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
