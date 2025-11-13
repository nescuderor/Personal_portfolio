# Data Science Portfolio
**Nicolás Escudero Rivera**

Welcome to my data science portfolio! This repository showcases my expertise in Python programming, machine learning, and data analysis through a collection of well-documented projects and implementations.

## 📋 Table of Contents
- [About](#about)
- [Technical Skills Demonstrated](#technical-skills-demonstrated)
- [Projects Overview](#projects-overview)
  - [01: Python Fundamentals & Data Analysis](#01-python-fundamentals--data-analysis)
  - [02: Machine Learning Models](#02-machine-learning-models)
  - [03: Time Series Analysis & Forecasting](#03-time-series-analysis--forecasting)
- [Technologies Used](#technologies-used)
- [Contact](#contact)

## 🎯 About

This portfolio demonstrates my proficiency in data science through hands-on implementations of fundamental concepts, from basic Python programming to advanced machine learning algorithms. Each project is thoroughly documented with detailed explanations, visualizations, and best practices.

## 💻 Technical Skills Demonstrated

- **Core Python**: Data structures, OOP, control flow, functions, comprehensions
- **Scientific Computing**: NumPy for vectorized operations and linear algebra
- **Data Manipulation**: Pandas for data transformation, aggregation, and relational operations
- **Machine Learning**: Regression models, regularization, feature engineering, hyperparameter tuning
- **Time Series Analysis**: Stationarity testing, ARMA/ARIMA models, financial data preprocessing
- **Advanced Forecasting**: ML-based forecasting, recursive prediction, backtesting frameworks
- **Data Visualization**: Matplotlib and Seaborn for exploratory data analysis and results presentation
- **Mathematical Foundations**: Linear algebra, calculus, and statistics applied to ML algorithms
- **Software Engineering**: Clean code, comprehensive documentation, modular design

## 📂 Projects Overview

### 01: Python Fundamentals & Data Analysis

A comprehensive demonstration of Python programming fundamentals and data analysis capabilities, organized across multiple Jupyter notebooks.

#### **01_Basics.ipynb** - Python Core Concepts
Comprehensive exploration of Python's fundamental features through practical examples:

- **Data Types & Structures**: Deep dive into strings, tuples, lists, sets, and dictionaries with practical manipulation examples
- **Control Flow**: Complex pattern generation (X-shaped squares, Pascal's Triangle) demonstrating mastery of nested loops and conditionals
- **Functions**: Lambda functions, type hints, docstrings, and error handling demonstrated through a BMI classification system
- **Object-Oriented Programming**:
  - Class design with constructors, methods, and attributes
  - Inheritance and method overriding
  - Dunder methods (`__repr__`, `__str__`, `__eq__`)
  - Class methods and factory patterns
  - Practical customer/store management system
- **Advanced Project - Custom Calculator**:
  - Built a from-scratch arithmetic calculator implementing PEMDAS order of operations
  - Uses algorithmic thinking with moving window pattern and priority-based evaluation
  - Includes tokenization, error handling, and operator precedence management
  - Demonstrates understanding of parsing and expression evaluation

#### **02_Numpy.ipynb** - Numerical Computing
Scientific computing and linear algebra applications:

- **Vectorized Operations**: Efficient array manipulations and transformations
- **Matrix Operations**: Dot products, matrix multiplication, transposition, determinants, and inverse matrices
- **Practical Application**: Manual implementation of Ordinary Least Squares (OLS) regression using pure linear algebra
- **Statistical Analysis**: Descriptive statistics, correlation analysis, and distribution analysis
- **Data Simulation**: Random data generation and statistical property verification

#### **03_Pandas.ipynb** - Data Analysis & Manipulation
Professional-grade data manipulation and exploratory data analysis:

- **Data Loading**: Reading from multiple formats (CSV, pickle)
- **Exploratory Data Analysis**:
  - Iris dataset: Statistical summaries, group-by operations, distribution analysis
  - Chicago business licenses: Multi-table joins, relational database operations
- **Data Transformation**: Aggregations, pivot tables, merging, and filtering
- **Real-World Analysis**:
  - Socioeconomic analysis by geographic region
  - Business license trends across political administrations
  - Income distribution and demographic insights
- **Data Quality**: Missing value detection, duplicate handling, outlier identification

**Key Datasets Used**: Iris flower dataset, Chicago business licenses, ward demographics, census data

---

### 02: Machine Learning Models

A collection of machine learning implementations demonstrating both theoretical understanding and practical application, from manual implementations to scikit-learn pipelines.

#### **01_manual_linear_regression.py** - Mathematical Foundations
Pure implementation of linear regression using the normal equation:

- **Mathematical Rigor**: Direct implementation of $(X^T X)^{-1} X^T y$ for computing optimal weights
- **Matrix Operations**: Leverages NumPy's linear algebra capabilities for efficient computation
- **Model Evaluation**: RMSE calculation on held-out test set
- **Purpose**: Demonstrates deep understanding of the mathematical foundations underlying ML algorithms

**Key Concept**: This implementation proves mastery of the theory before relying on libraries.

#### **02_linear_regression.py** - Polynomial Feature Engineering
Exploration of model complexity and the bias-variance tradeoff:

- **Incremental Feature Creation**: Progressively adds polynomial features (x, x², x³, ...)
- **Model Complexity Analysis**: Compares performance across polynomial orders 1-3
- **Visualization Strategy**:
  - Subplot grid system showing training vs. test data
  - Model fit visualization with prediction curves
  - Progressive overfitting demonstration
- **Advanced Plotting**: Object-oriented Matplotlib with dynamic subplot management
- **Practical Insight**: Demonstrates when increasing model complexity helps vs. hurts generalization

**Visual Output**: Multi-panel comparison showing how polynomial order affects model fit.

#### **03_feature_engineering.py** - Automated ML Pipeline
Production-ready feature engineering using scikit-learn pipelines:

- **Dynamic Feature Detection**: Programmatically distinguishes continuous vs. categorical columns
- **Pipeline Architecture**:
  - `ColumnTransformer` for column-specific preprocessing
  - `OneHotEncoder` for categorical features
  - `StandardScaler` for continuous features
  - `PolynomialFeatures` for interaction terms
- **Reusability**: Encapsulated in a function for easy deployment
- **Best Practices**: Demonstrates modern ML workflow patterns used in production systems

**Key Strength**: Shows ability to build scalable, maintainable ML preprocessing pipelines.

#### **04_linear_regression_l2.py** - Hyperparameter Optimization
Systematic approach to Ridge Regression regularization:

- **Regularization Theory**: Implementation of L2 penalty to prevent overfitting
- **Grid Search**: Manual hyperparameter tuning across 500 lambda values
- **Smart Search Space**: Logarithmically-spaced values using `np.geomspace`
- **Performance Visualization**:
  - Log-scale plot of RMSE vs. regularization strength
  - Clear identification of optimal lambda
- **Model Selection**: Demonstrates principled approach to finding the bias-variance sweet spot

**Insight Demonstrated**: Understanding that model complexity must be balanced with generalization.

#### **05_linear_regression_sgd.py** - Mini-Batch Stochastic Gradient Descent
Implementation of linear regression using mini-batch SGD:

- **SGD Optimization from Scratch**: Custom implementation of the stochastic gradient descent algorithm for linear regression, without relying on external ML libraries.
- **Mini-Batch Processing**: Efficient training using randomly sampled mini-batches, enabling scalable learning on larger datasets.
- **Weight Update Mechanism**: Iterative parameter updates based on mini-batch gradients, demonstrating understanding of optimization dynamics.
- **L2 Regularization Integration**: Incorporates Ridge-style penalty to control model complexity and prevent overfitting.
- **Learning Curve Visualization**: Plots training and validation RMSE over epochs to illustrate convergence and generalization.

**Key Strength**: Shows ability to implement core ML optimization techniques and analyze their performance.

---

### 03: Time Series Analysis & Forecasting

Advanced time series modeling combining statistical methods with machine learning techniques for financial forecasting applications.

#### **01_arma_models.ipynb** - Statistical Time Series Modeling
Comprehensive ARMA analysis of Apple stock returns:

- **Data Preparation**:
  - Financial data acquisition via yfinance
  - Business day frequency handling and holiday treatment
  - Log returns transformation for stationarity
- **Stationarity Analysis**:
  - Augmented Dickey-Fuller (ADF) test implementation
  - KPSS test for trend stationarity
  - Jarque-Bera normality testing
  - ACF/PACF visualization and interpretation
- **Model Selection**:
  - Grid search across AR and MA orders using scipy.optimize.brute
  - Information criteria comparison (AIC, BIC, HQIC)
  - Optimal ARMA specification selection
- **Forecasting & Validation**:
  - Rolling window forecasting
  - Out-of-sample performance evaluation
  - Residual diagnostics (Ljung-Box test, normality checks)
- **Statistical Rigor**: Demonstrates understanding of time series assumptions and diagnostic testing

**Key Concepts**: Stationarity, autocorrelation structure, model parsimony, residual analysis

#### **02_decision_tree.ipynb** - ML-Based Financial Forecasting
Machine learning approach to time series forecasting with Google stock data:

- **Advanced Data Engineering**:
  - Business day indexing with proper holiday handling (forward-fill methodology)
  - Economic rationale for treating market closures
  - Comprehensive missing data analysis
- **Statistical Foundation**:
  - Multi-test stationarity verification (ADF + KPSS)
  - Distribution analysis and normality assessment
  - Autocorrelation significance testing (Ljung-Box)
- **Dual Modeling Approach**:
  - **Statistical Baseline**: ARMA(0,0,1) model for benchmark comparison
  - **ML Forecaster**: Decision Tree Regressor with recursive prediction
- **Hyperparameter Optimization**:
  - Systematic grid search across lag structures (1-30 lags)
  - Tree complexity tuning (criterion, max_depth)
  - Time series cross-validation with expanding windows
  - Performance-driven parameter selection
- **Rigorous Validation**:
  - 80-20 train-test split with temporal ordering
  - Rolling window backtesting on last 3 months
  - Refit strategy for adapting to recent data
  - RMSE comparison between statistical and ML approaches
- **Feature Analysis**:
  - Feature importance identification
  - Optimal lag structure interpretation
  - Economic insight into predictive patterns
- **Production-Ready Implementation**:
  - skforecast library integration
  - Backtesting framework for realistic performance assessment
  - Clear visualization of predictions vs. actuals

**Key Strengths**:
- Bridges statistical and ML forecasting paradigms
- Demonstrates understanding of financial time series properties
- Rigorous out-of-sample validation methodology
- Economic reasoning combined with technical implementation

**Practical Insights**: Shows ability to evaluate when ML methods outperform traditional statistical approaches and how to properly validate time series models.

---

## 🛠️ Technologies Used

### Core Libraries
- **Python 3.x**: Primary programming language
- **NumPy**: Numerical computing and linear algebra
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization and plotting
- **Scikit-learn**: Machine learning algorithms and utilities
- **SciPy**: Statistical functions and advanced mathematics
- **Statsmodels**: Time series analysis and statistical modeling
- **skforecast**: Specialized forecasting library with backtesting capabilities
- **yfinance**: Financial data retrieval

### Development Tools
- **Jupyter Notebooks**: Interactive development and documentation
- **Git**: Version control
- **Virtual Environments**: Dependency management

### Skills Demonstrated
- Mathematical modeling and algorithm implementation
- Statistical analysis and hypothesis testing
- Time series analysis and stationarity testing
- Financial data preprocessing and transformation
- Data preprocessing and feature engineering
- Model evaluation and validation
- Backtesting and cross-validation strategies
- Code documentation and software engineering best practices

---

## 📊 Highlights

### What Sets This Portfolio Apart

1. **Depth of Understanding**: Not just using libraries—building algorithms from scratch to demonstrate mathematical foundations
2. **Comprehensive Documentation**: Every script and notebook includes detailed explanations, docstrings, and inline comments
3. **Best Practices**: Clean code, modular design, type hints, and proper error handling throughout
4. **Progression**: Clear learning path from basics to advanced concepts
5. **Practical Applications**: Real-world datasets and problems, not just toy examples
6. **Visualization**: Effective use of plots to communicate insights and model behavior

### Project Complexity Indicators

- ✅ Manual implementation of OLS regression using pure linear algebra
- ✅ Custom calculator with expression parsing and operator precedence
- ✅ Automated feature engineering pipelines with dynamic type detection
- ✅ Hyperparameter optimization with systematic grid search
- ✅ Multi-table relational data analysis
- ✅ Object-oriented design patterns in Python
- ✅ Time series stationarity testing and transformation
- ✅ ARMA model selection with information criteria optimization
- ✅ ML-based forecasting with recursive prediction
- ✅ Rolling window backtesting for financial time series

---

## 📫 Contact

**Nicolás Escudero Rivera**

Feel free to explore the code, and don't hesitate to reach out with questions or opportunities!

---

## 📝 Repository Structure

```
Personal_portfolio/
├── 01_Basics/              # Python fundamentals and data analysis
│   ├── 01_Basics.ipynb     # Core Python concepts and OOP
│   ├── 02_Numpy.ipynb      # Numerical computing
│   ├── 03_Pandas.ipynb     # Data manipulation
│   └── requirements.txt    # Dependencies
│
├── 02_ML_models/           # Machine learning implementations
│   ├── 01_manual_linear_regression.py    # OLS from scratch
│   ├── 02_linear_regression.py           # Polynomial regression
│   ├── 03_feature_engineering.py         # ML pipelines
│   ├── 04_linear_regression_l2.py        # Ridge regression
│   ├── 05_linear_regression_sgd.py       # Linear regression with SGD
│   └── requirements.txt                  # Dependencies
│
├── 03_Time_series/         # Time series analysis and forecasting
│   ├── 01_arma_models.ipynb              # Statistical ARMA modeling
│   ├── 02_decision_tree.ipynb            # ML-based forecasting
│   └── requirements.txt                  # Dependencies
│
└── README.md               # This file
```

---

*This portfolio is continuously updated as I expand my data science skills and tackle new challenges.*
