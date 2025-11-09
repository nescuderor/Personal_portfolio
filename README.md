# Data Science Portfolio
**Nicolás Escudero Rivera**

Welcome to my data science portfolio! This repository showcases my expertise in Python programming, machine learning, and data analysis through a collection of well-documented projects and implementations.

## 📋 Table of Contents
- [About](#about)
- [Technical Skills Demonstrated](#technical-skills-demonstrated)
- [Projects Overview](#projects-overview)
  - [01: Python Fundamentals & Data Analysis](#01-python-fundamentals--data-analysis)
  - [02: Machine Learning Models](#02-machine-learning-models)
- [Technologies Used](#technologies-used)
- [Contact](#contact)

## 🎯 About

This portfolio demonstrates my proficiency in data science through hands-on implementations of fundamental concepts, from basic Python programming to advanced machine learning algorithms. Each project is thoroughly documented with detailed explanations, visualizations, and best practices.

## 💻 Technical Skills Demonstrated

- **Core Python**: Data structures, OOP, control flow, functions, comprehensions
- **Scientific Computing**: NumPy for vectorized operations and linear algebra
- **Data Manipulation**: Pandas for data transformation, aggregation, and relational operations
- **Machine Learning**: Regression models, regularization, feature engineering, hyperparameter tuning
- **Data Visualization**: Matplotlib for exploratory data analysis and results presentation
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

---

## 🛠️ Technologies Used

### Core Libraries
- **Python 3.x**: Primary programming language
- **NumPy**: Numerical computing and linear algebra
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization and plotting
- **Scikit-learn**: Machine learning algorithms and utilities
- **SciPy**: Statistical functions and advanced mathematics

### Development Tools
- **Jupyter Notebooks**: Interactive development and documentation
- **Git**: Version control
- **Virtual Environments**: Dependency management

### Skills Demonstrated
- Mathematical modeling and algorithm implementation
- Statistical analysis and hypothesis testing
- Data preprocessing and feature engineering
- Model evaluation and validation
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
│   ├── 05_linear_regression_sgd.py        # Linear regression with SGD
│   └── requirements.txt                  # Dependencies
│
└── README.md               # This file
```

---

*This portfolio is continuously updated as I expand my data science skills and tackle new challenges.*
