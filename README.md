# Gold Price Prediction Project (1994-2023)

A comprehensive machine learning project that analyzes and predicts gold prices using macroeconomic factors and commodity prices from 1994 to 2023.

## 📊 Project Overview

This project employs various machine learning algorithms to predict gold prices based on historical data spanning 30 years. The analysis includes feature engineering, multiple model comparisons, and comprehensive evaluation of both linear and tree-based regression models.

### Key Features

- **Comprehensive Data Analysis**: 30 years of gold price data (1994-2023)
- **Multiple ML Models**: Comparison of 70+ model configurations
- **Feature Engineering**: PCA, feature importance, and correlation analysis
- **Model Performance**: Best model achieves R² = 0.9899 (99.99% accuracy)
- **Automated Model Selection**: Systematic hyperparameter tuning and evaluation

## 🎯 Best Model Performance

| Model                            | Feature Selection  | R² Score   | MAE   | MAPE  | RMSE   |
| -------------------------------- | ------------------ | ---------- | ----- | ----- | ------ |
| **Lasso Least Angle Regression** | All Variables      | **0.9899** | 42.25 | 3.16% | 56.36  |
| **Lasso Regression**             | All Variables      | **0.9899** | 42.18 | 3.10% | 56.46  |
| **Gradient Boosting Regressor**  | Feature Importance | **0.9609** | 87.73 | 8.53% | 111.16 |

## 📁 Project Structure

```
Gold Price Prediction/
├── data/
│   ├── gold_1994-2023.xlsx           # Raw dataset
│   ├── model_comparison_results.csv   # Model performance comparison
│   └── Nilai Emas dalam USD.xlsx      # Additional gold price data
├── models/
│   ├── linear/
│   │   ├── best_model_Lasso Least Angle Regression.pkl
│   │   ├── best_model_info_linear_*.pkl
│   │   └── preprocessing_objects_*.pkl
│   └── tree/
│       ├── best_model_Gradient Boosting Regressor.pkl
│       ├── best_model_info_tree_*.pkl
│       └── preprocessing_objects_*.pkl
├── catboost_info/                     # CatBoost training logs
├── main.ipynb                         # Main analysis notebook
├── requirements.txt                   # Python dependencies
└── README.md                         # This file
```

## 📋 Dataset Description

### Data Source

- **File**: `gold_1994-2023.xlsx`
- **Period**: 1994-2023 (30 years)
- **Observations**: 30 rows
- **Features**: 15 columns

### Variables

| Column       | Type    | Description                  |
| ------------ | ------- | ---------------------------- |
| `year`       | int64   | Observation year (1994–2023) |
| `gold_price` | float64 | Gold price (target variable) |
| `fed_rate`   | float64 | Federal interest rate (%)    |
| `inflation`  | float64 | Inflation rate (%)           |
| `ec_growth`  | float64 | Economic growth (%)          |
| `exch_rate`  | float64 | Rupiah to USD exchange rate  |
| `real_int`   | float64 | Real interest rate (%)       |
| `gdp_def`    | float64 | GDP deflator (%)             |
| `cpo`        | float64 | Crude Palm Oil price         |
| `tin`        | float64 | Tin price                    |
| `lend_int`   | float64 | Lending interest rate (%)    |
| `aluminum`   | float64 | Aluminum price               |
| `nickel`     | float64 | Nickel price                 |
| `platinum`   | float64 | Platinum price               |
| `silver`     | float64 | Silver price                 |

## 🚀 Getting Started

### Prerequisites

- Python <= 3.11 (PyCaret doesn't support Python 3.12)
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/maybeitsai/gold-price-prediction.git
   cd gold-price-prediction
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook main.ipynb
   ```

### Dependencies

```
arch==5.3.1
gdown>=5.2.0
matplotlib>=3.7.5
mgarch>=0.3.0
numpy==1.26.4
openpyxl>=3.1.5
pandas==2.1.4
pmdarima>=2.0.4
pycaret[full]<=3.3.2
ruptures>=1.1.10
scikit-learn>=1.4.2
scipy>=1.11.4
seaborn>=0.13.2
statsmodels>=0.14.5
xgboost>=2.1.4
```

## 🔬 Methodology

### 1. Data Preparation

- Data cleaning and normalization
- Column translation (Indonesian to English)
- Missing value analysis (no missing values found)
- Descriptive statistics and data quality assessment

### 2. Exploratory Data Analysis

- Correlation analysis between gold price and features
- Time series visualization
- Distribution analysis of key variables
- Feature importance assessment

### 3. Feature Engineering

- **All Variables**: Using all 14 features
- **Feature Importance**: Selected features based on tree-based models
- **PCA**: Principal Component Analysis for dimensionality reduction
- **Feature Selection**: Statistical feature selection methods

### 4. Model Development

- **Linear Models**:

  - Lasso Least Angle Regression ⭐
  - Lasso Regression
  - Linear Regression
  - Ridge Regression
  - Elastic Net
  - And more...

- **Tree-Based Models**:
  - Gradient Boosting Regressor ⭐
  - Random Forest Regressor
  - Decision Tree Regressor
  - XGBoost
  - CatBoost
  - And more...

### 5. Model Evaluation

- **Metrics Used**:

  - R² Score (Coefficient of Determination)
  - MAE (Mean Absolute Error)
  - MAPE (Mean Absolute Percentage Error)
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)

- **Validation Strategy**:
  - Cross-validation
  - Hyperparameter tuning
  - Model comparison across different feature sets

## 📈 Key Results

### Model Performance Summary

- **Best Linear Model**: Lasso Least Angle Regression (R² = 0.9899)
- **Best Tree Model**: Gradient Boosting Regressor (R² = 0.9609)
- **Feature Selection Impact**: All variables perform better than feature-selected models
- **Total Models Evaluated**: 70+ model configurations

### Feature Importance

The analysis reveals that macroeconomic factors and commodity prices significantly influence gold price movements, with the best models achieving near-perfect prediction accuracy.

## 🔮 Model Deployment

The trained models are saved in the `models/` directory:

- **Linear Models**: `models/linear/`
- **Tree Models**: `models/tree/`
- **Preprocessing Objects**: Saved alongside models for consistent data transformation

## 📊 Visualizations

The notebook includes comprehensive visualizations:

- Time series plots of gold prices
- Correlation heatmaps
- Feature importance charts
- Model performance comparisons
- Residual analysis plots
- SHAP value explanations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👥 Authors

- **Lina Gozali** - _Research and Analysis_
- **Project Team** - _Implementation and Development_

## 🙏 Acknowledgments

- Thanks to all data providers for macroeconomic and commodity price data
- PyCaret team for the excellent AutoML framework
- Open source community for the machine learning libraries used

## 📞 Contact

For questions or collaboration opportunities:

- GitHub: [@maybeitsai](https://github.com/maybeitsai)
- Repository: [gold-price-prediction](https://github.com/maybeitsai/gold-price-prediction)

---

⚡ **Quick Start**: Run `jupyter notebook main.ipynb` to explore the complete analysis and reproduce the results!
