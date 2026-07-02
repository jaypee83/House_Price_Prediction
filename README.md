# 🏠 House Price Prediction Using Machine Learning

## 📌 Project Overview

This project develops a machine learning model to predict house prices based on property features such as location, area, construction status, RERA approval, number of bedrooms, and other property characteristics.

The goal is to help buyers and real estate professionals estimate property prices using historical housing data.

---

# 🎯 Problem Statement

The objective of this project is to build a regression-based machine learning system that accurately predicts:

**TARGET(PRICE_IN_LACS)**

using available house attributes.

The project includes:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Machine learning model training
- Model comparison
- Hyperparameter tuning
- Model deployment using Streamlit

---

# 📂 Dataset Description

Dataset: House Price Dataset

Number of records:

- Original records: 29,451
- After removing duplicates: 29,050

Target variable:

The target represents house price in lakhs.

---

# 📊 Features Description

| Feature | Description |
|---|---|
| POSTED_BY | Person posting the property |
| UNDER_CONSTRUCTION | Whether property is under construction |
| RERA | RERA approval status |
| BHK_NO. | Number of bedrooms |
| BHK_OR_RK | Property type |
| SQUARE_FT | Area of property |
| READY_TO_MOVE | Ready for occupancy |
| RESALE | Resale property status |
| ADDRESS | Property location |
| LONGITUDE | Longitude coordinate |
| LATITUDE | Latitude coordinate |

---

# 🔎 Exploratory Data Analysis

Performed:

- Dataset information analysis
- Missing value checking
- Duplicate detection
- Statistical analysis
- Feature type analysis

Findings:

- No missing values were found
- 401 duplicate records were removed

---

# 🛠️ Data Preprocessing

Steps performed:

- Removed duplicate rows
- Separated features and target variable
- Handled numerical features
- Encoded categorical features using One-Hot Encoding
- Created preprocessing pipeline
- Split dataset into training and testing sets

Training data: 80%

Testing data: 20%

---

# 🤖 Machine Learning Models Used

Three regression models were trained:

## 1. Linear Regression

A baseline regression model used for comparison.

## 2. Random Forest Regressor

An ensemble learning model that handles nonlinear relationships.

## 3. XGBoost Regressor

A gradient boosting model designed for high predictive performance.

---

# 📈 Model Performance Comparison

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | 138.84 | 631.35 | 0.272 |
| Random Forest | 40.47 | 235.69 | 0.898 |
| XGBoost | 45.79 | 408.22 | 0.695 |

---

# 🏆 Best Model

The best performing model:
Random Forest Regressor


Performance:
R² Score ≈ 0.90

The model was optimized using GridSearchCV.

Best parameters:
n_estimators = 100
max_depth = None

Saved model:
models/best_house_price_model.pkl

---

# 🚀 Deployment

A Streamlit web application was created.

Users can enter:

- Location
- Property size
- BHK number
- Construction status
- RERA status
- Other features

The application predicts:
Estimated House Price in Lakhs

House_Price_Prediction/

│
├── data/
│ └── House Price.csv
│
├── models/
│ └── best_model.pkl
│
├── src/
│ ├── data_preprocessing.py
│ ├── train.py
│ ├── tuning.py
│ └── predict.py
│
├── app.py
│
├── requirements.txt
│
└── README.md

---

# ⚙️ Installation and Setup

Clone the repository:

```bash
git clone <repository-url>

Navigate to project folder:
cd House_Price_Prediction
Create virtual environment:
python -m venv myenv
myenv\Scripts\activate

Running the Project

Train models:

python src/train.py

Tune best model:

python src/tuning.py

Run prediction:

python src/predict.py

Launch web application:

streamlit run app.py

Technologies Used
Python
Pandas
NumPy
Scikit-learn
XGBoost
Matplotlib
Seaborn
Streamlit
Joblib

Author: Prince Thomas
