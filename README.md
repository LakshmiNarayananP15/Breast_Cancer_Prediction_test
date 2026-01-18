# Breast Cancer Prediction Project 🎗️

**Author:** Lakshmi Narayanan P

## 📌 Project Overview
This project aims to develop and evaluate machine learning models to accurately classify breast tumors as either **Malignant (M)** or **Benign (B)**. Using a dataset of tumor measurements, the project implements a complete machine learning pipeline—from data cleaning and visualization to model training and feature selection—to identify the most effective classification strategy.

## 📂 Dataset
The project utilizes the `Cancer_Data.csv` dataset.
* **Target Variable:** `diagnosis` (M = Malignant, B = Benign)
* **Features:** 30+ numerical features describing tumor characteristics, including:
  * `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`
  * `smoothness_mean`, `compactness_mean`, `concavity_mean`, etc.
* **Data Cleaning:** The dataset contained an unnecessary `Unnamed: 32` column and an `id` column, both of which were removed during preprocessing.

## 🛠️ Technologies & Libraries Used
The project is implemented in Python using the following libraries:
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn` (Model selection, Preprocessing, Linear Models, Metrics)

## ⚙️ Methodology

### 1. Data Exploration & Visualization
* Analyzed the distribution of the target variable (`diagnosis`).
* Generated count plots to visualize the balance between malignant and benign cases.
* Created pair plots to explore relationships between features like `radius_mean`, `texture_mean`, and `perimeter_mean`.

### 2. Data Preprocessing
* **Encoding:** Converted the `diagnosis` column into numerical values (Malignant = 1, Benign = 0).
* **Splitting:** Divided data into training (80%) and testing (20%) sets.
* **Scaling:** Applied `StandardScaler` to normalize features for better model performance.

### 3. Model Training
Three machine learning models were trained and evaluated:
1. **Logistic Regression**
2. **Support Vector Classifier (SVC)**
3. **Random Forest Classifier**

## 📊 Results

The models were evaluated based on accuracy, with a comparison between using all original features vs. a selected subset of highly correlated features (>70% correlation).

| Model | Accuracy (Original Features) | Accuracy (Selected Features) |
| :--- | :--- | :--- |
| **Logistic Regression** | 97.37% | **99.12%** |
| **Support Vector Classifier (SVC)** | **98.25%** | 94.74% |
| **Random Forest Classifier** | 96.49% | 95.61% |

### 🔑 Key Findings
* **Best Performance:** The **Logistic Regression** model, when combined with feature selection, achieved the highest accuracy of **99.12%**, with only one misclassification on the test set.
* **Feature Importance:** Feature selection proved highly effective for Logistic Regression but caused a slight performance drop for SVC and Random Forest, highlighting that feature engineering impact is model-dependent.

## 🚀 How to Run
1. Clone the repository.
2. Ensure `Cancer_Data.csv` is in the project directory.
3. Install dependencies:
      pip install pandas numpy matplotlib seaborn scikit-learn


### 4. Run the Jupyter Notebook:

jupyter notebook Breast_Cancer_Prediction_test.ipynb
