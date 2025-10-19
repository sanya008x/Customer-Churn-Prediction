# Customer Churn Prediction for a Telecom Company

## 1. Project Overview

This project focuses on predicting customer churn for a fictional telecom company. Customer churn, the rate at which customers stop doing business with a company, is a critical metric. By predicting which customers are likely to leave, the company can proactively offer them special discounts and incentives to retain them, which is often more cost-effective than acquiring new customers.

This repository contains a Jupyter notebook that walks through the entire machine learning pipeline, from data cleaning and exploratory data analysis to model training, evaluation, and deployment preparation.

## 2. Dataset

The project uses the **"Telco Customer Churn"** dataset, which is a popular dataset for classification tasks. It contains customer account information, demographic data, and services they have signed up for. The target variable is `Churn`, indicating whether the customer left within the last month.

## 3. Project Workflow

The project follows a structured machine learning workflow:

1.  **Data Loading and Cleaning:**
    * Loaded the dataset from `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
    * The `customerID` column was dropped as it's not a predictive feature.
    * Handled missing values in the `TotalCharges` column, which were present as empty spaces for new customers. These were imputed with `0.0`.
    * The `TotalCharges` column was converted from an object to a float data type.

2.  **Exploratory Data Analysis (EDA):**
    * Analyzed the distribution of numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) using histograms and boxplots.
    * Visualized the count of each category for all categorical features using count plots.
    * Created a correlation heatmap to understand the relationships between numerical features. `TotalCharges` and `tenure` showed a strong positive correlation.

3.  **Data Preprocessing & Feature Engineering:**
    * The target variable `Churn` was label encoded (`Yes` -> `1`, `No` -> `0`).
    * All categorical features were converted to numerical format using `LabelEncoder`. The encoders for each column were saved to a `encoders.pkl` file for use in the predictive system.

4.  **Handling Class Imbalance:**
    * The target variable `Churn` was found to be imbalanced (more 'No Churn' than 'Churn').
    * **SMOTE (Synthetic Minority Oversampling Technique)** was applied to the training data to create synthetic samples of the minority class, resulting in a balanced dataset for model training.

5.  **Model Training and Selection:**
    * The data was split into training (80%) and testing (20%) sets.
    * Three different classification models were trained and evaluated using 5-fold cross-validation on the SMOTE-balanced training data:
        * Decision Tree Classifier
        * Random Forest Classifier
        * XGBoost Classifier
    * Based on the cross-validation accuracy, the **Random Forest Classifier** was selected as the best-performing model.

6.  **Model Evaluation:**
    * The selected Random Forest model was trained on the entire SMOTE-balanced training set.
    * The model's performance was evaluated on the unseen test set, achieving an accuracy of approximately **78%**.
    * A detailed classification report and a confusion matrix were generated to assess precision, recall, and F1-score for each class.

7.  **Predictive System & Model Persistence:**
    * The final trained Random Forest model and the list of feature names were saved to a pickle file (`customer_churn_model.pkl`).
    * A simple predictive system was built to demonstrate how to load the saved model and encoders to make a prediction on a new, single data point.

## 4. How to Use

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Jupyter notebook:**
    ```bash
    jupyter notebook customer_churn_prediction.ipynb
    ```

## 5. Files in the Repository

* `customer_churn_prediction.ipynb`: The main Jupyter notebook containing the complete analysis and model training.
* `WA_Fn-UseC_-Telco-Customer-Churn.csv`: The dataset used for the project.
* `customer_churn_model.pkl`: The saved trained Random Forest model.
* `encoders.pkl`: The saved label encoders for the categorical features.
* `requirements.txt`: A list of the Python libraries required to run the project.
* `README.md`: This file.
