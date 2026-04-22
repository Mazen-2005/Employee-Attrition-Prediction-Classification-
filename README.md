# Employee Attrition Prediction

## Introduction

Retaining top talent is one of the most critical challenges facing organizations today. Employee attrition—when employees leave a company—not only incurs significant financial costs related to recruiting and training new hires, but it also disrupts team dynamics, lowers morale, and slows down productivity. To stay competitive, companies can no longer rely on guesswork to manage workforce retention. Instead, they are increasingly turning to data-driven solutions to understand the underlying causes of turnover and proactively identify employees who may be at risk of leaving.

## Problem Statement

The organization is experiencing unexpected employee turnover, which impacts operational efficiency and steadily drives up recruitment costs. Currently, human resources teams rely on reactive measures rather than predictive insights to manage workforce retention, making it difficult to intervene before an employee decides to quit.

The objective of this project is to develop a machine learning classification model capable of accurately predicting whether an employee is likely to leave or stay. By analyzing historical workforce data—such as compensation, job satisfaction, work-life balance, and tenure—this model will provide actionable insights, enabling HR to identify high-risk employees early and implement targeted, preventative retention strategies.

## Dataset

1. General Overview

   Total Rows (Records): 14,900 employees

   Total Columns (Features): 24

   Missing Values: None (All columns have 14,900 non-null values, making it a very clean dataset).

2. Dataset Features

   The 24 columns can be categorized into three main groups:

   1. Demographics \& Personal Info:
   * Age: Ranging from 18 to 59 years (Average: 38 years).
   * Gender: Male, Female, etc.
   * Marital Status: Married, Single, Divorced.
   * Number of Dependents: Ranges from 0 to 6.
   * Distance from Home: Distance of the employee's commute (ranges from 1 to 99 units/miles).
   * Education Level: E.g., Associate Degree, Bachelor's Degree, Master's Degree.
   1. Employment details:
   * Employee ID: Unique identifier for each employee.
   * Job Role: The department/field (e.g., Healthcare, Education, Finance, Media, Technology).
   * Job Level: Entry, Mid, Senior.
   * Years at Company: Ranges from 1 to 51 years (Average: \~15.6 years).
   * Company Tenure: Overall tenure length (Ranges from 2 to 127 months).
   * Monthly Income: Salary ranging from 1,226 to 15,063 (Average: \~7,287).
   * Number of Promotions: Number of times promoted (0 to 4 times).
   * Company Size: Small, Medium, Large.
   * Remote Work: Indicates if they work remotely (Yes/No).
   * Overtime: Indicates if they regularly work overtime (Yes/No).
   1. Satisfaction \& Environment Metrics:
   * Work-Life Balance: Excellent, Good, Fair, Poor.
   * Job Satisfaction: Very High, High, Medium, Low.
   * Performance Rating: Average, High, etc.
   * Company Reputation: Poor, Good, etc.
   * Employee Recognition: Low, Medium, High.
   * Leadership Opportunities: (Yes/No)
   * Innovation Opportunities: (Yes/No)
   1. Target Variable:
   * Attrition: Whether the employee "Left" or "Stayed". This is the primary outcome variable if you are trying to build predictive models.
3. Key Insights from Data Statistics

   The workforce leans quite experienced with an average Years at Company of roughly 15.6 years.

   There is a wide variance in Monthly Income (as low as 1,226 and as high as 15,063), aligning with the presence of multiple job levels (Entry to Senior).

   The average employee commute (Distance from Home) is roughly 50 units.

   ## Methodology

1. Data Cleaning

   One of the key advantages of this project was the high quality of the initial dataset. Upon initial inspection, it was confirmed that the dataset contained 14,900 records and 24 features with zero missing values or duplicated rows. Because the data was already perfectly clean and well-structured out of the box, standard data cleaning steps—such as missing value imputation, fixing inconsistencies, or outlier removal—were entirely unnecessary. The only structural adjustment made before analysis was dropping the Employee ID column, as it serves solely as a unique identifier and holds no predictive value for the machine learning models.

2. Exploratory Data Analysis (EDA)

   * Before building the predictive models, a comprehensive Exploratory Data Analysis (EDA) was conducted to uncover underlying patterns, correlations, and business insights. Key analytical steps included:
   * Target Variable Distribution: A visual inspection of the Attrition target variable was performed to understand the proportion of employees who stayed versus those who left, helping to identify if there was a class imbalance.
   * Feature Distributions: Statistical summaries and distributions of numerical features like Age, Monthly Income, and Years at Company were analyzed to understand the demographic and financial makeup of the workforce.
   * Feature Relationships: We explored how different categorical and numerical features (e.g., Work-Life Balance, Job Satisfaction, and Distance from Home) impacted the likelihood of an employee leaving the company.
3. Modeling

   Since employee attrition prediction is a binary classification problem (Stayed vs. Left), we opted to train and evaluate multiple machine learning algorithms to compare their performance and select the one that best captured the underlying patterns in the HR data. The models implemented included:

   * Logistic Regression: Used as a baseline model due to its simplicity, efficiency, and high interpretability.
   * Support Vector Machine (SVM): Implemented to capture complex boundaries between the two employee classes in high-dimensional space.
   * XGBoost: A powerful gradient boosting framework utilized to maximize predictive accuracy and handle complex feature interactions.
   * Hyperparameter Tuning: We utilized Random Search (RandomizedSearchCV) across our models to systematically fine-tune hyperparameters. This step ensured we achieved the highest possible accuracy and generalization without overfitting the training data.

   Each model was evaluated to ensure the final selected algorithm was highly reliable in identifying at-risk employees.



   ## Results

   We evaluated multiple machine learning algorithms—including Logistic Regression, Random Forest, XGBoost, Neural Networks, and Support Vector Machines (SVM)—to determine the most effective model for predicting employee turnover. Because identifying at-risk employees was the primary goal, we evaluated the models using Accuracy, Precision, Recall, and the F1-Score to ensure false positives and false negatives were minimized.

   Among the baseline models evaluated, the Support Vector Machine (SVM) emerged as the strongest performer. It achieved the highest overall accuracy at 76.28%. More importantly, it delivered a strong Recall score of 74% and a Precision of 75% for the attrition class (employees who left), yielding a balanced F1-Score of 0.75.

   To further optimize the SVM model, we applied hyperparameter tuning using RandomizedSearchCV across 50 fits. The tuning process identified the optimal parameters as kernel = 'rbf', gamma = 0.01, and C = 1, which produced a final test set accuracy of 75.6%, confirming the model's consistent ability to generalize to unseen data.

   ## Conclusion

   This project successfully demonstrated the power of machine learning in addressing employee attrition. By leveraging a comprehensive HR dataset, we evaluated an array of classification models and identified the Support Vector Machine (SVM) algorithm as the most robust solution for this specific workforce data. Achieving an accuracy of over 76% alongside strong recall and precision, the model proves highly capable of identifying employees who are genuinely at risk of leaving without raising unnecessary false alarms.

   By transitioning away from reactive measures and deploying predictive insights like these, human resources teams can proactively identify high-risk employees early. This enables the implementation of targeted, data-driven retention strategies, ultimately preserving top talent and saving the organization significant recruitment and training costs.

   ## Future Work

   To further enhance the business value of this project, future iterations could focus on the following areas:

* System Integration: Wrap the finalized model in a backend architecture (such as an ASP.NET Core Web API) to integrate directly with the company's existing HR software, enabling a real-time attrition risk dashboard for management.
* Explainable AI (XAI): Implement techniques like SHAP (SHapley Additive exPlanations) values to make the model's internal decision-making completely transparent, telling HR exactly why an employee's risk score is high.
* Cost-Benefit \& ROI Analysis: Connect the model's predictions to actual financial data to estimate the monetary cost of the predicted attrition, allowing the company to calculate the direct Return on Investment (ROI) of offering preemptive raises or flexibility to at-risk employees.

