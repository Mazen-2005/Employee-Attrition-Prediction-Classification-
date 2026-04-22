## 🎯 Problem Definition \& Objectives

•	Objective: To proactively predict employee turnover risk within an organization.

•	Task Type: Supervised Learning (Binary Classification).

•	Target Variable: Attrition (1 = Left, 0 = Stayed).

•	Core Metrics: The project focuses primarily on Recall and F1-Score.

•	Baseline Significance: Establishing a baseline attrition rate helps evaluate the severity of the retention problem before applying ML models.



## 📊 Dataset Overview

•	Total Employees: 14,900.

•	Total Features: 23.

•	Attrition Rate: 47.19% of employees in the dataset have left.

•	Feature Categories:

•	Numerical: Age, Monthly Income, Years at Company, etc .

•	Nominal/Ordinal: Job Role, Marital Status, Job Satisfaction, and Performance Rating .

•	Binary: Gender, Overtime, and Remote Work .



## 🛠️ Preprocessing \& EDA Insights

•	Cleaning Steps:

•	Identified and removed outliers in 'Years at Company' and 'Monthly Income' using the IQR Method.

•	Formula used for bounds: $IQR = Q3 - Q1$.

•	Impactful Features: Job Level and Marital Status (Single/Married) were identified as top factors .

•	Pipeline:

•	Label Encoding: Converted categorical strings like 'Yes/No' or 'Male/Female' into numerical values (1/0) .

•	Scaling: Applied RobustScaler to handle features with remaining outliers.

•	Data Split: 80% Training (11,920 rows) and 20% Testing (2,980 rows).



## 🤖 Modeling Strategy

We explored various algorithms to capture different patterns in the HR data:

1\.	Baseline Models: Logistic Regression and Support Vector Machines (SVM).

2\.	Ensemble Methods: Random Forest (100 trees) and XGBoost (Sequential boosting).

3\.	Deep Learning: An Artificial Neural Network (ANN) featuring 3 Dense layers with ReLU activation.

4\.	📈 Model Performance \& Results

5\.	Hyperparameter tuning and 5-fold cross-validation were performed using RandomizedSearchCV.



Model			             > F1 Score	    > ROC-AUC

Logistic Regression	   > 0.76 		    > 0.753 

Random Forest		       --- 0.75 		    --- 0.7594 

XGBoost			           --- 0.74 		    --- 0.7432 

SVM 		               --- 0.76 		    --- 0.7628 

ANN (Deep Learning)    --- 0.76 	      ---	0.7580 



Why SVM? SVM's ability to use the "Kernel Trick" and "Soft Margin" allows it to effectively separate overlapping classes in higher dimensions, making it a robust choice for this dataset .



💡 Conclusion

The models achieved a realistic performance range of 74% to 76%. While higher accuracy is technically possible, in HR contexts, accuracy exceeding 90% is often a sign of Data Leakage or Overfitting. Our results reflect the complexity of human behavior, which involves subjective factors not always present in numerical datasets.





