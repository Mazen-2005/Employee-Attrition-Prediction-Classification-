from matplotlib.pyplot import grid
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

def model_selection(df : pd.DataFrame) :
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaled_columns = ['Age', 'Years at Company', 'Monthly Income', 'Distance from Home', 'Company Tenure']
    for col in scaled_columns:
        scaler = RobustScaler()
        X_train[col] = scaler.fit_transform(X_train[[col]])
        X_test[col] = scaler.transform(X_test[[col]])
        
    #1st model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    yield "Random Forest", accuracy_score(y_test, y_pred_rf), classification_report(y_test, y_pred_rf)
    
    
    #2nd model
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    yield "Logistic Regression", accuracy_score(y_test, y_pred_lr), classification_report(y_test, y_pred_lr)
    
    #3rd model
    svc = SVC(kernel='rbf', C=1.0)
    svc.fit(X_train, y_train)
    y_pred_svc = svc.predict(X_test)
    yield "SVM", accuracy_score(y_test, y_pred_svc), classification_report(y_test, y_pred_svc)
    
    
    #4th model
    xg_model = XGBClassifier()
    xg_model.fit(X_train, y_train)
    y_pred_xg = xg_model.predict(X_test)
    yield "XGBoost", accuracy_score(y_test, y_pred_xg), classification_report(y_test, y_pred_xg)
    
    #5th model
    model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid') 
    ])
    model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
    )
    early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
    )
    
    history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop]
    )
    
    #7th model
    pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('svc', SVC())
    ])

    param_grid = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': [1, 0.1, 0.01, 0.001],
        'svc__kernel': ['rbf', 'poly', 'sigmoid']
    }

    grid = RandomizedSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=2)

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred_grid = best_model.predict(X_test)
    yield "SVM with Grid Search", accuracy_score(y_test, y_pred_grid), classification_report(y_test, y_pred_grid)
    
    
