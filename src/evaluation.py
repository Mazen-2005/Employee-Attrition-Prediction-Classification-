from model import model_selection
import pandas as pd
def evaluation(df : pd.DataFrame) :
    for model_name, accuracy, report in model_selection(df):
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
        print("-" * 50)