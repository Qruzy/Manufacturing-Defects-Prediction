import pandas as pd

def load_data():
    df = pd.read_csv(r"C:\Users\User\Desktop\datascience\Final\manufacturing_defect_dataset.csv")
    
    return df