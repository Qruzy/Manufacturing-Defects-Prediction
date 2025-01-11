import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv(r"C:\Users\User\Desktop\datascience\Final\manufacturing_defect_dataset.csv")

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = MinMaxScaler()

x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

smote = SMOTE()

x_train, y_train = smote.fit_resample(x_train, y_train)

rfc = RandomForestClassifier()

rfc.fit(x_train, y_train)
pred = rfc.predict(x_test)
score = accuracy_score(pred, y_test)

print(score)

joblib.dump(rfc, r"C:\Users\User\Desktop\datascience\Final\model\manu_def.pkl")