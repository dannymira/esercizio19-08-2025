
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

"""
caricamento dei dati
"""
data = pd.read_csv('EKPC_hourly.csv')
data['Datetime'] = pd.to_datetime(data['Datetime']) 

"""
calcolo della media giornaliera
"""
daily_avg = data.resample('D', on='Datetime').mean()
weekly_avg = data.resample('W', on='Datetime').mean()


"""Merge della media giornaliera sui dati orari"""
data = data.merge(daily_avg, on='Datetime', suffixes=('', '_Daily_Avg'))
data = data.merge(weekly_avg, on='Datetime', suffixes=('', '_Weekly_Avg'))

"Creazione della classe: 1 = alto consumo, 0 = basso consumo"
data['Consumption_Class_Daily'] = data['EKPC_MW'].apply(lambda x: 1 if x > data['EKPC_MW_Daily_Avg'].mean() else 0)
data['Consumption_Class_Weekly'] = data['EKPC_MW'].apply(lambda x: 1 if x > data['EKPC_MW_Weekly_Avg'].mean() else 0)

"""# Definizione feature e label"""
X = data[['EKPC_MW', 'EKPC_MW_Daily_Avg']]
y = data['Consumption_Class']

"""# Split train/test"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""# Normalizzazione"""
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""# Decision Tree"""
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt.fit(X_train, y_train)
y_pred_dt = clf_dt.predict(X_test)

"""# Rete neurale MLP"""
clf_mlp = MLPClassifier(random_state=42)
clf_mlp.fit(X_train, y_train)
y_pred_mlp = clf_mlp.predict(X_test)

"""# Classification report"""
print("Decision Tree Classifier Report:\n", classification_report(y_test, y_pred_dt))
print("MLP Classifier Report:\n", classification_report(y_test, y_pred_mlp))