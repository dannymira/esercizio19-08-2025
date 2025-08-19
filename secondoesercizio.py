import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Caricamento e pulizia dataset
df = pd.read_csv('AirQualityUCI.csv', sep=';')
df = df.iloc[:, :-2]  # rimuovi colonne vuote in fondo

# Conversione virgola decimale in punto per le colonne oggetto
def to_float(x):
    try:
        return float(str(x).replace(',', '.'))
    except:
        return np.nan

df['NO2(GT)'] = df['NO2(GT)'].apply(to_float)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df['Time'] = pd.to_datetime(df['Time'], format='%H.%M.%S', errors='coerce').dt.time
df = df.dropna(subset=['NO2(GT)', 'Date'])

# Calcolo media giornaliera di NO2
df['giorno'] = df['Date']
media_giornaliera = df.groupby('giorno')['NO2(GT)'].transform('mean')
media_settimanale = df.groupby(pd.Grouper(key='Date', freq='W'))['NO2(GT)'].transform('mean')
media_globale = df['NO2(GT)'].mean()

# Classificazione qualità dell'aria rispetto alla media giornaliera
df['scarsa_qualita'] = (df['NO2(GT)'] > media_giornaliera).astype(int)

# Logistic Regression sul dato orario
feature_cols = ['NO2(GT)']  #qui si possono aggiungere altre colonne se voglio, per il momento testo su un solo dato
X = df[feature_cols]
y = df['scarsa_qualita']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Valutazione
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Identificazione delle 3 ore di picco per ogni giorno
df['hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
peak_hours = df.groupby('giorno').apply(lambda x: x.nlargest(3, 'NO2(GT)'))
peak_hours = peak_hours[['Date', 'Time', 'NO2(GT)']]

print("Ore di picco giornaliere:")
print(peak_hours)

# Percentuale di "scarsa qualità" giornaliera e confronto con la media globale
perc_scarse_giorno = df.groupby('giorno')['scarsa_qualita'].mean() * 100
perc_scarse_globale = df['scarsa_qualita'].mean() * 100

print(f"Percentuale media ore 'scarsa qualità' giornaliera: \n{perc_scarse_giorno}")
print(f"Percentuale globale ore 'scarsa qualità': {perc_scarse_globale:.2f}%")