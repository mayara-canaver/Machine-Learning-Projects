# IMPORTANDO BIBLIOTECAS NECESSARIAS
import pandas as pd
import pickle
import numpy as np
import math

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

jog_treino = pd.read_csv("arquivos_dado/Train_rev1.csv", error_bad_lines=False)

# TRATAMENTO DOS DADOS DE CADA COLUNA
jog_treino.columns = [x.upper() for x in jog_treino.columns]

for coluna in jog_treino.columns:
    if jog_treino[coluna].dtype == object:
        jog_treino[coluna] = jog_treino[coluna].str.strip()
        jog_treino[coluna] = jog_treino[coluna].str.upper()

# PREENCHIMENTO DE VALURES NAN
jog_treino["CONTRACTTIME"] = jog_treino["CONTRACTTIME"].fillna("CONTRACT")

jog_treino["CONTRACTTYPE"] = jog_treino["CONTRACTTYPE"].fillna("NAO_LOCALIZADO")

# TRATAMENTO E REDUCAO DA COLUNA SOURCENAME
sourcename_restrito = list(jog_treino["SOURCENAME"].value_counts()[:10].index.values)

jog_treino = jog_treino[jog_treino["SOURCENAME"].isin(sourcename_restrito)]

# NORMALIZACAO DOS DADOS CATEGORICOS
cat_att = jog_treino.select_dtypes(include=["object"]).columns.to_list()

imputer_frequencia = SimpleImputer(strategy="most_frequent")

encoder = OrdinalEncoder()

for cat in cat_att:
    jog_treino[cat] = imputer_frequencia.fit_transform(np.array(jog_treino[cat]).reshape(-1, 1))
    jog_treino[cat] = encoder.fit_transform(np.array(jog_treino[cat]).reshape(-1, 1))

# REMOCAO DE COLUNAS MENOS UTILIZADAS
jog_treino = jog_treino.drop(["LOCATIONRAW", "FULLDESCRIPTION"], axis=1)

# SEPARANDO OS VALORES DE X E Y

X = jog_treino[["CONTRACTTIME", "CONTRACTTYPE", "CATEGORY", "SOURCENAME", "LOCATIONNORMALIZED", "COMPANY", "TITLE"]]
y = jog_treino["SALARYNORMALIZED"]

# SEPARANDO OS VALORES ENTRE TREINO E TESTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)

# REALIZANDO O TUNING DE PARAMETROS
"""np.random.seed(1)
grid = {
    "n_estimators": np.arange(10, 100, 10),
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": np.arange(2, 20, 2),
    "min_samples_leaf" : np.arange(1, 20, 2),
    "max_features": [0.5, 1, "sqrt", "auto"],
    "max_samples": [10000, 12000, 15000, 20000]
}

rs_model = RandomizedSearchCV(RandomForestRegressor(random_state=1), param_distribuitions=grid)
"""
# REALIZANDO O PREDICT E FIT DO MODELO
modelo = RandomForestRegressor(random_state=1, n_estimators=50, min_samples_split=6, max_samples=20000)

modelo_previsao = modelo.fit(X_train, y_train)

predicao = modelo_previsao.predict(X_test)

# REALIZANDO AS METRICAS DO MODELO

mse = mean_squared_error(y_test, predicao)
r2 = r2_score(y_test, predicao)
mae = mean_absolute_error(y_test, predicao)
mape = mean_absolute_percentage_error(y_test, predicao)

print("RANDOM FOREST:\n\nMSE: %.4f\nR2: %.4f\nMAE: %.4f\nMAPE: %.4f" % (mse, r2, mae, mape))

# DESVIO PADRÃO
rse = math.sqrt(mse/(X.shape[0]-2))

df_final = pd.DataFrame()

df_final["PREDICAO"] = predicao
df_final["MIN_PREDICAO"] = predicao - rse
df_final["MAX_PREDICAO"] = predicao + rse

# EXPORTANDO O MODELO
finalizado = "modelo_finalizado.sav"

pickle.dump(modelo, open(finalizado, "wb"))
