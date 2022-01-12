import pandas as pd
import pickle
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

try:
    job_teste = pd.read_csv("arquivos_dado/Valid_rev1.csv", error_bad_lines=False)

except FileNotFoundError:
    print("ARQUIVO NAO FOI ENCONTRADO")

except Exception as e:
    print(f"ERRO AO SUBIR ARQUIVO: {e}")

# CARREGANDO O MODELO PRONTO
modelo_carregado = pickle.load(open("modelo_finalizado.sav", "rb"))

# TRATAMENTO DOS DADOS DE CADA COLUNA
job_teste.columns = [x.upper() for x in job_teste.columns]

for coluna in job_teste.columns:
    if job_teste[coluna].dtype == object:
        job_teste[coluna] = job_teste[coluna].str.strip()
        job_teste[coluna] = job_teste[coluna].str.upper()

# PREENCHIMENTO DE VALURES NAN
job_teste["CONTRACTTIME"] = job_teste["CONTRACTTIME"].fillna("CONTRACT")

job_teste["CONTRACTTYPE"] = job_teste["CONTRACTTYPE"].fillna("NAO_LOCALIZADO")

# TRATAMENTO E REDUCAO DA COLUNA SOURCENAME
sourcename_restrito = list(job_teste["SOURCENAME"].value_counts()[:10].index.values)

job_teste = job_teste[job_teste["SOURCENAME"].isin(sourcename_restrito)]

# NORMALIZACAO DOS DADOS CATEGORICOS
cat_att = job_teste.select_dtypes(include=["object"]).columns.to_list()

imputer_frequencia = SimpleImputer(strategy="most_frequent")

encoder = OrdinalEncoder()

for cat in cat_att:
    job_teste[cat] = imputer_frequencia.fit_transform(np.array(job_teste[cat]).reshape(-1, 1))
    job_teste[cat] = encoder.fit_transform(np.array(job_teste[cat]).reshape(-1, 1))

# REMOCAO DE COLUNAS MENOS UTILIZADAS
job_teste = job_teste.drop(["LOCATIONRAW", "FULLDESCRIPTION"], axis=1)

# SEPARANDO AS COLUNAS PARA O MODELO

X = job_teste[["CONTRACTTIME", "CONTRACTTYPE", "CATEGORY", "SOURCENAME", "LOCATIONNORMALIZED", "COMPANY", "TITLE"]]

# REALIZANDO O PREDICT DO ARQUIVO DE TESTE COM O MODELO IMPORTADO

job_teste["RESULTADO"] = modelo_carregado.predict(X)

# MOSTRANDO O RESULTADO DO ARQUIVO DE TESTE

print(job_teste[["ID", "RESULTADO"]])