import pandas as pd
import pickle
import numpy as np

from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

try:
    arquivo_teste = pd.DataFrame(pd.read_csv("test.csv", sep=","))

except FileNotFoundError:
    print("ARQUIVO NAO FOI ENCONTRADO")

except Exception as e:
    print(f"ERRO AO SUBIR ARQUIVO: {e}")

modelo_carregado = pickle.load(open("modelo_finalizado.sav", "rb"))

# REALIZANDO O TRATAMENTO NO ARQUIVO DE TESTE

arquivo_teste = arquivo_teste.replace(".", "").replace(",", ".")
arquivo_teste.columns = arquivo_teste.columns.str.upper()

arquivo_teste = arquivo_teste.drop(["ID", "SALECONDITION", "YRSOLD", "MOSOLD", "YEARBUILT", "MISCFEATURE", "POOLQC",
                                    "FENCE", "ALLEY", 'STREET', 'LANDCONTOUR', 'UTILITIES', 'LANDSLOPE', 'CONDITION1',
                                    'CONDITION2', 'BLDGTYPE', 'ROOFMATL', 'EXTERCOND', 'BSMTCOND', 'BSMTFINTYPE2',
                                    'BSMTFINSF2', 'HEATING', 'CENTRALAIR', 'ELECTRICAL', 'LOWQUALFINSF', 'BSMTHALFBATH',
                                    'KITCHENABVGR', 'FUNCTIONAL', 'GARAGEQUAL', 'GARAGECOND', 'PAVEDDRIVE',
                                    'ENCLOSEDPORCH', '3SSNPORCH', 'SCREENPORCH', 'POOLAREA', 'MISCVAL', 'SALETYPE',
                                    'LOTAREA', 'LOTCONFIG', 'OVERALLQUAL', 'YEARREMODADD', 'ROOFSTYLE', 'EXTERIOR1ST',
                                    'FOUNDATION', 'BSMTEXPOSURE', 'BSMTFINTYPE1', '2NDFLRSF', 'LOTSHAPE', 'EXTERIOR2ND',
                                    'MASVNRTYPE', 'BSMTUNFSF', '1STFLRSF', 'KITCHENQUAL', 'MASVNRAREA', 'FULLBATH',
                                    'NEIGHBORHOOD', 'GARAGETYPE', 'HOUSESTYLE', 'BEDROOMABVGR', 'EXTERQUAL',
                                    'OVERALLCOND', 'FIREPLACES', 'BSMTFINSF1', "TOTRMSABVGRD"], axis="columns")

num_att = arquivo_teste.select_dtypes(exclude=["object", "datetime"]).columns.to_list()
cat_att = arquivo_teste.select_dtypes(include=["object"]).columns.to_list()

imputer_mediana = SimpleImputer(strategy="median")
imputer_frequencia = SimpleImputer(strategy="most_frequent")

encoder = OrdinalEncoder()
scaler = MinMaxScaler()

for num in num_att:
    arquivo_teste[num] = imputer_mediana.fit_transform(np.array(arquivo_teste[num]).reshape(-1, 1))
    arquivo_teste[num] = scaler.fit_transform(np.array(arquivo_teste[num]).reshape(-1, 1))

for cat in cat_att:
    arquivo_teste[cat] = imputer_frequencia.fit_transform(np.array(arquivo_teste[cat]).reshape(-1, 1))
    arquivo_teste[cat] = encoder.fit_transform(np.array(arquivo_teste[cat]).reshape(-1, 1))

arquivo_teste.dropna()

X = arquivo_teste[["BSMTQUAL", "TOTALBSMTSF", "GARAGEFINISH", "MSZONING",
                    "OPENPORCHSF", "WOODDECKSF", "FIREPLACEQU", "LOTFRONTAGE"]]

# REALIZANDO O PREDICT DO ARQUIVO DE TESTE COM O MODELO IMPORTADO

arquivo_teste["RESULTADO"] = modelo_carregado.predict(X) * 10000

# MOSTRANDO O RESULTADO DO ARQUIVO DE TESTE

print(arquivo_teste)
