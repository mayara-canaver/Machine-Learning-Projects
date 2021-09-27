# IMPORTANDO BIBLIOTECAS E PACOTES UTILIZADOS

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error

warnings.simplefilter("ignore")
pd.options.display.max_columns = None

# REALIZANDO A IMPORTAÇÃO DO ARQUIVO DO AVOCADO

try:
    arquivo = pd.read_csv("avocado.csv", sep=",")

except FileNotFoundError:
    print("ARQUIVO NAO FOI ENCONTRADO")

except Exception as e:
    print(f"ERRO AO SUBIR ARQUIVO: {e}")

# REALIZANDO O TRATAMENTO DOS DADOS

arquivo = arquivo.replace(".", "").replace(",", ".")
arquivo.columns = arquivo.columns.str.upper()

# SUBSTITUINDO O TIPO DE AVOCADO CONVENCIONAL POR 1 E O ORGANICO POR 0

arquivo["TYPE"] = arquivo["TYPE"].replace("conventional", 1).replace("organic", 0)

# CRIANDO UM ARRANJAMENTO DOS VALORES DOS AVOCADOS POR ANO

ano_unico = arquivo["YEAR"].unique()

gby_arquivo = arquivo.groupby(["YEAR"]).sum(["TOTAL VOLUME"])

# VISUALIZANDO QUANTIDADE DE AVOCADOS VENDIDOS POR ANO

plt.title("Volume de Avocados vendidos através dos anos")
plt.xlabel("Anos")
plt.ylabel("Volume Total")
plt.scatter(ano_unico, gby_arquivo["TOTAL VOLUME"])
plt.plot(ano_unico, gby_arquivo["TOTAL VOLUME"])
plt.xticks(range(min(ano_unico), max(ano_unico), 1))
plt.show()

# COMO O DATASET TEM OS DADOS DE NO MÁXIMO ATÉ 2018, ENTÃO NÃO PEGAREMOS OS AVOCADOS VENDIDOS ATÉ ESSA DATA
# PODEMOS PERCEBER UM AUMENTO NO NÚMERO DE AVOCADOS

arquivo = arquivo.loc[arquivo["YEAR"] != 2018]

plt.title("Volume de Avocados vendidos através dos anos")
plt.xlabel("Anos")
plt.ylabel("Volume Total")
plt.scatter(ano_unico, gby_arquivo["TOTAL VOLUME"])
plt.plot(ano_unico, gby_arquivo["TOTAL VOLUME"])
plt.show()

# SEM O ANO DE 2018 PODEMOS PERCEBER O AUMENTO CRESCENTE DAS VENDAS DOS AVOCADOS
# ANALISANDO QUAL REGIÃO MAIS CONSOME AVOCADO E QUAL REGIÃO CONSOME MENOS

# CRIANDO UM CONJUNTO DAS REGIOES POR VOLUME TOTAL DE AVOCADO

gby_arquivo = arquivo.groupby(["REGION"], as_index=False).sum()
gby_arquivo.drop(gby_arquivo.index[gby_arquivo["REGION"] == "TotalUS"], inplace=True)

plt.title("Análise de consumo de Avocado por região")
plt.xlabel("Região")
plt.ylabel("Volume Total")
plt.bar(gby_arquivo["REGION"], gby_arquivo["TOTAL VOLUME"])
plt.xticks(rotation=75)
plt.tick_params(axis='x', which='major', labelsize=5)
plt.show()

# VERIFICAMOS QUE A REGIÃO WEST É A QUE MAIS CONSOME AVOCADOS, ENQUANTO A REGIÃO QUE MENOS CONSOME É A SYRACUSE

# VISUALIZANDO A MÉDIA DE PREÇO DE CADA REGIÃO

plt.title("Média de Preço dos Avocados por Região")
plt.xlabel("Região")
plt.ylabel("Média de Preço")
plt.scatter(gby_arquivo["REGION"], gby_arquivo["AVERAGEPRICE"] / 314)
plt.plot(gby_arquivo["REGION"], gby_arquivo["AVERAGEPRICE"] / 314)
plt.xticks(rotation=75)
plt.tick_params(axis='x', which='major', labelsize=5)
plt.show()

# PODEMOS VERIFICAR QUE A REGIÃO MAIS CARA É A HARTFORDSPRINGFIELD ENQUANTO A MAIS BARATA É A HOUSTON

# DROPANDO A COLUNA UNNAMED PELA FALTA DE INFORMAÇÕES DA COLUNA
# DROPANDO A COLUNA DATE E YEAR POR SER UM MODELO DE REGRESSÃO
# DROPANDO AS COLUNAS DE BAGS POR NÃO TER INFORMAÇÕES NO DATASET SOBRE O QUE É

arquivo = arquivo.drop(["UNNAMED: 0", "DATE", "YEAR", "SMALL BAGS", "LARGE BAGS", "XLARGE BAGS"], axis=1)

# REALIZANDO A PORCENTAGEM DO VOLUME TOTAL COMPARADO COM OS CODIGOS DOS ABACATES

for codigo in arquivo[["4046", "4225", "4770", "TOTAL BAGS"]]:
    arquivo[codigo] = (arquivo[codigo] / arquivo["TOTAL VOLUME"]) * 100

data = arquivo.sample(frac=0.9, random_state=786).reset_index(drop=True)

print('Data for Modeling: ' + str(data.shape))
from pycaret.regression import *

#

