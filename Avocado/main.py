# IMPORTANDO BIBLIOTECAS E PACOTES UTILIZADOS

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error

warnings.simplefilter("ignore")
pd.options.display.max_columns = None

def retorna_resultado_modelo(tipo_modelo_parametro):
    # REALIZANDO O FIT E PREDICT DO MODELO

    pred = tipo_modelo_parametro.fit(X_train, y_train).predict(X_test)

    # REALIZANDO TESTES DE DESEMPENHO DE MODELO

    """MEAN SQUARED ERROR"""
    mse = mean_squared_error(y_test, pred)

    """MEAN ABSOLUTE ERROR"""
    mae = mean_absolute_error(y_test, pred)

    """R2"""
    r2 = r2_score(y_test, pred)

    """MAX ERROR"""
    max_erro = max_error(y_test, pred)

    return print(f"MODELO {tipo_modelo_parametro}\nMSE: {mse}\nMAE: {mae}\nR2: {r2}\n"
                 f"MAX ERROR: {max_erro}\n")


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

gby_arquivo = arquivo.groupby(["YEAR"]).sum(["TOTAL VOLUME"])

ano_unico = arquivo["YEAR"].unique()

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

# DIVIDINDO EM VARIÁVEL CATEGÓRICA PARA REMOVER NAN E INSERIR A FREQUÊNCIA NO LUGAR E TRANSFORMANDO EM NUMÉRICO

cat_att = arquivo.select_dtypes(include=["object"]).columns.to_list()

imputer_frequencia = SimpleImputer(strategy="most_frequent")

encoder = OrdinalEncoder()

for cat in cat_att:
    arquivo[cat] = imputer_frequencia.fit_transform(np.array(arquivo[cat]).reshape(-1, 1))
    arquivo[cat] = encoder.fit_transform(np.array(arquivo[cat]).reshape(-1, 1))

# REALIZANDO O SHUFFLE PARA MISTURAR AS LINHAS DO DATASET E CRIANDO UM ARQUIVO DE TESTE

arquivo = arquivo.sample(frac=1, random_state=1)

arquivo_teste = arquivo[0: (len(arquivo) // 2)]

arquivo = arquivo[(len(arquivo) // 2):]

# SEPARANDO O ARQUIVO EM ARQUIVO DE TREINO E TESTE E EXPORTANDO O DE TESTE

arquivo_teste.to_csv(r"C:\Users\Mayara Lopes\Desktop\GitHub\machine_learning_projects\Avocado\teste.csv", index=False)

# ESCOLHENDO AS VARIAVEIS PARA O X E FAZENDO O TRAIN TEST SPLIT

X = arquivo.drop(["AVERAGEPRICE"], axis=1)
y = arquivo["AVERAGEPRICE"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, shuffle=True)

# VERIFICANDO RESULTADO DE CADA MODELO DE REGRESSÃO FEITO

lista_modelos = [RandomForestRegressor(random_state=1), LinearRegression(), Ridge(solver="auto", alpha=1.0),
                 DecisionTreeRegressor(max_depth=3, random_state=1)]

for tipo_modelo in lista_modelos:
    retorna_resultado_modelo(tipo_modelo)

"VERIFICOU-SE ENTÃO QUE O MELHOR MODELO EM QUESTÃO É DE RIDGE"

# EXPORTANDO O MODELO COM O PICKLE

modelo = Ridge(solver="auto", alpha=1.0).fit(X_train, y_train)

finalizado = "modelo_finalizado.sav"

pickle.dump(modelo, open(finalizado, "wb"))
