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


def remove_correlacao(arquivo, valor_corte):
    matriz_correlacao = arquivo.corr().abs()

    matriz_superior = matriz_correlacao.where(np.triu(np.ones(matriz_correlacao.shape), k=1).astype(np.bool))

    exclusao_correlacao = [coluna for coluna in matriz_superior.columns if any(matriz_superior[coluna] >= valor_corte)]

    return arquivo.drop(exclusao_correlacao, axis=1)


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


# REALIZANDO A IMPORTAÇÃO DO ARQUIVO DE TESTE E DE TREINO

try:
    arquivo_treino = pd.read_csv("train.csv", sep=",")

except FileNotFoundError:
    print("ARQUIVO NAO FOI ENCONTRADO")

except Exception as e:
    print(f"ERRO AO SUBIR ARQUIVO: {e}")

# REALIZANDO FORMATACAO DE PONTUACAO, DEIXAR AS COLUNAS PADRONIZADAS EM MAIUSCULA E REMOVENDO ACENTUACAO DAS COLUNAS

arquivo_treino = arquivo_treino.replace(".", "").replace(",", ".")
arquivo_treino.columns = arquivo_treino.columns.str.upper()

cols = arquivo_treino.select_dtypes(include=["object"]).columns
arquivo_treino[cols] = (arquivo_treino[cols].
                        apply(lambda x1: x1.str.normalize("NFKD").str.encode('ascii', errors='ignore').
                              str.decode('utf-8')))

# VERIFICANDO QUANTIDADE DE LINHAS CONTENDO VALORES NAN DAS COLUNAS

for coluna in arquivo_treino:
    "print(arquivo_treino[coluna].isna().sum(), coluna)"

# COLUNAS QUE SE MOSTRARAM INVIÁVEIS PARA O MODELO:
# -> MISCFEATURE = 1406 NAN
# -> POOLQC = 1453 NAN
# -> FENCE = 1179 NAN
# -> ALLEY = 1369 NAN

arquivo_treino = arquivo_treino.drop(["MISCFEATURE", "POOLQC", "FENCE", "ALLEY"], axis=1)

# VERIFICANDO VALORES ÚNICOS PARA RETIRADA DE COLUNAS

lista_dropagem = []

for coluna in arquivo_treino:
    if arquivo_treino[coluna].value_counts().max() >= 1200:
        lista_dropagem.append(coluna)

arquivo_treino = arquivo_treino.drop(lista_dropagem, axis=1)

# FAZENDO O CORTE DE COLUNAS CORRELACIONADAS DE 0.60 OU MAIS

corte = 0.6

arquivo_sem_target = arquivo_treino.drop(["SALEPRICE"], axis=1)

arquivo_sem_target = remove_correlacao(arquivo_sem_target, corte)

arquivo_treino = pd.concat([arquivo_treino["SALEPRICE"], arquivo_sem_target], axis=1)

# VERIFICANDO CORRELAÇÃO ENTRE COLUNAS PELO MATPLOT

plt.matshow(arquivo_treino.corr())
plt.title('MATRIZ CORRELACAO', fontsize=10)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)
plt.show()

# DIVIDINDO EM VARIÁVEL NUMÉRICA PARA REMOVER NAN E INSERIR A MEDIANA NO LUGAR
# DIVIDINDO EM VARIÁVEL CATEGÓRICA PARA REMOVER NAN E INSERIR A FREQUÊNCIA NO LUGAR

num_att = arquivo_treino.select_dtypes(exclude=["object", "datetime"]).columns.to_list()
cat_att = arquivo_treino.select_dtypes(include=["object"]).columns.to_list()

imputer_mediana = SimpleImputer(strategy="median")
imputer_frequencia = SimpleImputer(strategy="most_frequent")

encoder = OrdinalEncoder()
scaler = MinMaxScaler()

for num in num_att:
    arquivo_treino[num] = imputer_mediana.fit_transform(np.array(arquivo_treino[num]).reshape(-1, 1))
    arquivo_treino[num] = scaler.fit_transform(np.array(arquivo_treino[num]).reshape(-1, 1))

for cat in cat_att:
    arquivo_treino[cat] = imputer_frequencia.fit_transform(np.array(arquivo_treino[cat]).reshape(-1, 1))
    arquivo_treino[cat] = encoder.fit_transform(np.array(arquivo_treino[cat]).reshape(-1, 1))

arquivo_treino.dropna()

# ESCOLHENDO AS VARIAVEIS PARA O X

y = arquivo_treino["SALEPRICE"]
X = arquivo_treino.drop(["ID", "SALECONDITION", "YRSOLD", "MOSOLD", "YEARBUILT", "SALEPRICE"], axis=1)

# VERIFICANDO MELHORES VARIÁVEIS COM KBEST FEATURES

bestfeatures = SelectKBest(k=10)
fit = bestfeatures.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']
print(featureScores.nlargest(15, 'Score'))

# ATRIBUINDO COLUNAS PARA OS VALORES DE TREINO E TESTE

X = arquivo_treino[["BSMTQUAL", "TOTALBSMTSF", "GARAGEFINISH", "MSZONING",
                    "OPENPORCHSF", "WOODDECKSF", "FIREPLACEQU", "LOTFRONTAGE"]]

y = arquivo_treino["SALEPRICE"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, shuffle=True)

# VERIFICANDO RESULTADO DE CADA MODELO DE REGRESSÃO FEITO

lista_modelos = [RandomForestRegressor(random_state=1), LinearRegression(), Ridge(solver="auto", alpha=1.0),
                 DecisionTreeRegressor(max_depth=3, random_state=1)]

for tipo_modelo in lista_modelos:
    retorna_resultado_modelo(tipo_modelo)

"VERIFICOU-SE ENTÃO QUE O MELHOR MODELO EM QUESTÃO É DE RANDOM FOREST"

# EXPORTANDO O MODELO COM O PICKLE

modelo = RandomForestRegressor(random_state=1).fit(X_train, y_train)

finalizado = "modelo_finalizado.sav"

pickle.dump(modelo, open(finalizado, "wb"))
