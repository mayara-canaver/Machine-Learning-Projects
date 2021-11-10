# IMPORTANDO BIBLIOTECAS E PACOTES UTILIZADOS

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, accuracy_score, roc_curve, auc, precision_score, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

warnings.simplefilter("ignore")
pd.options.display.max_columns = None

def retorna_resultado_modelo(tipo_modelo_parametro):
    # REALIZANDO O FIT E PREDICT DO MODELO

    y_pred_parametro = tipo_modelo_parametro.fit(X_train, y_train).predict(X_test)

    # REALIZANDO TESTES DE DESEMPENHO DE MODELO

    """F1 SCORE"""
    f1 = f1_score(y_test, y_pred_parametro, average='macro')

    """RECALL"""
    recall = recall_score(y_test, y_pred_parametro, average='macro')

    """ACURÁCIA"""
    acuracia = accuracy_score(y_test, y_pred_parametro)

    """PRECISÃO"""
    precisao = precision_score(y_test, y_pred_parametro, average='macro')

    lista_testes[tipo_modelo] = [f1, recall, acuracia, precisao]

    return print(f"MODELO {tipo_modelo_parametro}\nF1: {f1}\nRECALL: {recall}\nACURACIA: {acuracia}\n"
                 f"PRECISAO: {precisao}\n")


# REALIZANDO A IMPORTAÇÃO DO ARQUIVO DE FRAUDE

try:
    arquivo_completo = pd.read_csv("creditcard.csv", sep=",")
except FileNotFoundError:
    print("ARQUIVO NAO FOI ENCONTRADO")
except Exception as e:
    print(f"ERRO AO SUBIR ARQUIVO: {e}")

# REALIZANDO FORMATACAO DE PONTUACAO E DEIXAR AS COLUNAS PADRONIZADAS EM MAIUSCULA

arquivo_completo = arquivo_completo.replace(".", "").replace(",", ".")
arquivo_completo.columns = arquivo_completo.columns.str.upper()

# REMOVENDO ACENTUACAO DAS COLUNAS

cols = arquivo_completo.select_dtypes(include=["object"]).columns
arquivo_completo[cols] = (arquivo_completo[cols].
                          apply(lambda x1: x1.str.normalize("NFKD").str.encode('ascii', errors='ignore').
                                str.decode('utf-8')))

# VERIFICANDO QUANTOS CASOS DE FRAUDE EXISTEM NO DATASET

arquivo_completo["CLASS"].value_counts()

"0  ==  284315  ||   1   ==    492"
"APENAS 0.17% DA BASE É FRAUDE"

# REALIZANDO A EXTRAÇÃO DE APENAS OS CASOS DE FRAUDE
# SEPARANDO APENAS 50 EXEMPLOS PARA QUE SEJA REALIZADO O TREINO

arquivo_fraudes = arquivo_completo.loc[np.where(arquivo_completo["CLASS"] == 1)][:50]

"""CRIADO UM NOVO DATAFRAME ONDE CONTEM APENAS OS CASOS DE FRAUDE"""

# VERIFICANDO NO NOVO DATAFRAME A RELAÇÃO ENTRE FRAUDES

plt.matshow(arquivo_fraudes.corr())
plt.title('MATRIZ CORRELACAO', fontsize=10)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)
plt.show()

"""COLUNAS QUE TEM MAIOR CORRELAÇÃO: V1, V2, V5, V7, V8, V12, V14, V17"""
"""MATRIZ DE CORRELAÇÃO CRIADA E VISUALIZAÇÃO FEITA TAMBÉM"""

# DEIXANDO O DF 50/50 COM VARIAVEIS DE FRAUDE E SEM FRAUDE PARA PODER REALIZAR O ESTUDO

arquivo_sem_fraudes = arquivo_completo.loc[np.where(arquivo_completo["CLASS"] == 0)][:50]

# MESCLANDO OS DOIS DF DE FRAUDE E SEM FRAUDE

arquivo_fraudes = pd.concat([arquivo_sem_fraudes, arquivo_fraudes])

# MANTENDO O ARQUIVO ORIGINAL MENOS COM AS LINHAS DO TREINO QUE SERA REALIZADO

arquivo_resto_fraudes = arquivo_completo.loc[np.where(arquivo_completo["CLASS"] == 1)][50:]

arquivo_resto_sem_fraudes = arquivo_completo.loc[np.where(arquivo_completo["CLASS"] == 0)][50:]

arquivo = pd.concat([arquivo_resto_fraudes, arquivo_resto_sem_fraudes])

# CRIANDO UM BACKUP

arquivo_fraudes_backup = arquivo_fraudes.copy()

# DIVIDINDO EM VARIÁVEL NUMÉRICA PARA REMOVER NAN E INSERIR A MEDIANA NO LUGAR

num_att = arquivo_fraudes.select_dtypes(exclude=["object", "datetime"]).columns.to_list()

imputer_mediana = SimpleImputer(strategy='median')

scaler = MinMaxScaler()

for num in num_att:
    arquivo_fraudes[num] = imputer_mediana.fit_transform(np.array(arquivo_fraudes[num]).reshape(-1, 1))
    arquivo_fraudes[num] = scaler.fit_transform(np.array(arquivo_fraudes[num]).reshape(-1, 1))

arquivo_fraudes.dropna()

# COLOCANDO COMO PARÂMETRO O VALOR DE 0.40, IREMOS RETIRAR AS COLUNAS QUE SÃO RELACIONADAS.

"""TODAS AS COLUNAS DO DF"""
"""'TIME', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21'"""

X = arquivo_fraudes
y = arquivo_fraudes["CLASS"]

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']
print(featureScores.nlargest(20, 'Score'))

""""TOP 10 COLUNAS COM MAIOR PONTUAÇÃO DO DF"""
"""    Specs       Score
30  CLASS  492.000000
4      V4   53.974577
11    V11   44.133827
14    V14   31.839693
12    V12   28.666539
16    V16   20.267755
17    V17   15.792533
3      V3   12.678611
10    V10   11.473454
18    V18   11.305559"""

# DEFININDO X E Y E ESCOLHENDO AS COLUNAS PARA AS VARIÁVEIS

X = arquivo_fraudes[["V4", "V11", "V16", "V3", "V10", "V18", "V9"]]
y = arquivo_fraudes[["CLASS"]]

# DIVIDINDO AS VARIÁVEIS EM VARIÁVEL DE TREINO E DE TESTE

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, shuffle=True)

# CRIANDO UMA LISTA COM OS TIPOS DE MODELOS QUE IRÃO SER UTILIZADOS

lista_modelos = [RandomForestClassifier(random_state=1), LogisticRegression(random_state=1),
                 DecisionTreeClassifier(), GaussianNB()]

for tipo_modelo in lista_modelos:
    retorna_resultado_modelo(tipo_modelo)


# REALIZANDO O TESTE AGORA NA BASE INTEIRA DO ARQUIVO PRINCIPAL
# DEFININDO X E Y E ESCOLHENDO AS COLUNAS PARA AS VARIÁVEIS

X = arquivo[["V4", "V11", "V16", "V3", "V10", "V18", "V9"]]
y = arquivo[["CLASS"]]

# FAZENDO O MODELO COM O ALGORÍTMO DE RANDOM FOREST POR TER DADO O MELHOR RESULTADO

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, shuffle=True)

modelo = GaussianNB().fit(X_train, y_train)
y_pred = modelo.predict(X_test)

recall_final = recall_score(y_test, y_pred, average='macro')
acuracia_final = accuracy_score(y_test, y_pred)

fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

confusion_matrix(y_test, y_pred)
