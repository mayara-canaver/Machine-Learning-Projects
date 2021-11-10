# REALIZANDO IMPORTAÇÃO DE BIBLIOTECAS UTILIZADAS
import warnings
import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import TimeSeriesSplit
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error

warnings.simplefilter("ignore")
pd.options.display.max_columns = None


def trata_coluna(data):
    data.columns = [x.upper() for x in data.columns]

    for coluna in data:
        data[coluna] = data[coluna].str.strip()
        data[coluna] = data[coluna].str.replace('"', '')
        data[coluna] = data[coluna].str.upper()

# IMPORTANDO OS ARQUIVOS DE SÉRIES TEMPORAIS
d_shampoo = "C:/Users/Mayara Lopes/Desktop/shampoo_sales.csv"

shampoo = pd.read_csv(d_shampoo, sep=",", encoding="UTF-8", dtype=str, error_bad_lines=False, quoting=csv.QUOTE_NONE)

# CRIANDO UMA LISTA PARA VERIFICAR E TRATAR CADA DATASET
trata_coluna(shampoo)

shampoo = shampoo.rename({'"MONTH"': "MONTH", '"SALES"': "SALES"}, axis=1)

shampoo["SALES"] = pd.to_numeric(shampoo["SALES"])

shampoo["MONTH"] = "0" + shampoo["MONTH"]
shampoo["MONTH"] = pd.to_datetime(shampoo["MONTH"], format="%y-%m", yearfirst=True)

# VERIFICANDO A PROGRESSÃO DE PREÇO ATRAVÉS DO TEMPO
sns.lineplot(data=shampoo, x="MONTH", y="SALES")

# PLOT DA AUTO CORRELAÇÃO DOS DADOS
plot_acf(shampoo["SALES"].values.astype("float32"))

# PLOT DOS DADOS SEZONAIS
resultado = seasonal_decompose(shampoo["SALES"], model='additive', period=5)
resultado.plot()

# DIVIDINDO EM DADOS DE TREINO E TESTE
shampoo_treino = shampoo[:30]
shampoo_teste = shampoo[30:]

modelo = ARIMA(shampoo_treino["SALES"], order=[1, 0, 1])
resultado = modelo.fit().forecast(7)
shampoo_teste["PREVISAO"] = resultado

sns.lineplot(data=shampoo_treino, x="MONTH", y="SALES")
sns.lineplot(data=shampoo_teste, x="MONTH", y="PREVISAO")
sns.lineplot(data=shampoo_teste, x="MONTH", y="SALES")
plt.show()

modelo_auto_arima = auto_arima(shampoo_treino["SALES"].values, error_action="ignore", trace=True, seasonal=True, m=5)

previsao_auto_arima = modelo_auto_arima.predict(6)
shampoo_teste["PREVISAO_AUTO_ARIMA"] = previsao_auto_arima
sns.lineplot(data=shampoo_teste, x="MONTH", y="PREVISAO_AUTO_ARIMA")
sns.lineplot(data=shampoo_teste, x="MONTH", y="SALES")
plt.show()

resultado_auto_arima = mean_squared_error(shampoo_teste["SALES"], shampoo_teste["PREVISAO_AUTO_ARIMA"])
resultado_arima = mean_squared_error(shampoo_teste["SALES"], shampoo_teste["PREVISAO"])

print(f"AUTO ARIMA = f{resultado_auto_arima}")
print(f"ARIMA = f{resultado_arima}")