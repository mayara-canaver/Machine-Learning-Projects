import pandas as pd
import csv
import seaborn as sns

from scipy.stats import shapiro
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller


def trata_coluna(data):
    data.columns = [x.upper() for x in data.columns]

    for coluna in data:
        data[coluna] = data[coluna].str.strip()
        data[coluna] = data[coluna].str.replace('"', '')
        data[coluna] = data[coluna].str.upper()


d_shampoo = "C:/Users/Mayara Lopes/Desktop/shampoo_sales.csv"

shampoo = pd.read_csv(d_shampoo, sep=",", encoding="UTF-8", dtype=str, error_bad_lines=False, quoting=csv.QUOTE_NONE)

trata_coluna(shampoo)

shampoo = shampoo.rename({'"MONTH"': "MONTH", '"SALES"': "SALES"}, axis=1)

shampoo["SALES"] = pd.to_numeric(shampoo["SALES"])

shampoo["MONTH"] = "0" + shampoo["MONTH"]
shampoo["MONTH"] = pd.to_datetime(shampoo["MONTH"], format="%y-%m", yearfirst=True)

#perform Shapiro-Wilk test
shapiro(shampoo["SALES"])

sns.lineplot(data=shampoo, x="MONTH", y="SALES")


plot_acf(shampoo["SALES"].values.astype("float32"))

X = shampoo["SALES"].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

resultado = seasonal_decompose(shampoo["SALES"], model='additive', period=5)
resultado.plot()

rolling = shampoo.rolling(window=3)
rolling_mean = rolling.mean()
shampoo["MEDIAS MOVEIS"] = rolling_mean
sns.lineplot(data=shampoo, x="MONTH", y="SALES")
sns.lineplot(data=shampoo, x="MONTH", y="MEDIAS MOVEIS")


def difference(dataset, interval):
	diff = list()

	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)

	return pd.Series(diff)


diferenciacao = difference(shampoo["SALES"], 1)
shampoo["DIFERENCIACAO"] = diferenciacao
sns.lineplot(data=shampoo, x="MONTH", y="DIFERENCIACAO")

shampoo_treino = shampoo[:24]
shampoo_teste = shampoo[24:]

modelo = ARIMA(shampoo_treino["SALES"], order=[2, 1, 3])
resultado = modelo.fit().forecast(12)
shampoo_teste["PREVISAO"] = resultado

sns.lineplot(data=shampoo_treino, x="MONTH", y="SALES")
sns.lineplot(data=shampoo_teste, x="MONTH", y="PREVISAO")
sns.lineplot(data=shampoo_teste, x="MONTH", y="SALES")

modelo_auto_arima = auto_arima(shampoo_treino["SALES"].values, error_action="ignore", trace=True)

previsao_auto_arima = modelo_auto_arima.predict(12)
shampoo_teste["PREVISAO_AUTO_ARIMA"] = previsao_auto_arima
sns.lineplot(data=shampoo_teste, x="MONTH", y="PREVISAO_AUTO_ARIMA")
sns.lineplot(data=shampoo_teste, x="MONTH", y="SALES")

resultado_auto_arima = mean_squared_error(shampoo_teste["SALES"], shampoo_teste["PREVISAO_AUTO_ARIMA"])
resultado_arima = mean_squared_error(shampoo_teste["SALES"], shampoo_teste["PREVISAO"])

print(f"AUTO ARIMA = {resultado_auto_arima}")
print(f"ARIMA = {resultado_arima}")
