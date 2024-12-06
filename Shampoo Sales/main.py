import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv("shampoo_sales.csv")

df["Month"] = "0" + df["Month"]
df["Month"] = pd.to_datetime(df["Month"], format="%y-%m")

X = df["Sales"].values

result = adfuller(X)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')

for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
	
df['Difference'] = df['Sales'].diff(periods=1)

df.dropna(inplace=True)

plot_acf(df["Difference"])

df['Moving Average'] = df['Difference'].rolling(window=3).mean()

sns.lineplot(data=df, x='Month', y='Difference', label='Difference')
sns.lineplot(data=df, x='Month', y='Moving Average', label='Moving Average')
plt.legend()
plt.show()

df_train = df[:24]
df_test = df[24:]

model = ARIMA(df_train["Difference"], order=[1, 1, 3])
predict = model.fit().forecast(12)
df_test["Predict"] = predict

sns.lineplot(data=df_train, x="Month", y="Difference", label="Actual Sales")
sns.lineplot(data=df_test, x="Month", y="Predict", label="Predicted Sales")
sns.lineplot(data=df_test, x="Month", y="Difference", label="Actual Sales (Test Set)")
plt.legend()
plt.show()

result = mean_squared_error(df_test["Difference"], df_test["Predict"])

print(f"ARIMA = {result}")