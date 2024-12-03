import kagglehub
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, max_error

pd.options.display.max_columns = None

path = kagglehub.dataset_download("neuromusic/avocado-prices")

df = pd.read_csv(path)

df["type"] = df["type"].replace("conventional", 1).replace("organic", 0)

ano_unico = df["year"].unique()

total_volume = df.groupby(["year"]).sum(["Total Volume"])

plt.title("Avocado Sold by Year")
plt.xlabel("Year")
plt.ylabel("Total Quantity (Billion)")
plt.scatter(ano_unico, total_volume["Total Volume"])
plt.plot(ano_unico, total_volume["Total Volume"])
plt.xticks(range(min(ano_unico), max(ano_unico), 1))
plt.show()

gby_consumer = df.groupby('region')['Total Volume'].sum().reset_index()
gby_consumer = gby_consumer[gby_consumer['region'] != 'TotalUS']
gby_consumer = gby_consumer.sort_values(by='Total Volume', ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.barh(gby_consumer['region'], gby_consumer['Total Volume'])
plt.xlabel('Total Volume (Billion)')
plt.ylabel('Region')
plt.title('Top 10 Region Consumers')
plt.show()

le = LabelEncoder()
df['region'] = le.fit_transform(df['region'])

X = df[['type', 'region', 'Total Volume']]
y = df["AveragePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, shuffle=True)

model = RandomForestRegressor(random_state=1)

trained_model = model.fit(X_train, y_train)

prediction = trained_model.predict(X_test)

mse = mean_squared_error(y_test, prediction)
r2 = r2_score(y_test, prediction)
max_erro = max_error(y_test, prediction)

print(f"MSE: {mse}\nMAE: {mae}\nR2: {r2}\nMAX ERROR: {max_erro}\n")
