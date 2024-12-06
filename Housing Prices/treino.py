import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error


df = pd.read_csv("train.csv")

le = LabelEncoder()

df['SaleCondition'] = le.fit_transform(df['SaleCondition'])
df['Foundation'] = le.fit_transform(df['Foundation'])

df = df[["SaleCondition", "GarageCars", "Fireplaces", "FullBath", "Foundation", "SalePrice"]]

X = df.loc[:, df.columns != "SalePrice"]
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, shuffle=True)

model = RandomForestRegressor(random_state=1)

trained_model = model.fit(X_train, y_train)
predict = trained_model.predict(X_test)

mse = mean_squared_error(y_test, predict)
mae = mean_absolute_error(y_test, predict)
r2 = r2_score(predict, predict)
max_erro = max_error(y_test, predict)

print(f"MSE: {mse}\nMAE: {mae}\nR2: {r2}\nMAX ERROR: {max_erro}\n")