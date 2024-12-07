import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv("Train_rev1.csv")

df["ContractType"] = df["ContractType"].fillna("unknown")
df["ContractTime"] = df["ContractTime"].fillna("unknown")

le = LabelEncoder()

df['ContractType'] = le.fit_transform(df['ContractType'])
df['ContractTime'] = le.fit_transform(df['ContractTime'])
df['Category'] = le.fit_transform(df['Category'])

X = df[["ContractTime", "ContractType", "Category"]]
y = df["SalaryNormalized"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, train_size=0.7)

model = RandomForestRegressor(random_state=1)

trained_model = model.fit(X_train, y_train)

prediction = trained_model.predict(X_test)

mse = mean_squared_error(y_test, prediction)
r2 = r2_score(y_test, prediction)
mape = mean_absolute_percentage_error(y_test, prediction)

print("MSE: %.2f\nR2: %.2f\nMAPE: %.2f\n" % (mse, r2, mape))
