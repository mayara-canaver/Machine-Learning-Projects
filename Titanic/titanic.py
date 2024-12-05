import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

df = pd.read_csv("train.csv")
df = df.dropna()

sex_map = {"male": 1, "female": 0}
df["Sex"] = df["Sex"].map(sex_map)

embarked_map = {"Q": 1, "S": 1, "C": 0}
df["Embarked"] = df["Embarked"].map(embarked_map)

df["Name"] = df["Name"].str.extract(r",(.*)\.")
df["Name"] = df["Name"].str.replace(" ","")

name_map = {"Mrs": 0, "Mr": 1, "Miss": 2, "Other": 3}
df["Name"] = df["Name"].map(name_map).fillna(4)

X = df[["Pclass", "Sex", "Embarked", "Name"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.8)

model = DecisionTreeClassifier(random_state=1)

model_trained = model.fit(X_train, y_train)

prediction = model_trained.predict(X_test)

acuracia = accuracy_score(y_test, prediction)
recall = recall_score(y_test, prediction, average='macro')

print(f"""Acuracy %{acuracia}, Recall %{recall}""")
