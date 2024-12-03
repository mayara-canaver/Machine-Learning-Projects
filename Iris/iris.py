import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, recall_score

iris = load_iris()
iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

decision_model = tree.DecisionTreeClassifier(random_state=1)

trained_model = decision_model.fit(X_train, y_train)
predictions = trained_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions) * 100
recall = recall_score(y_test, predictions, average='micro') * 100

print(f"Model accuracy %{accuracy:.2f} and recall %{recall:.2f}")
