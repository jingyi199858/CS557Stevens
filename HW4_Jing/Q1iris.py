from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0).fit(X, y)
clf.predict(X[:2, :])
pro = clf.predict_proba(X[:2, :])
accuracy = clf.score(X, y)
print("accuracy is: ", accuracy)
print("probability is: ", pro)
