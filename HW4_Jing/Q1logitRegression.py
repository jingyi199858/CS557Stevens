import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_excel('RFMdataMPJ.xlsx')
X = df.drop('Respond', axis=1)
y = df['Respond']
clf = LogisticRegression()
clf.fit(X, y)
pro = clf.predict_proba(X)
score = clf.score(X, y)
print(clf.predict(X))
print("accuracy：", score)
print('probility from training set:(first value is 1, second is 0)')
print(pro)


