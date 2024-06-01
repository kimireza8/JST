import pandas as pd
from sklearn.linear_model import Perceptron

dataset = pd.read_csv('XOR.csv') #ubah sesuai yang dibutuhkan
data = dataset.iloc[:,0:-1]
label = dataset.iloc[:,-1]

clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(data, label)

print(clf.score(data,label))
print(clf.predict(data))