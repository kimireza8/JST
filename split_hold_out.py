import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as MLP


dataset = pd.read_csv('heartt.csv')
data = dataset.iloc[:,0:-1]
datalabel = dataset.loc[:,['target']]

xtrain, xtest, ytrain, ytest = train_test_split(data, datalabel, test_size=0.30, random_state=100)

model = MLP(hidden_layer_sizes=(9), max_iter=999, activation = 'relu', solver='lbfgs', random_state=9)
model.fit(xtrain,ytrain.values.ravel())

print('score training:', model.score(xtrain,ytrain))

print('score testing:', model.score(xtest,ytest))
