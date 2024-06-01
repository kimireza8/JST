import pandas as pd
from sklearn.neural_network import MLPClassifier as MLP

dataset = pd.read_csv('XOR.csv')
data = dataset.iloc[:,0:-1]
label = dataset.iloc[:,-1]

model = MLP(hidden_layer_sizes=(4), max_iter=10000, activation = 'relu', learning_rate_init=0.1, solver='sgd', random_state=99)
model.fit(data,label)

print('score:', model.score(data,label))
print('predictions:', model.predict(data))
print('expected:',label)