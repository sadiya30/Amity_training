
import tensorflow
from tensorflow import keras
import numpy
import math
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import sklearn
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("final50hc16f168.csv", header=None)
print(dataset.shape)
X = dataset.loc[:,0:2672]
y = dataset.loc[:,2673]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

#print(X)
#print(y)
print(X_train.shape)
print(X_test.shape)

model = Sequential()
model.add(Dense(3000, input_dim =  2673, activation = 'relu'))
model.add(Dense(500, activation  = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train,y_train, epochs = 100, batch_size = 25)

accuracy = model.predict(X_test)
#print(accuracy)
#print("\n%s: %.2f%%" % (model.metrics_names[1], accuracy[1]*100))

correct=0
print(len(y_test))
Y = y_test.tolist()
for i in range (0,len(y_test),1):
   if(accuracy[i] >= 0.5):
      p = 1
   else:
      p=0 
   if(p == Y[i]):
      correct=correct+1
print("correct: ",correct," Total: ",len(Y)," : ",correct/len(Y))
