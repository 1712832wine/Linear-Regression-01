import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#Read Data
##############
x= []
y= []
with open('X_train.csv','rt') as f:
  data1 = csv.reader(f)
  for i in data1:
    x.append(i)
with open('Y_train.csv','rt') as g:
  data2 = csv.reader(g)
  for i in data2:
    y.append(i)

a = []
with open('X_test.csv','rt') as h:
  data3 = csv.reader(h)
  for i in data3:
    a.append(i)


#Pre Processing
x=np.array(x)[1:,1:]                                         #Del First row and first col
a=np.array(a)[1:,1:]                                         #Del First row and first col

y = np.array(y)[1:,1:].flatten().astype(np.float)            #Del First row and first col
index2 = list(range(0,22))
labelencoder = LabelEncoder()
for i in index2:
  labelencoder = labelencoder.fit(x[:,i])
  x[:,i] = labelencoder.transform(x[:,i])
  a[:,i] = labelencoder.transform(a[:,i])
x = x.astype(np.int)
a = a.astype(np.int)

onehot_encoder = OneHotEncoder(dtype=int)
onehot_encoder = onehot_encoder.fit(x)
x = onehot_encoder.transform(x)
a = onehot_encoder.transform(a)
#print(a)
#Linear LinearRegression
model = LinearRegression().fit(x, y)
print('score:',model.score(x,y))
print('intercept_:',model.intercept_)
print('coef_:',model.coef_)


##Predict
print(model.predict(a))
