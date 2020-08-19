import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn import metrics                             ## dùng để đánh giá
from sklearn.model_selection import train_test_split    ## dùng để tách dữ liệu

# Read Data
def Read(fpath):
  data = pd.read_csv(fpath)
  return data

# PreProcessing
# Data
def Data_PreProc(x):
  x = np.array(x)[0:, 1:]  # Del First row and first col'
  return x

# Label
def Label_PreProc(y):
  y = np.array(y)[0:, 1:].flatten().astype(np.float)  # Del First row and first col
  return y

# Data Encoder
def Data_Encoder(x,a):
  index2 = list(range(0, 21))
  labelencoder = LabelEncoder()
  for i in index2:
      labelencoder = labelencoder.fit(x[:, i])
      x[:, i] = labelencoder.transform(x[:, i])
      a[:, i] = labelencoder.transform(a[:, i])
  x = x.astype(np.float)
  a = a.astype(np.float)
  onehot_encoder = OneHotEncoder(dtype=float)
  onehot_encoder = onehot_encoder.fit(x)
  x = onehot_encoder.transform(x)
  a = onehot_encoder.transform(a)
  return x,a

# Train
def LinearReg(x,y):
  model = LinearRegression().fit(x, y)
  return model




# CÁC BƯỚC THỰC HIỆN LINEAR REGRESSION
# BƯỚC 1: ĐỌC DỮ LIỆU TRAIN(DATA TRAIN, LABEL TRAIN)
# BẰNG PHƯƠNG THỨC: Read()
x_train = Read('X_train.csv')  # data train
y_train = Read('Y_train.csv')  # label train

# ĐÁNH GIÁ DỮ LIỆU
print(x_train.info())
x_train = x_train.drop('engineCapacity', axis=1, inplace=False)
print(x_train.info())

tempx, x_test, tempy, y_test = train_test_split(x_train, y_train, test_size=0.2)

#x_test = Read('X_test.csv') # data test
#x_test= x_test.drop('engineCapacity', axis=1, inplace=False)
#x_train = x_train.drop('feature_0', axis=1, inplace=False)
#x_train = x_train.drop('odometer', axis=1, inplace=False)
#y_test = Read('Y_test.csv') # label test




# BƯỚC 2: TIỀN XỬ LÍ (DATA TRAIN, LABEL TRAIN, DATA TEST)
# BẰNG PHƯƠNG THỨC: Data_PreProc() VÀ Label_PreProc()
x_train = Data_PreProc(x_train)
y_train = Label_PreProc(y_train)
x_test = Data_PreProc(x_test)
y_test = Label_PreProc(y_test)

# BƯỚC 3: ENCODER DATA (DATA TRAIN, DATA TEST)
# BẰNG PHƯƠNG THỨC: Data_Encoder()
print(x_train.shape)
print(x_test.shape)
x_train,x_test = Data_Encoder(x_train,x_test)

# BƯỚC 4: TRAIN DATA VÀ DỰ ĐOÁN KẾT QUẢ (DATA TRAIN, LABEL TRAIN)
# BẰNG PHƯƠNG THỨC: LinearReg()
model = LinearReg(x_train,y_train)
print('intercept:',model.intercept_)
print('coef:',model.coef_)

# DỰ ĐOÁN GIÁ CỦA 10 DỮ LIỆU ĐẦU TRONG FILE X_TRAIN
label_predict = model.predict(x_test)
print('predict:',label_predict)

# ĐÁNH GIÁ TRUNG BÌNH DIỆN TÍCH SAI SỐ
mse = metrics.mean_squared_error(y_test, label_predict)
print('score:',model.score(x_train,y_train))
print('mse:',mse)
print('rmse:',math.sqrt(mse))

