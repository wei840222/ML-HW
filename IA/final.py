# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 15:07:45 2017

@author: wei84
"""

'''數據預處理'''
import numpy as np
import pandas as pd

# 載入資料集
#dataset = pd.read_csv('bank.csv')
dataset = pd.read_csv('bank-full.csv')
X = dataset.iloc[:, :9].values
y = dataset.iloc[:, 16].values

# 針對類別資訊進行獨熱編碼
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])
X[:, 2] = labelencoder.fit_transform(X[:, 2])
X[:, 3] = labelencoder.fit_transform(X[:, 3])
X[:, 4] = labelencoder.fit_transform(X[:, 4])
X[:, 6] = labelencoder.fit_transform(X[:, 6])
X[:, 7] = labelencoder.fit_transform(X[:, 7])
X[:, 8] = labelencoder.fit_transform(X[:, 8])
y = np.array(y == 'yes', np.int32)

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
onehotencoder = OneHotEncoder(categorical_features=[13])
X = onehotencoder.fit_transform(X).toarray()
onehotencoder = OneHotEncoder(categorical_features=[16])
X = onehotencoder.fit_transform(X).toarray()
onehotencoder = OneHotEncoder(categorical_features=[24])
X = onehotencoder.fit_transform(X).toarray()


# 分割資料為訓練資料集與測試資料集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 特徵縮放
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


'''建置ANN'''
from keras.models import Sequential
from keras.layers import Dense

# 定義循序模型
classifier = Sequential()

# 定義輸入層與第一隱藏層
classifier.add(Dense(units=14, kernel_initializer='uniform', activation='sigmoid', input_dim=27))

# 定義第二隱藏層
classifier.add(Dense(units=14, kernel_initializer='uniform', activation='sigmoid'))

# 定義輸出層
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# 編譯 ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 訓練 ANN
classifier.fit(X_train, y_train, batch_size=10, epochs=100)


'''使用ANN預測'''
from sklearn.metrics import confusion_matrix

# 從測試資料得到預測結果
y_pred = classifier.predict(X_test)
y_pred = np.array(y_pred >= 0.5, np.int32)

# 得到準確率矩陣輸出結果
cm = confusion_matrix(y_test, y_pred)
print('Accuracy in test set: ' + str((cm[0,0]+cm[1,1])/np.sum(cm)))