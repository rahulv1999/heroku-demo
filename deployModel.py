import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

train = pd.read_csv('C:/Users/RV/Downloads/depolydata.csv')
X = train.drop('salary',axis =1)
y = train['salary']
X = np.array(X)

model = LinearRegression()
model.fit(X,y)
print(model.predict(X))
k = np.array([1,2,3])
k = k.reshape(3,-1)
print(k)
pickle.dump(model,open('ML.pkl','wb'))

model = pickle.load(open('ML.pkl','rb'))

model.predict(X)