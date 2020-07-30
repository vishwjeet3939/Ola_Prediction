# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

#Upload dataset here
dataset = pd.read_csv("uber.csv")
#print(dataset.head())
x = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,-1].values
#print(x)
print(y)
# splitting dataset into training and testing set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 0)

# LinearRegression module import
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)

print("Train Score : ", reg.score(x_train,y_train))
print("Test Score : ",reg.score(x_test,y_test))

pickle.dump(reg, open('uber.pkl','wb'))
model = pickle.load(open("uber.pkl","rb"))
print(model.predict([[63,1610000,16200,200]]))
