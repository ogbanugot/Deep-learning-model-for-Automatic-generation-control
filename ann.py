# Classification template

# Importing the libraries

import pandas as pd
import numpy as np

# Importing the train dataset
dataset = pd.read_csv('consumption_data2.csv')
X = dataset.iloc[:, 0:7].values
y = dataset.iloc[:, 10].values


dataset2 = pd.read_csv('training_data.csv')
X2 = dataset2.iloc[:, 0:7].values
y2 = dataset2.iloc[:, 10].values

y_train = y
y_test = y2
# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X)
X_test = sc.transform(X2)



#Evaluating improving and tuning the ANN 
#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import KFold


from keras.models  import Sequential
from keras.layers import Dense
def build_model():
   model = Sequential()
   model.add(Dense(activation = 'relu', input_dim = 7, units = 6, kernel_initializer = 'uniform',  ))
   model.add(Dense(activation = 'relu', units = 6, kernel_initializer = 'uniform', ))
   model.add(Dense(units = 1, kernel_initializer = 'uniform', ))
   model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mse','mae'])
   return model


# fix random seed for reproducibility
#seed = 7
#np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=build_model, epochs=100, batch_size=10)
kfold = KFold(n_splits=10, shuffle=False, random_state= None)


estimator.fit(X_train, y_train, batch_size = 10, epochs = 100)
results = cross_val_score(estimator,X_train, y_train, cv=kfold)
prediction = cross_val_predict(estimator, X_test, y_test, cv=kfold)

