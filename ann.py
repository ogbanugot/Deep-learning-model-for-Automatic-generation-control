# Classification template

# Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from keras.models import model_from_json


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
from keras.layers import Dropout
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold


from keras.models  import Sequential
from keras.layers import Dense
def build_model():
   model = Sequential()
   model.add(Dense(activation = 'relu', input_dim = 7, units = 6, kernel_initializer = 'uniform',  ))
   model.add(Dropout(rate = 0.2))
   model.add(Dense(activation = 'relu', units = 6, kernel_initializer = 'uniform', ))
   model.add(Dropout(rate = 0.2))
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
prediction = cross_val_predict(estimator, X_test, y_test, cv=kfold, n_jobs=-1)

prediction = estimator.predict(X_test)

#Variants of scoring
msle = mean_squared_log_error(y_test, prediction)
mse = mean_squared_error(y_test,prediction)
mae = mean_absolute_error(y_test,prediction)
r2 = r2_score(y_test,prediction)
from math import sqrt
rmse = sqrt(mse) #root mean --

#Model visualization

plt.plot(y_test, color = 'red', label = 'Real load')
plt.plot(prediction, color = 'blue', label = 'Predicted load')
plt.title('Time lapse interval predictions')
plt.xlabel('Frequency')
plt.ylabel('Time lapse')
plt.legend()
plt.show()

from keras.utils import plot_model
plot_model(estimator.model, to_file='model.png')

# serialize model to JSON
model_json = estimator.model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
estimator.model.save_weights("model.h5")
print("Saved model to disk")

# later...

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model 
loaded_model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mse','mae'])

