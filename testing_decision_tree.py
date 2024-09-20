import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import decision_tree_python_code as dt


# get the data and print sample for verificiation
df = pd.read_csv('airfoil_noise_data.csv')
print(df.sample())

# y is some noise created by the airfoil.
# airfloil is a kind of a wing of the plane.
# features are like
    #  angle of wind, 
    # speed of wind, 
    # length of wing, etc.



# split the data
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y.reshape(-1,1), test_size=0.2, random_state= 41)


# fit the model
model_1 = dt.DecisionTreeRegressor()
model_1.fit(X_train, y_train)
model_1.print_tree()

