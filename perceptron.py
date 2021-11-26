import numpy as np
X = np.array([[6,1,-11],[-5,6,7],[-5,-11,1],
            [-11,3,5],[-5,-2,1],[5,-6,-7],
            [2,0,2],[4,-9,-11],[9,-7,-9],[-8,4,-8]])
t = np.array([0,1,0,0,1,1,1,0,0,1])
w = np.array([-0.1,-0.3,-0.2])
b = 2
y = np.dot(X,w) + b
for i in range(len(X)):
    activation = np.dot(X[i],w) + b
    prediction_for_sample = 1
    if activation<0:
        prediction_for_sample = 0
    err = t[i] - prediction_for_sample
    print(err)
    w = w + 0.02*X[i]*err
    b = b + 0.02*err