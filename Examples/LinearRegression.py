import importlib
lr = importlib.import_module("absotone-melz.LinearRegression")

"""
Training the Model

Current Accepted Methods: Closed-Form, Gradient Descent
"""

# Training Data
X = [[0.1,0.2],[0.4,-5.8]]
y = [0.01,0.03]

regressor = lr.LinearRegression(X,y)

weights_closed_form = regressor.getParametersClosedForm()

weights_gradient_descent = regressor.getParametersGradientDescent(
    learningRate = 0.00001, # Learning Rate of the Algorithm
    numInterations = 10000, # Number of iterations of the Learning Algorithm
    decay = 0 # Decay of the Learning Rate of the Algorithm
)

"""
Get Training Accuracy

The output format is:
{
    "MeanSquaredError" : rmseError,
    "MeanAbsoluteError" : mareError
}
"""

accuracyDict_closed_form = regressor.getAccuracyTraining(weights_closed_form)

accuracyDict_gradient_descent = regressor.getAccuracyTraining(weights_gradient_descent)

"""
Testing the Model

The output format is:
{
    "MeanSquaredError" : rmseError,
    "MeanAbsoluteError" : mareError
}
"""

# Testing Data

x_test = [[0.4,0.2],[0.4,-0.5]]
y_test = [0.03,-0.07]

test_accuracyDict_closed_form = regressor.getAccuracyTesting(x = x_test, y = y_test, weights = weights_closed_form)
test_accuracyDict_gradient_descent = regressor.getAccuracyTesting(x = x_test, y = y_test, weights = weights_gradient_descent)




