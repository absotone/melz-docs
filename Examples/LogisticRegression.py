import importlib
lr = importlib.import_module("absotone-melz.LogisticRegression")

"""
Training the Model

Current Accepted Methods: Gradient Descent, Newton's Method
"""

# Training Data
X = [[0.1,0.2],[0.4,-5.8]]
y = [1,0]

classifier = lr.LogisticRegression(X,y)


weights_gradient_descent = classifier.getParametersGradientDescent(
    learningRate = 0.00001, # Learning Rate of the Algorithm
    numInterations = 10000, # Number of iterations of the Learning Algorithm
    decay = 0 # Decay of the Learning Rate of the Algorithm
)


weights_newton_method = classifier.getParametersGradientDescent(
    numInterations = 10000 # Number of iterations of the Learning Algorithm
)

"""
Get Training Accuracy

The output format is:
{
    "accuracy" = accuracy,
    "tp" = true_positives, 
    "tn" = true_negatives, 
    "fp" = false_positives,
    "fn" = false_negatives 
}
"""

accuracyDict_gradient_descent = classifier.getAccuracyTraining(weights_gradient_descent)

accuracyDict_newton_method = classifier.getAccuracyTraining(weights_newton_method)

"""
Testing the Model

The output format is:
{
    "accuracy" = accuracy,
    "tp" = true_positives,
    "tn" = true_negatives, 
    "fp" = false_positives,
    "fn" = false_negatives
}
"""

# Testing Data

x_test = [[0.4,0.2],[0.4,-0.5]]
y_test = [1,0]

test_accuracyDict_gradient_descent = classifier.getTestingAccuracy(x = x_test, y = y_test, weights = weights_gradient_descent)
test_accuracyDict_newton_method = classifier.getTestingAccuracy(x = x_test, y = y_test, weights = weights_newton_method)
