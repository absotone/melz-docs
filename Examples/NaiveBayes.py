import importlib
lr = importlib.import_module("absotone-melz.NaiveBayes")

"""
Training the Model

Current Accepted Methods: Bayes' Theorem
"""

# Training Data
X = [[0.1,0.2],[0.4,-5.8]]
y = [1,0]

classifier = lr.NaiveBayes(X,y)


class_labels = classifier.getClassLabels()



"""
Get Training Accuracy
"""

accuracy_training = classifier.getAccuracy(class_labels)

"""
Get testing accuracy
"""

# Testing Data

x_test = [[0.4,0.2],[0.4,-0.5]]
y_test = [1,0]

accuracy_testing = classifier.getAccuracyTesting(x = x_test, y = y_test)
