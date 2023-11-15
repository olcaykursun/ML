import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Initialize and train the Logistic Regression model
logreg = LogisticRegression(max_iter=200)
logreg.fit(X, y)

# Logistic regression optimization iteratively adjusts parameters to maximize log-likelihood,
# targeting where the first derivative (gradient) is zero. Different solvers, like lbfgs, newton-cg, 
# liblinear, sag, and saga, apply distinct methods to estimate and leverage derivatives. Dataset specifics, 
# feature characteristics, and numerical precision influence convergence. The 'max_iter' parameter sets a 
# ceiling on the number of iterations, terminating the algorithm when it has adequately approximated the 
# log-likelihood's maximum, thus ensuring a balance between search precision and computational efficiency 
# across varying solvers. For instance, the Newton-Raphson method, by seeking parameter values that zero 
# the first derivative, resembles a binary search in refining the interval of search via the second derivative 
# to inform step size adjustments.

# After optimization, we can interpret the coefficients of the learned logistic regression model:
# The "weights" array is of shape 3-by-4 because there are 3 classes and 4 features in the dataset.
# exp(weights[c][i]) gives the odds ratio for the i-th feature in discriminating the c-th class against all other classes.
# This implies that for a one-unit increase in the i-th feature, the odds of the sample being in class c (as opposed to any other class) are multiplied by exp(weights[c][i]).
# An exp(weights[c][i]) value greater than 1 indicates that the odds increase with the feature's value; if it is less than 1, the odds decrease.
weights = logreg.coef_

odds_ratios = np.exp(weights)
# The raw weights in logistic regression quantify the additive change in log-odds for a one-unit increase in the corresponding feature value. However, to capture the multiplicative effect on the odds themselves, we exponentiate the weights. 
# This transformation shifts the interpretation from an additive change in log-odds to a multiplicative change in the actual odds ratio for each feature as previously detailed above.

# Print the weights and odds ratios for each class
for i, class_label in enumerate(logreg.classes_):
    print(f"Class {iris.target_names[class_label]}:")
    print(f"Weights: {weights[i]}")
    print(f"Odds Ratios: {odds_ratios[i]}")
    print()
