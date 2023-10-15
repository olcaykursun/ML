import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle

from matplotlib import use as set_backend
set_backend('QtAgg') # To set the backend to QtAgg, you can also use `%matplotlib qt` as opposed to `%matplotlib inline`


# Function to plot the data and decision boundary
def plot_data_and_boundary(data, target, clf=None):
    plt.scatter(data[:, 0], data[:, 1], c=target)
    plt.xlabel('x1')
    plt.ylabel('x2')
    x_min, x_max = data[:, 0].min() - .5, data[:, 0].max() + .5
    y_min, y_max = data[:, 1].min() - .5, data[:, 1].max() + .5
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    if clf is not None:
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    plt.pause(0.5)

# Generate synthetic data
d = 2  # number of dimensions
n_samples = 100
cluster_std = 0.1
mean0 = [0, 0]
mean1 = [1, 0]
mean2 = [1, 1]
mean3 = [0, 1]
pos_class = 2  # which center is positive?

data, target = make_blobs(n_samples=n_samples, n_features=2, 
      centers=[mean0, mean1, mean2, mean3], cluster_std=cluster_std, random_state=1)

# Map target values to binary (0 or 1)
target = (target == pos_class).astype(int)

plot_data_and_boundary(data, target)

# Linear perceptron using scikit-learn
Nepochs = 50
clf = SGDClassifier(random_state=21, eta0=1e-2, loss='log_loss', penalty=None, learning_rate='constant', warm_start=True)

# Initialize variables for storing progress
accuracies = []
losses = []

for epoch in range(Nepochs + 2):
    # Fit the model
    if epoch == 0:
        # Shuffle the data to randomize the order
        shuffled_data = shuffle(data)
        clf.partial_fit(shuffled_data, target, classes = [0, 1])
    elif epoch == Nepochs+1:
        pass
    else:
        clf.partial_fit(data, target)
        
    plot_data_and_boundary(data, target, clf)
            
    # Accuracy
    predictions = clf.predict(data)
    accuracy = accuracy_score(target, predictions)    
    accuracies.append(accuracy)
    
    # Calculate log loss (cross-entropy) based on predicted class probabilities
    predicted_probabilities = clf.predict_proba(data)
    loss = log_loss(target, predicted_probabilities)
    losses.append(loss)  

    print(f'{epoch=}, {accuracy=}, {loss=}')  
    
#%%
# Plot the accuracy/error values
set_backend('module://matplotlib_inline.backend_inline') 

plt.figure()
plt.plot(range(Nepochs + 2), losses, marker='o')
plt.xlabel('No of Epochs Trained')
plt.ylabel('Loss')
plt.show()

plt.figure()
plt.plot(range(Nepochs + 2), accuracies, marker='o')
plt.xlabel('No of Epochs Trained')
plt.ylabel('Accuracy (%)')
plt.show()

input("Press Enter to exit...")
