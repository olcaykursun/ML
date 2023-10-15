# Linear Perceptron: Implementation follows the approach described on Slide 10 of:
# https://www.cmpe.boun.edu.tr/~ethem/i2ml3e/3e_v1-0/i2ml3e-chap11.pdf

import numpy as np
import matplotlib.pyplot as plt
from math import exp
from sklearn.datasets import make_blobs
from matplotlib import use as set_backend

set_backend('QtAgg') # To set the backend to QtAgg, you can also use `%matplotlib qt` as opposed to `%matplotlib inline`

def sigmoid(a):
    return 1 / (1 + exp(-a))

# Plot the training points
def plot_data_2D(data, labels):
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.xlabel('x1')
    plt.ylabel('x2')
    x_min, x_max = data[:, 0].min() - .5, data[:, 0].max() + .5
    y_min, y_max = data[:, 1].min() - .5, data[:, 1].max() + .5
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.pause(0.5)
    
def plot_linear_discriminant(w):
    xx = np.linspace(-5, 5)
    yy = (-w[0] - xx * w[1]) / w[2]
    plt.plot(xx, yy, 'k-')
    plt.pause(0.5)

#%%
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

pos_examples = target == pos_class
neg_examples = target != pos_class
target[pos_examples] = 1
target[neg_examples] = 0

plot_data_2D(data, target)

#%%
#linear perceptron
Nepochs = 50
eta = 0.01  # learning rate
train_x = np.insert(data, 0, 1, axis=1)

init_randomly = False

if init_randomly:
    rng = np.random.RandomState(1234567)
    w = rng.randn(d+1) / 5  # 1D is sufficient until we add a hidden layer with hidden units
else:
    w = np.array([1, 1, -1]) #a bad start to see how it would progress..
    
plot_linear_discriminant(w) #show the initial one, can be deliberately selected as a bad start by the previous line

accuracies = [] #to keep track of accuracies as a function of epochs
losses = [] #to keep track of loss as a function of epochs
for epoch in range(Nepochs + 2):  # No of times to go over the data (no updates in the first and last epochs to see accuracies)
    numcorr = 0  #number of correct classifications needed for accuracy calculation
    total_error = 0  #negative-log-likelihood needed for error calculation
    for t in range(n_samples):
        r = target[t]
        x = train_x[t, :]
        o = np.dot(x, w)
        y = sigmoid(o)
        if (y > 0.5 and r == 1) or (y <= 0.5 and r == 0):
            numcorr = numcorr + 1  # this calculates the training accuracy
            
        E_t = -r * np.log2(y) - (1 - r) * np.log2(1 - y) #negative log-likelihood or cross-entropy between labels and predicted posteriors
        total_error = total_error + E_t   #accumulate total error

        # update except the first and last epochs not to affect the accuracy&error
        if epoch not in [0, Nepochs + 1]:
            delta = r - y
            w = w + eta * delta * x
            
    # Accuracy and Error is expected to reduce with updates
    accuracy = numcorr / n_samples
    accuracies.append(accuracy)
    loss = total_error / n_samples
    losses.append(loss)    
    print(f'{epoch=}, {accuracy=}, {loss}')  
    
    if epoch % 2 == 1: #plot it once every 2 epochs
        plot_linear_discriminant(w)

#%%
# Plot the accuracy/error values

set_backend('module://matplotlib_inline.backend_inline') 
# You can set the Matplotlib backend either programmatically or by typing the magic command.
# Programmatically:
# set_backend('module://matplotlib_inline.backend_inline')
# Magic command (for Jupyter Notebook) in console:
# %matplotlib inline

plt.figure()
plt.plot(range(Nepochs + 2), losses, marker='o')
plt.xlabel('No of Epochs Trained')
plt.ylabel('Loss')
plt.show(block=True)

plt.figure()
plt.plot(range(Nepochs + 2), accuracies, marker='o')
plt.xlabel('No of Epochs Trained')
plt.ylabel('Accuracy (%)')
plt.show(block=True)

input("Press Enter to exit...")
