
# Import needed libraries (Numpy and matplotlib)
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


# ## Create the data
# We will create a spiral dataset.
# * https://www.quora.com/What-is-the-general-equation-of-a-2-d-spiral-in-x-y-plane-centered-at-origin

# In[2]:

# Define dimensions, number of classes, and number of points
N = 100
D = 2
K = 3

X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels

# Basically out input vector is a matrix [100x2]
print("Input shape:", X.shape)
print("Target(y) shape:", X.shape)

# Notice that in python 3 xrange is called range
for j in range(K):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

num_examples = X.shape[0]
print("Training with %d samples"%(num_examples))

# lets visualize the data:
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()

# Save data to a .mat file (Don't need to take care with the col-major, it's automatic.)
dictSaveMat={}
dictSaveMat['X']=X.astype('float')
dictSaveMat['Y']=y.astype('float')
scipy.io.savemat('ToyExample',dictSaveMat)


# ## Use a different hypothesis/model
# Now we will try a 2 layer neural network.
# * First layer (input layer): 100 neurons (Relu Activation)
# * Second layer (output layer): 3 neurons (Softmax Activation)

# In[12]:

# initialize parameters randomly
h = 100 # size of hidden layer

# Initialize weigts and bias for layers 1,2

# Load from pre-saved matfile, this will be used to debug
dictMat = scipy.io.loadmat('../../datasets/ToyExample_Init_Weights.mat')
W1 = dictMat['W1']
W2 = dictMat['W2']
b1 = dictMat['b1']
b2 = dictMat['b2']

print("W1 shape:", W1.shape)
print("W2 shape:", W2.shape)

# ### Train the neural network

# In[13]:

# some hyperparameters (gradient descent alpha and regularization)
step_size = 0.2
#reg = 1e-6 # regularization strength
reg = 0

# Run for 10000 epochs
for i in range(10000):
  
    # Forward propagation
    
    # First layer
    z = np.dot(X, W1) + b1
    hidden_layer = np.maximum(0, z)
    scores = np.dot(hidden_layer, W2) + b2

    # Second layer
    # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

    # Calculate loss
    # compute the loss: average cross-entropy loss and regularization
    corect_logprobs = -np.log(probs[range(num_examples),y])
    data_loss = np.sum(corect_logprobs)/num_examples
    reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)
    loss = data_loss + reg_loss
    if i % 1000 == 0:
        print ("iteration %d: loss %f" % (i, loss))

    # Backpropagation
    # compute the loss gradient w.r.t to scores
    dscores = probs
    dscores[range(num_examples),y] -= 1
    dscores /= num_examples

    # backpropate the gradient to the parameters
    # first backprop into parameters W2 and b2
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    # next backprop into hidden layer
    dhidden = np.dot(dscores, W2.T)
    # backprop the ReLU non-linearity
    dhidden[hidden_layer <= 0] = 0
    # finally into W,b
    dW1 = np.dot(X.T, dhidden)
    db1 = np.sum(dhidden, axis=0, keepdims=True)

    # add regularization gradient contribution
    dW2 += reg * W2
    dW1 += reg * W1

    # perform a parameter update
    W1 += -step_size * dW1
    b1 += -step_size * db1
    W2 += -step_size * dW2
    b2 += -step_size * db2


# In[14]:

# evaluate training set accuracy
hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: %.2f' % (np.mean(predicted_class == y)))

# plot the resulting classifier
h = 0.02
# Get the biggest and smallest value of X on each dimension
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# Create a meshgrid
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Do forward propagation
Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W1) + b1), W2) + b2

# Get class idx of highest score
Z = np.argmax(Z, axis=1)
# Put on the same shape as xx
Z = Z.reshape(xx.shape)

fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()

