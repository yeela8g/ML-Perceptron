



# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive

drive.mount('/content/drive', force_remount=True)
#!ls "/content/drive/My Drive"

# Enter the foldername in your Drive where you have saved the code and datasets.
# Recommended path: 'machine_learning_intro/assignments/assignment1/'
FOLDERNAME = 'machine_learning_intro/assignments/'
ASSIGNMENTNAME = 'assignment1'

# %cd drive/My\ Drive
# %cp -r $FOLDERNAME/$ASSIGNMENTNAME ../../
# %cd ../../

"""### **3. Dataset & Preprocessing**
"""

# Load numpy package
import numpy as np

"""Load the pre-generated data provided to you. Using numpy, load the file "dataset.csv" and print its shape. You should see that the data is a numpy array (a matrix) with 500 rows (called **data samples**) and 3 columns. The first two columns are the **features** of the samples and the last column is the **label** of each sample."""

# load data using "np.genfromtxt"
data = np.genfromtxt(f"{ASSIGNMENTNAME}/dataset.csv", delimiter=',',dtype=np.float64)


"""Split the data into features and labels and print their shape. Be careful not to change the content of the data."""

features = data[:,0:2]
labels = data[:,2]

"""now split the data into train and test """

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=36)


"""### ** Perceptron**

**Handle the bias term:**
We add a column of ones to the train and test features so we won't need to learn the bias. The train and test shapes will be (#samples,3). There are plenty of ways to do it. One way is using ```np.hstack``` function.
"""

def add_ones_column(X):
  # add a column of ones to the data
  ones_col = np.ones((X.shape[0],1))
  print(np.hstack((ones_col,X)).shape)
  return np.hstack((ones_col,X))


X_train = add_ones_column(X_train)
X_test = add_ones_column(X_test)

"""Implement all methods of the Perceptron class below."""

class Perceptron(object):
    def __init__(self, n_features, iterations=10, learning_rate=0.01):
        '''
        The function initialized the Perceptron model.
        n_features - number of features of each sample (excluding the bias)
        iterations - number of iterations on the training data
        learning_rate - learning rate, how much the weight will change during update
        '''
        self.iterations = iterations
        self.learning_rate = learning_rate
        np.random.seed(30) # set random seed, should not be altered!
        self.weights = np.random.randn(n_features + 1) #intialize random W


    def predict(self, input):
        '''
        The function makes a prediction for the given input.
        Output: -1 or 1.
        '''
        return np.sign(np.dot(input,self.weights))


    def evaluate(self, inputs, labels):
        '''
        The function makes a predictions for the given inputs and compares
        against the labels (ground truth). It returns the accuracy.
        Accuracy = #correct_classification / #total
        '''
        yPredicted = []
        for input in inputs:
          yPredicted.append(self.predict(input))
        yPredictedNp = np.array(yPredicted)
        return np.sum(yPredictedNp == labels) / len(labels)

    def train(self, training_inputs, train_labels, test_inputs, test_labels, verbose=True):
        '''
        The function train a perceptron model given training_inputs and train_labels.
        It also evaluates the model on the train set and test set after every iteration.
        '''

        for i in range(self.iterations):
            for x, y in zip(training_inputs, train_labels):
              if y != self.predict(x):
                self.weights += self.learning_rate * y * x


            if verbose:
              print(f"Iteration No.{i},\
               Train accuracy: {self.evaluate(training_inputs, train_labels)},\
                Test accuracy: {self.evaluate(test_inputs, test_labels)}")


# create a Perceptron model and train it.

model = Perceptron(len(X_train[0,:]) -1, 10, 0.01)
model.train(X_train, y_train, X_test, y_test)


