import sys, getopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

# get arguments
args = sys.argv[1:]

# set parameters
train_file = ''
test_file = ''
output = ''

try:
    opts, args = getopt.getopt(args, "ht:i:o:", ["train=", "test=", "output="])
except getopt.GetoptError:
    print('perceptrons.py -i <inputfile> -o <outputfile>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('test.py -i <inputfile> -o <outputfile>')
        sys.exit()
    elif opt in ("-t", "--train"):
        train_file = arg
    elif opt in ("-i", "--test"):
        test_file = arg
    elif opt in ("-o", "--output"):
        output_file = arg


class ConstantReduction:
    """Reduces learning rate by dividing by a constant factor"""

    def __init__(self, start=0.01, factor=2):
        self.rate = start
        self.factor = factor

    def get(self):
        # update rate by dividing by factor
        self.rate = self.rate / self.factor
        return self.rate


class FibonacciReduction:
    """Reduces learning rate by dividing the rate by increasing numbers
    in the Fibonacci sequence"""

    def __init__(self, start=0.01):
        self.a = 0
        self.b = 1
        self.rate = start

    def get(self):
        # get next num in sequence
        num = self.a + self.b
        # update values
        self.a = self.b
        self.b = num
        # update rate by dividing by fibonacci number
        self.rate = self.rate / num
        return self.rate


class Perceptron:
    """Perceptron implementation"""

    def __init__(self, rate=0.01, n=50, random_state=1, reduction=None):
        self.rate = rate  # learning rate
        self.n = n  # number of iterations
        self.random_state = random_state  #set random state
        self.reduction = reduction  # optional if dynamic learning rate is desired

    def fit(self, X, y):
        """Fit training data"""
        rand = np.random.RandomState(self.random_state)
        # initialize random weights
        self.w = rand.uniform(low=0.0, high=1, size=1+X.shape[1])
        # initialize array to hold # errors at each epoch
        self.err = []
        # loop through epochs
        for i in range(self.n):
            # initialize variable to keep track of errors
            errors = 0
            # if a rate reduction function is given
            if self.reduction is not None:
                # get updated rate
                self.rate = self.reduction.get()
            # loop through features and targets in X and y
            for xi, y_true in zip(X, y):
                # calulate update using perceptron learning rule
                u = self.rate * (y_true - self.predict(xi))
                # update weights
                self.w[1:] += (u * xi)
                self.w[0] += u
                # add to errors if update is not 0
                errors += int(u != 0.0)
            # append total number of errors in epoch to err
            self.err.append(errors)
        return self

    def calculate_dot(self, X):
        """Calculates dot product of values and weights wTx"""
        return np.dot(X, self.w[1:]) + self.w[0]

    def predict(self, X):
        """Predict results based on the output of the dot product"""
        # return 1 if dot product is greater or equal to 0.5, else 0
        return np.where(self.calculate_dot(X) >= 0.5, 1, 0)

    def plot_results(self):
        """Returns a plot of error vs epoch"""
        # close any open plt
        plt.close()
        # plot epoch as x and num errors as y
        plt.plot([i + 1 for i in range(len(self.err))], self.err, 'b')
        plt.xlabel("Epoch")
        plt.ylabel("Number of Updates")
        plt.ylim(bottom=0)
        return plt


class PerceptronPML(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of
          samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,
                              size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.5, 1, 0)  # changed for classes 1 and 0

    def plot_results(self):
        """Returns a plot of error vs epoch"""
        plt.close()
        plt.plot([i + 1 for i in range(len(self.errors_))], self.errors_, 'b')
        plt.xlabel("Epoch")
        plt.ylabel("Number of Updates")
        plt.ylim(bottom=0)
        return plt


class Heuristic:
    """Heuristic classifier implementation"""
    def __init__(self, random_state=1):
        self.random_state = random_state  # set random state

    def fit(self, X, y):
        """Fit training data"""
        rand = np.random.RandomState(self.random_state)
        # initialize random weights
        self.w = rand.uniform(low=0.0, high=1, size=X.shape[1])
        return self

    def calculate_dot(self, X):
        """Calculates dot product of values and weights wTx"""
        return np.dot(X, self.w)

    def predict(self, X):
        """Predict results based on the output of the dot product"""
        # return 1 if dot product is greater or equal to 0.5, else 0
        return np.where(self.calculate_dot(X) >= 0.5, 1, 0)


def plot_decision_regions(X, y, classifier, resolution=0.02):
    """Function from Ch.2 of Python Machine Learning ISBN: 9781787125933"""
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')


dec_rate = ConstantReduction(factor=1.2)
fib_rate = FibonacciReduction()

# read data using pandas
train_data = pd.read_csv(train_file, sep='\t')

# set target
y = train_data.group

# subset data for features
X = train_data.iloc[:, [1, 2]].values

# split data
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1)

# set and train model (using learning rate 0.0001)
model = Perceptron(rate=0.0001, n=125)
model.fit(X_train, y_train)
plt1 = model.plot_results()
plt1.title('Updates per Epoch (Perceptron)')
plt1.savefig(str(output_file) + '_updates_per_epoch_perceptron' + '.png')

# plot decision boundary
plt.close()
plot_decision_regions(X_train, y_train, classifier=model)
plt.xlabel("value_1")
plt.ylabel("value_2")
plt.legend(loc='upper left')
plt.title("Decision Boundary (Perceptron)")
plt.savefig(str(output_file) + '_boundary_perceptron' + '.png')

# set and train model using PML implementation
model2 = PerceptronPML(eta=0.0001, n_iter=10)
model2.fit(X_train, y_train)
# plot errors vs epoch for PML implementation
plt2 = model2.plot_results()
plt2.title('Updates per Epoch (PML Perceptron)')
plt2.savefig(str(output_file) + '_updates_per_epoch_pml_perceptron' + '.png')

# plot decision boundary
plt.close()
plot_decision_regions(X_train, y_train, classifier=model2)
plt.xlabel("value_1")
plt.ylabel("value_2")
plt.legend(loc='upper left')
plt.title("Decision Boundary (PML Perceptron)")
plt.savefig(str(output_file) + '_boundary_pml_perceptron' + '.png')

# set and train model using heuristic
model3 = Heuristic()
model3.fit(X_train, y_train)

# plot decision boundary
plt.close()
plot_decision_regions(X_train, y_train, classifier=model3)
plt.xlabel("value_1")
plt.ylabel("value_2")
plt.legend(loc='upper left')
plt.title("Decision Boundary (Heuristic)")
plt.savefig(str(output_file) + '_boundary_heuristic' + '.png')

