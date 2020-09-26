import sys, getopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from itertools import combinations

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


class AdalineGD(object):
    """ADAptive LInear NEuron classifier.
    From from Ch.2 of Python Machine Learning ISBN: 9781787125933
    modified to accept labels of 0 or 1 instead of -1 or 1.

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
    cost_ : list
      Sum-of-squares cost function value in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

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
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X))
                        >= 0.5, 1, 0)  # modified for 0 and 1 data


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


def label_encoding(train_X, val_X, cols):
    """Function using label encoding to convert categorical columns
    to numbers"""
    # copy data
    label_train_X = train_X.copy()
    label_val_X = val_X.copy()

    # define label encoder
    label_encoder = LabelEncoder()

    # encode labels for each feature in list
    for col in cols:
        label_train_X[col] = label_encoder.fit_transform(train_X[col])
        label_val_X[col] = label_encoder.transform(val_X[col])

    return label_train_X, label_val_X


def test_model(train_X, val_X, train_y, val_y, model):
    """Function to return the accuracy of a given model and dataset"""
    # Fit model
    model.fit(train_X, train_y)

    # get predicted values
    val_predict = model.predict(val_X)

    # return accuracy
    return accuracy_score(val_y, val_predict)


class ConstantReduction:
    """Reduces learning rate by dividing by a constant factor"""

    def __init__(self, start=0.01, factor=2):
        self.rate = start
        self.factor = factor

    def get(self):
        # update rate by dividing by factor
        r = self.rate
        self.rate = self.rate / self.factor
        return r


def test_features(X_train, X_val, y_train, y_val, feature_combinations, model):
    res = []
    for f in feature_combinations:
        sub_X_train = X_train[:, f]
        sub_X_val = X_val[:, f]
        res.append(test_model(sub_X_train, sub_X_val, y_train, y_val, model))
    return res

# read data using pandas
data = pd.read_csv(train_file)

# drop missing values
data = data.dropna(axis=0)

# set target
y = data.Survived

# select features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# subset data
X = data[features]

# split data into 70% training 30% testing
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1, train_size=0.7)

# get list of categorical variables
c = (X.dtypes == 'object')
# if categorical features exist extract index
if len(c > 0):
    features_to_encode = list(c[c].index)
    X_train, X_val = label_encoding(X_train, X_val, features_to_encode)

# scale data using sklearn StandardScaler
sc = StandardScaler()
# fit scaler on training data
sc.fit(X_train)
# transform training and validation data
X_train = sc.transform(X_train)
X_val = sc.transform(X_val)

# test different learning rates
reducing_rate = ConstantReduction(start=0.001, factor=2)
rates = []
accuracies = []
for i in range(10):
    rate = reducing_rate.get()
    rates.append(rate)
    accuracy = test_model(X_train, X_val, y_train, y_val, model=AdalineGD(eta=rate))
    accuracies.append(accuracy)

# plot accuracy vs learning rate
plt.plot(rates, accuracies)
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.savefig(str(output_file) + "_learning_rate_tests.png")
plt.close()

# get list to index features
features_index = [[i] for i in range(len(features))]
features_index.append([i for i in range(len(features))])

# test model using a single feature for each feature
res = test_features(X_train, X_val, y_train, y_val, features_index, model=AdalineGD(eta=0.0005))

# plot accuracy vs feature used
xlabels = features.copy()
xlabels.append("All")
plt.bar(xlabels, res)
plt.xlabel("Features")
plt.ylabel("Accuracy")
plt.title("Prediction Accuracy")
plt.savefig(str(output_file) + "_feature_accuracy.png")
plt.close()

# Run model using all features
model = AdalineGD(eta=0.0005)
model.fit(X_train, y_train)

# plot feature weights
abs_w = [abs(model.w_[i]) for i in range(1, len(model.w_))]
plt.bar(features, abs_w)
plt.xlabel("Features")
plt.ylabel("Absolute Value Weights")
plt.title("Adaline Determined Weights")
plt.savefig(str(output_file) + "_adaline_weights.png")
plt.close()

# plot cost vs epoch
plt.plot([i for i in range(len(model.cost_))], model.cost_, )
plt.xlabel("Epoch")
plt.ylabel("Sum-squared error")
plt.savefig(str(output_file) + "_error_per_epoch.png")
plt.close()

# accuracy for adaline and heuristic
ad = test_model(X_train, X_val, y_train, y_val, model=AdalineGD(eta=0.0005))
heu = test_model(X_train, X_val, y_train, y_val, model=Heuristic())

# plot accuracy compared to Heuristic
plt.bar(["AdalineGD", "Heuristic"], [ad, heu])
plt.ylim(0, 1)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Performance")
plt.savefig(str(output_file) + '_performance.png')
plt.close()