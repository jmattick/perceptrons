# Perceptrons

### Create synthetic dataset

Use the `create_linearly_sep_dataset.py` script to create 
a synthetic dataset. By default, a linearly separated dataset 
containing two groups will be output to a `dataset.txt` file.
Values of the two groups consist of integers between 1 and 100.
For linearly separated datasets, a random threshold is created
and the values (x1) for the first group will be 1 <= x1 < threshold.
The values (x2) for the second group wil be threshold <= x2 <= 100.
For non-linearly separated datasets, the values of both groups will 
be an integer between 1 and 100 inclusively. To ensure that the two 
groups are non-linearly separated, at least one of the random integers 
is inserted into both groups. The group assignments and values are
output to a tab-delimited file. A plot of the first two dimensions
is saved if the number of dimensions is greater than one.

Parameters:

- `-s` or `--size`: set sample size of dataset'

- `-o` or `--output`: set output file name

- `-l` or `--linearly_sep`: boolean to describe if dataset is 
linearly separable.

- `-f` or `--features`: set number of features

Example creating a two dimensional linearly separable dataset:
```
python create_linearly_sep_dataset.py -s 10 -o linear_sep_dataset.txt -f 2
```

Plot linearly separable dataset:

![linear plot](linear_sep_dataset.txt.png)

Example creating a two dimensional non-linearly separable dataset:
```
python create_linearly_sep_dataset.py -s 10 -o non_linear_sep_dataset.txt -f 2 -l F
```

Plot non-linearly separable dataset:

![non-linear plot](non_linear_sep_dataset.txt.png)

### Implement Perceptron

A perceptron was implemented in `perceptrons.py`. Weights are originally set as 
random values between 0 and 1. To predict values, the dot product of weights and 
values are calculated then rounded to the nearest class (0 if less than 0.5, else 1). 
The fit function updates the weights over a set number of iterations. The weights 
are updated using the perceptron learning rule: 
`dw = learning rate * (actual - predicted) * xi`. Additionally, a plot_results 
function was implemented to plot the number of errors at each epoch. 

```
class Perceptron:
    """Perceptron implementation"""

    def __init__(self, rate=0.01, n=10, random_state=1, reduction=None):
        self.rate = rate  # learning rate
        self.n = n  # number of iterations
        self.random_state = random_state  #set random state
        self.reduction = reduction  # optional if dynamic learning rate is desired

    def fit(self, X, y):
        """Fit training data"""
        rand = np.random.RandomState(self.random_state)
        # initialize random weights
        self.w = rand.uniform(low=0.0, high=1, size=X.shape[1])
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
                self.w += (u * xi)
                # add to errors if update is not 0
                errors += int(u != 0.0)
            # append total number of errors in epoch to err
            self.err.append(errors)
        return self

    def calculate_dot(self, X):
        """Calculates dot product of values and weights wTx"""
        return np.dot(X, self.w)

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
        return plt
```

Additionally, the perceptron in Python Machine Learning (PML) Ch. 2 
was tested. The textbook implementation was modified slightly. The 
textbook implementation used data with -1 or 1 classes, so the 
rounding in the predict class was changed for 0 or 1 
classes. A plot_results function was added to facilitate data reporting.
One difference between the implementation above and the textbook
implementation is that the implementation above sets the initial
weights to random values, where the random numbers are taken from 
a normal distribution with a standard deviation of 0.01. The 
textbook implementation also includes an extra value as the bias
unit.

```
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
        return plt
```

### Test perceptron on synthetic datasets

To test the implemented perceptrons, additional code was added
to the `perceptrons.py` script to run both models on a given 
dataset. The script will train and fit the data using each 
implementation and save a plot of udates vs epochs for each one.

```
# read data using pandas
train_data = pd.read_csv(train_file, sep='\t')

# set target
y = train_data.group

# subset data for features
X = train_data.iloc[:, [1, 2]].values

# split data
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1)

# set and train model (using learning rate 0.01)
model = Perceptron()
model.fit(X_train, y_train)
plt1 = model.plot_results()
plt1.title('Updates per Epoch (Perceptron)')
plt1.savefig(str(output_file) + '_updates_per_epoch_perceptron' + '.png')

# set and train model using PML implementation
model2 = PerceptronPML(n_iter=10)
model2.fit(X_train, y_train)
# plot errors vs epoch for PML implementation
plt2 = model2.plot_results()
plt2.title('Updates per Epoch (PML Perceptron)')
plt2.savefig(str(output_file) + '_updates_per_epoch_pml_perceptron' + '.png')
```

The `perceptrons.py` script was run on both the linearly and non-linearly
separable datasets generated above. 

```
python perceptrons.py -t linear_sep_dataset.txt -o synthetic_results\linear_sep_data
python perceptrons.py -t non_linear_sep_dataset.txt -o synthetic_results\non_linear_sep_data
```

For the linearly separable dataset, both models converged (no more updates). 
The PML implementation converged one epoch earlier as seen in the following plots. 

![linear plot](synthetic_results/linear_sep_data_updates_per_epoch_perceptron.png)
![PML linear plot](synthetic_results/linear_sep_data_updates_per_epoch_pml_perceptron.png)

Both models were not able to converge using the non-linearly separable dataset. 

![non-linear plot](synthetic_results/non_linear_sep_data_updates_per_epoch_perceptron.png)
![PML non-linear plot](synthetic_results/non_linear_sep_data_updates_per_epoch_pml_perceptron.png)

# Usage

### Create synthetic dataset

Use the `create_linearly_sep_dataset.py` script to create 
a synthetic dataset. 

Parameters:

- `-s` or `--size`: set sample size of dataset'

- `-o` or `--output`: set output file name

- `-l` or `--linearly_sep`: boolean to describe if dataset is 
linearly separable.

- `-f` or `--features`: set number of features

Example creating a two dimensional linearly separable dataset:

```
python create_linearly_sep_dataset.py -s 10 -o linear_sep_dataset.txt -f 2
```

Example creating a two dimensional non-linearly separable dataset:

```
python create_linearly_sep_dataset.py -s 10 -o non_linear_sep_dataset.txt -f 2 -l F
```

### Test Perceptron Implementation

Use `perceptrons.py` to run two implementations of a perceptron on a given
dataset.

Parameters:

- `-t` or `--train`: input training dataset

- `-o` or `--output`: set output file base name

Example:

```
python perceptrons.py -t linear_sep_dataset.txt -o synthetic_results\linear_sep_data
```