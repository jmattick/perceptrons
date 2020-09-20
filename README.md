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
output to a tab-delimited file.

Parameters:

- `-s` or `--size`: set sample size of dataset'

- `-o` or `--output`: set output file name

- `-l` or `--linearly_sep`: boolean to describe if dataset is 
linearly separable.

Example creating linearly separable dataset:
```
python create_linearly_sep_dataset.py -s 10 -o linear_sep_dataset.txt
```

Example creating non-linearly separable dataset:
```
python create_linearly_sep_dataset.py -s 10 -o non_linear_sep_dataset.txt -l F
```

