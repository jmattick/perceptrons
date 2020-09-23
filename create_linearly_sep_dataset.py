# import libraries
import getopt
import sys
import random
import matplotlib.pyplot as plt

# get arguments
args = sys.argv[1:]

# set default parameters
size = 10
n_features = 1
outname = 'dataset.txt'
sep = True

# get parameters
try:
    opts, args = getopt.getopt(args, "hs:o:l:f:", ["size=", "output=", "linearly_sep=", "features="])
except getopt.GetoptError:
    print('create_linearly_sep_dataset.py -s <size> -o <outputfile> -l <boolean linearly sep>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('create_linearly_sep_dataset.py -s <size> -o <outputfile> -l <boolean linearly sep>')
        sys.exit()
    elif opt in ('-s' or '--size'):
        size = int(arg)
    elif opt in ('-f' or '--features'):
        n_features = int(arg)
    elif opt in ('-o' or '--output'):
        outname = arg
    elif opt in ('-l' or '--linearly_sep'):
        if arg == 'F' or 'False':
            sep = False

# initialize data array
group = []  # holds group number
values = [[] for i in range(n_features)]  # holds value
threshold = random.randint(2, 101)  # threshold if linearly separable
overlap = True  # boolean to add value to first dataset if not separable
overlap2 = True  # boolean to add value to second dataset if not separable
group_size = size/2  # 2 groups per dataset


for i in range(size+1):
    if i < group_size:  # first group
        group.append(0)
        if sep is True:  # if linearly separable
            for value in values:
                value.append(random.randint(1, threshold))
        else:  # if not linearly separable
            if overlap is True:  # adds threshold value to first group
                for value in values:
                    value.append(threshold)
                overlap = False
            else:  # else add random value
                for value in values:
                    value.append(random.randint(1, 101))
    else:  # second group
        group.append(1)
        if sep is True:  # if linearly separable
            for value in values:
                value.append(random.randint(threshold, 101))
        else:  # if not linearly separable
            if overlap2 is True: # adds threshold value to second group
                for value in values:
                    value.append(threshold)
                overlap2 = False
            else:  # else add random value
                for value in values:
                    value.append(random.randint(1, 101))


# plot values for first two features
plt.scatter(values[0][0:int(group_size)], values[1][0:int(group_size)], color='red', label='group 0')
plt.scatter(values[0][int(group_size+1):], values[1][int(group_size+1):], color='blue', label='group 1')
plt.legend(loc='upper left')
plt.xlabel('value_1')
plt.ylabel('value_2')
plt.savefig(outname + '.png')  # save as png with basename as outname


# output to file
with open(outname, 'w') as z:
    title = "group"
    for i in range(len(values)):
        title = title + '\t' + 'value_' + str(i+1)
    z.write(title + '\n')
    for i in range(len(group)):
        values_string = ''
        for value in values:
            values_string = values_string + '\t' + str(value[i])
        z.write(str(group[i]) + values_string + '\n')


