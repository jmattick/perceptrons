# import libraries
import getopt
import sys
import random

# get arguments
args = sys.argv[1:]

# set default parameters
size = 10
outname = 'dataset.txt'
sep = True

# get parameters
try:
    opts, args = getopt.getopt(args, "hs:o:l", ["size=", "output=", "linearly_sep="])
except getopt.GetoptError:
    print('create_linearly_sep_dataset.py -s <size> -o <outputfile> -l <boolean linearly sep>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('create_linearly_sep_dataset.py -s <size> -o <outputfile> -l <boolean linearly sep>')
        sys.exit()
    elif opt in ('-s' or '--size'):
        size = int(arg)
    elif opt in ('-o' or '--output'):
        outname = arg
    elif opt in ('-l' or '--linearly_sep'):
        if arg == 'F' or 'False':
            sep = False

# initialize data array
group = []  # holds group number
value = []  # holds value
threshold = random.randint(2, 101)  # threshold if linearly separable
overlap = True  # boolean to add value to first dataset if not separable
overlap2 = True  # boolean to add value to second dataset if not separable
group_size = size/2  # 2 groups per dataset


for i in range(size+1):
    if i < group_size:  # first group
        group.append(0)
        if sep is True:  # if linearly separable
            value.append(random.randint(1, threshold))
        else:  # if not linearly separable
            if overlap is True:  # adds threshold value to first group
                value.append(threshold)
                overlap = False
            else:  # else add random value
                value.append(random.randint(1, 101))
    else:  # second group
        group.append(1)
        if sep is True:  # if linearly separable
            value.append(random.randint(threshold, 101))
        else:  # if not linearly separable
            if overlap2 is True: # adds threshold value to second group
                value.append(threshold)
                overlap2 = False
            else: # else add random value
                value.append(random.randint(1, 101))

# output to file
with open(outname, 'w') as z:
    z.write("group\tvalue\n")
    for i in range(len(group)):
        z.write(str(group[i]) + '\t' + str(value[i]) + '\n')


