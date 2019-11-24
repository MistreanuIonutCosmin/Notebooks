#%%
import numpy as np 
print(np.__version__)

# %% # 1D array
np.arange(10)

# %% 3x3 boolean matrix
np.ones((3,3), dtype=bool)


# %% # extract odd numbers
arr = np.arange(10)
arr[arr %2 == 1]


# %% replace items 
arr = np.arange(10)
arr[arr %2 == 1] = -1
arr

# %% replace without changing 
arr = np.arange(10) 
np.where(arr % 2 == 1, -1, arr)

# %% 7 reshape array
arr = np.arange(10)
arr.reshape((2, 5))


#%% 8 concatenate vertically  
a = np.arange(10).reshape(2, -1)
b = np.repeat(1, 10).reshape(2, -1)
np.concatenate([a, b], axis=0)

# %% 9 concatenate horizontally
a = np.arange(10).reshape(2, -1)
b = np.repeat(1, 10).reshape(2, -1)
np.concatenate([a, b], axis=1)

# %% 10 repeat array a for 3 times / repeat elements of an array 3 times
a = np.array([1,2,3])
np.tile(a, 3)
np.repeat(a, 3)

# %% 11 common items between two python arrays
a = np.array([1,2,3,2,4,5,6])
b = np.array([2,2,3,8,4,9,6])
np.intersect1d(a, b)

#%% 12 arr1 - arr2 
a = np.array([1,2,3,2,4,5,6])
b = np.array([2,2,3,8,4,9,6])
np.setdiff1d(a, b)


#%% 13 postions where elements match 
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.where(a == b)

# %% 14 all numbers in a range from array
a = np.array([2, 6, 1, 9, 10, 3, 27])
a[np.where((5 <= a) & (a <= 10))]


# %% 15 make a function that workd on scalars to work on vectors
def maxx(x, y):
    if x >= y:
        return x
    else: 
        return y
maxx(1, 5)

vmax = np.vectorize(maxx, otypes=[float])
a = np.array([5, 7, 9, 8, 6, 4, 5])
b = np.array([6, 3, 4, 8, 9, 7, 1])

vmax(a, b)

# %% 16 swap column 1 and 2 !! 
arr = np.arange(9).reshape((3, -1))
arr[:, [1, 0, 2]]

#%% 17 swap two rows in a 2d array 
arr = np.arange(9).reshape((3, -1))
arr[[1, 0, 2],:]


#%% 18 reverse rows of a 2D array
arr[::-1]

# %% 19 reverse columsn of a 2D array, '::step' syntax chan be used to iterate over a vector 
arr = np.arange(9).reshape((3, -1))
arr[:,::-1]

# %% 20 random numbers between [a, b]
arr = np.random.uniform(5, 10, size=(5, 3))
arr

# %% 21 print only 3 decimal places in python
arr = np.random.random((5,3))
np.set_printoptions(precision=3)
arr[:4]

# %% 22 suprpressing scientific notation
arr = np.random.random((5,3)) / 1e3
arr
np.set_printoptions(suppress=True, precision=6)
arr
# %% 23 limit the number of items printed
arr = np.arange(20)
print(arr)
np.set_printoptions(threshold=5)
arr

# %% 24 force printing all the numbers
arr = np.arange(20)
arr
import sys
np.set_printoptions(threshold=sys.maxsize)
arr

# %% 25 import a dataset with numbers and texts keeping the text intact in python numpy
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
print(iris[:3])
iris = np.genfromtxt(url, delimiter=',', dtype=None)
iris[:3]

# %% 26 extract a particular column from 1D array of tuples?
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)
species = np.array([row[4] for row in iris_1d])
species[:5]

# %% 27 How to convert a 1d array of tuples to a 2d numpy array?
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)
print(type(iris_1d))
print(iris_1d[:3])
np.array([row.tolist() for row in iris_1d])[:3]

# %% 28 compute the mean, median, standard deviation of a numpy array?
#http://www.ltcconline.net/greenl/courses/201/descstat/mean.htm
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
mu, med, sd = np.mean(sepallength), np.median(sepallength), np.std(sepallength)
print(mu, med, sd)

# %% 29 normalize an array so the values range exactly between 0 and 1
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
Smax, Smin = sepallength.max(), sepallength.min()
S = (sepallength - Smin) / (Smax - Smin)
S
# %% 30 How to compute the softmax score?
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
print(softmax(sepallength))

# %% https://www.machinelearningplus.com/python/101-numpy-exercises-python/
