# http://softwareengineering.stackexchange.com/questions/254475/how-do-i-move-away-from-the-for-loop-school-of-thought
# Losing your Loops Fast Numerical Computing with NumPy - https://www.youtube.com/watch?v=EEUXKG97YRw

import numpy as np

# Problem 1
def sumproducts(x, y):
    """Return the sum of x[i] * y[j] for all pairs of indices i, j.

    >>> sumproducts(np.arange(3000), np.arange(3000))
    20236502250000

    """
    result = 0
    for i in range(len(x)):
        for j in range(len(y)):
            result += x[i] * y[j]
    return result

def sumproducts_np(x, y):
    # return np.sum(x) * np.sum(y)
    return np.sum(np.outer(x,y))


# Problem 2
def countlower(x, y):
    """Return the number of pairs i, j such that x[i] < y[j].

    >>> countlower(np.arange(0, 200, 2), np.arange(40, 140))
    4500

    """
    result = 0
    for i in range(len(x)):
        for j in range(len(y)):
            if x[i] < y[j]:
                result += 1
    return result

def countlower_np(x, y):
    return np.sum(np.searchsorted(np.sort(x),y))

# Problem 3
def cleanup(x, missing=-1, value=0):
    """Return an array that's the same as x, except that where x ==
    missing, it has value instead.

    >>> cleanup(np.arange(-3, 3), value=10)
    ... # doctest: +NORMALIZE_WHITESPACE
    array([-3, -2, 10, 0, 1, 2])

    """
    result = []
    for i in range(len(x)):
        if x[i] == missing:
            result.append(value)
        else:
            result.append(x[i])
    return np.array(result)

def cleanup_np(x, missing=-1, value=0):
    # np.place(x,x==missing,value)
    return x

##### Problem 1 #####
# a = np.arange(3)
# b = np.arange(4)

# print sumproducts(a,b)
# print sumproducts_np(a,b)

##### Problem 2 #####
# x = np.array([1,2,3,4])
# y = np.array([0,2,5])

##### Problem 3 #####

print cleanup(np.arange(-3, 3), value=10)
print cleanup_np(np.arange(-3, 3), value=10)


