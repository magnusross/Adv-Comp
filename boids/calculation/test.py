import numpy as np
import numba as nb

test_numpy_dtype = np.dtype([("blah", np.int64)])
test_numba_dtype = nb.from_dtype(test_numpy_dtype)


@nb.njit
def working_fn(n):
    b = []     
    for j in range(n):
            b.append(1)
    return np.array(b)
        

@nb.njit
def desired_fn(thing):
    print(thing.blah[:])

a = np.zeros(3,test_numpy_dtype)
print(a)
print(working_fn(10))

desired_fn(a)