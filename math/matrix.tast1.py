import numpy as np
a = np.random.random((5,5))
print(a.sum(), np.mean(a), a.max(), a.min(), np.sum(a,
axis = 0), np.sum(a, axis = 1 ))
print(a.transpose())
print(a @ a.T)