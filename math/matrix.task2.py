import numpy as np

a = np.random.random((10,10))

a[a > 0.5] = 1
a[a <= 0.5] = 0

print(a)
print(a[a > 0.5].size)
print(a[a <= 0.5].size)
print(a[a > 0.5].size + a[a <= 0.5].size)

