import numpy as np
import random


dim = 256
test_size = 10
for t in range(test_size):
    u = []
    sum = 0.0
    for i in range(dim):
        u.append(random.random())
        sum += u[i]
    for i in range(dim):
        u[i] = u[i] / sum
    print(u)

    with open("../u/u_dim_" + str(dim) + "_size_" + str(test_size)+ ".txt", "a") as out_cp:  # "a" represents adding to the end of the file
        print(*u, file=out_cp)