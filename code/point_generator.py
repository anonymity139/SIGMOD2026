import numpy as np
import random


dim = 20
test_size = 100000
with open("./input/Unit" + str(dim) + "d" + str(int(test_size/1000))+ "k.txt", "a") as out_cp:  # "a" represents adding to the end of the file
        print(test_size, dim, file=out_cp)

for t in range(test_size):
    u = []
    sum = 0.0
    for i in range(dim):
        u.append(random.random())
    u = np.array(u)
    u = u / np.linalg.norm(u)

    with open("./input/Unit" + str(dim) + "d" + str(int(test_size/1000))+ "k.txt", "a") as out_cp:  # "a" represents adding to the end of the file
        print(*u, file=out_cp)