from structure.hyperplane import Hyperplane
from structure.point import Point
from structure.point_set import PointSet
import uh
import single_pass
import highdim
import random
import extremehigh
import lowdim
from structure import constant
import time
import highRL
import lowRL
import argparse
import os

def delete_same(pset: PointSet):
    result_set = PointSet()
    for i, p in enumerate(pset.points):
        if i%100 == 0:
            print(f'{i}/{len(pset.points)}')
        is_exist = False
        for q in result_set.points:
            if p.is_same(q):
                is_exist = True
                break
        if not is_exist:
            result_set.add_point(p)
    return result_set

if __name__ == '__main__':
    # dataset
    dataset_name = 'audi'
    pset = PointSet(f'{dataset_name}.txt')
    dim = pset.points[0].dim
    # pset = pset.skyline()
    pset = delete_same(pset)
    for i in range(len(pset.points)):
        pset.points[i].id = i


    # write skyline data to txt file for backup
    with open(f'input/{dataset_name}Skyline.txt', 'w') as file:
        file.write(f'{len(pset.points)} {pset.points[0].dim}\n')
        for p in pset.points:
            file.writelines(f'{value:.6f} ' for value in p.coord)
            file.write('\n')