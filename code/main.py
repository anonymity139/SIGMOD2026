from structure.hyperplane import Hyperplane
from structure.point import Point
from structure.point_set import PointSet
import uh
import single_pass
import highdim
import random
import activeRanking
from structure import constant
import time
import argparse
import os
import highdim
import preferenceLearning


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run experiments with different parameters.")
    parser.add_argument('params', nargs='+', help="List of parameters")
    args = parser.parse_args()

    print(args.params)
    alg_name = str(args.params[0])
    dataset_name = str(args.params[1])
    epsilon = float(args.params[2])

    '''
    with open("../config.txt", 'r') as config:
        alg_name, dataset_name, epsilon = config.readline().split()
        epsilon = float(epsilon)
    '''

    # dataset
    pset = PointSet(f'{dataset_name}.txt')
    dim = pset.points[0].dim

    '''
    pset = pset.skyline()
    print(f"Dataset: {dataset_name}, Dimension: {dim}, Size: {len(pset.points)}")
    # write skyline data to txt file for backup
    with open(f'input/{dataset_name}Skyline.txt', 'w') as file:
        file.write(f'{len(pset.points)} {pset.points[0].dim}\n')
        for p in pset.points:
            file.writelines(f'{value:.6f} ' for value in p.coord)
            file.write('\n')

    
    pset_org = PointSet(f'{dataset_name}_org.txt')
    with open(f'input/{dataset_name}_orgSkyline.txt', 'w') as file:
        file.write(f'{len(pset.points)} {pset.points[0].dim}\n')
        for p in pset.points:
            file.writelines(f'{value:.6f} ' for value in pset_org.points[p.id].coord)
            file.write('\n')

    
    # utility vector
    u = Point(dim)
    utility = list(map(float, config.readline().split()))
    for i in range(dim):
        u.coord[i] = float(utility[i])
        
    sum_coord = 0.0
    for i in range(dim):
        u.coord[i] = random.random()
        sum_coord += u.coord[i]
    for i in range(dim):
        u.coord[i] = u.coord[i] / sum_coord
    # for i in range(dim):
    #    u.coord[i] = 1.0 / dim
    '''

    for i in range(len(pset.points)):
        pset.points[i].id = i
    u = Point(dim)
    for i in range(dim):
        try:
            value = float(args.params[3 + i])
            u.coord[i] = value
        except ValueError:
            print(f"无法解析参数：{args.params[3 + i]}")
    print(f"{alg_name}, {dataset_name}, {epsilon}, {u.coord}")

    # ground truth
    top_set = pset.find_top_k(u, 1)
    top_set[0].printAlgResult("GroundTruth", 0, time.time(), 0)

    # groups = pset.group_by_nonzero_dims()
    # for group in groups:
    #     print(group, len(groups[group].points))

    if alg_name == "UH-Random":  # the UH-Random algorithm
        uh.max_utility(pset, u, 2, epsilon, 1000, constant.RANDOM, constant.EXACT_BOUND, dataset_name)
    elif alg_name == "UH-Simplex":  # the UH-Simplex algorithm
        uh.max_utility(pset, u, 2, epsilon, 1000, constant.SIMPlEX, constant.EXACT_BOUND, dataset_name)
    elif alg_name == "singlePass":  # the singlePass algorithm
        single_pass.singlePass(pset, u, epsilon, dataset_name)
    elif alg_name == "APC":
        highdim.APC(pset, u, epsilon, dataset_name, u_r=1, round_num=5)
    elif alg_name == "ActiveRanking":
        activeRanking.active_ranking(pset, u, epsilon, dataset_name)
    elif alg_name == "PrefLearning":
        preferenceLearning.preference_learning(pset, u, epsilon, dataset_name)
    elif alg_name == "APCExact":
        highdim.APCExact1(pset, u, epsilon, dataset_name)
    elif alg_name == "UH-RandomMiddle":
        uh.max_utilityMiddle(pset, u, 2, epsilon, 1000, constant.RANDOM, constant.EXACT_BOUND, dataset_name)
    elif alg_name == "UH-SimplexMiddle":
        uh.max_utilityMiddle(pset, u, 2, epsilon, 1000, constant.SIMPlEX, constant.EXACT_BOUND, dataset_name)
    elif alg_name == "APCMiddle":
        highdim.APCEnhanceMiddle(pset, u, epsilon, dataset_name)
    elif alg_name == "singlePassMiddle":
        single_pass.singlePassMiddle(pset, u, epsilon, dataset_name)
    elif alg_name == "ActiveRankingMiddle":
        activeRanking.active_rankingMiddle(pset, u, epsilon, dataset_name)
    elif alg_name == "PrefLearningMiddle":
        preferenceLearning.preference_learningMiddle(pset, u, epsilon, dataset_name)
    elif alg_name == "UH-Random(reduction)":
        uh.dimension_reduction_max_utility(pset, u, 2, epsilon, 1000, constant.RANDOM, constant.EXACT_BOUND, dataset_name)
    elif alg_name == "UH-Simplex(reduction)":
        uh.dimension_reduction_max_utility(pset, u, 2, epsilon, 1000, constant.SIMPlEX, constant.EXACT_BOUND, dataset_name)