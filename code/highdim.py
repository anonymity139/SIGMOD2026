from typing import List
from structure.point import Point
from structure.point_set import PointSet
from structure.hyperplane_set import HyperplaneSet
from structure.hyperplane import Hyperplane
import time
import math
import numpy as np
import cdd
import swiglpk as glp
from collections import Counter

"""
Johnson-Lindenstrauss
"""
def jl_projection(points, target_dim):
    if isinstance(points, list):
        points = np.array([p.coord for p in points])
    
    n_points, original_dim = points.shape
    
    # generate random projection matrix
    projection_matrix = np.random.normal(0, 1/np.sqrt(target_dim), (target_dim, original_dim))
    
    # execute projection
    projected_points = np.dot(points, projection_matrix.T)
    
    projected_point_objects = PointSet()
    index = -1
    for p in projected_points:
        index += 1
        projected_point_objects.points.append(Point(dim=target_dim, id=index, coord=p))
    
    return projected_point_objects


def calculate_constraints_batch(original_utility_range, direction, current_point_u):
    """
    批量计算所有约束的上界
    
    参数:
    original_utility_range: shape为(n_constraints, dim+1)的数组，每行为[b, a1, a2, ...]
    direction: shape为(dim,)的数组
    current_point_u: shape为(dim,)的数组，当前点的效用向量
    
    返回:
    t_max: 最小的上界值
    constraint_index: 对应的约束索引
    """
    # 将约束转换为NumPy数组（如果还不是的话）
    constraints = np.array(original_utility_range)
    
    # 分离偏移量b和法向量a
    b = constraints[:, 0]  # shape: (n_constraints,)
    a = constraints[:, 1:]  # shape: (n_constraints, dim)
    
    # 计算分母 (a·direction)
    denom = np.dot(a, direction)  # shape: (n_constraints,)
    
    # 计算分子 -(a·current_u + b)
    numer = -(np.dot(a, current_point_u) + b)  # shape: (n_constraints,)
    
    # consider the case that denom is 0
    mask = np.abs(denom) >= 1e-10
    t_values = np.full_like(denom, np.inf)
    t_values[mask] = numer[mask] / denom[mask]
    
    # 找到最小的正t值
    valid_t = t_values[t_values > 0]
    if len(valid_t) > 0:
        t_max = np.min(valid_t)
        constraint_index = np.where(t_values == t_max)[0][0]
    else:
        t_max = float('inf')
        constraint_index = -1
    
    return t_max, constraint_index


def calculate_constraints_batch_multi_directions(A, directions, current_point_u):
    constraints = np.array(A)
    b = constraints[:, 0]  # shape: (n_constraints,)
    a = constraints[:, 1:]  # shape: (n_constraints, dim)
    
    # calculate the denominator (a·direction)
    denom = np.dot(a, directions.T)
    
    # calculate the numerator -(a·current_u + b)
    numer = -(np.dot(a, current_point_u) + b).reshape(-1, 1)
    # broadcast the numerator to all directions
    numer = np.broadcast_to(numer, denom.shape)
    
    # handle the case that denom is 0
    mask = np.abs(denom) >= 1e-10
    t_values = np.full_like(denom, np.inf)
    t_values[mask] = numer[mask] / denom[mask]
    
    # find the smallest positive t value for each direction
    # replace the negative and 0 with inf, so that they will be ignored when finding the minimum
    t_values[t_values <= 0] = np.inf
    
    # find the smallest positive t value for each direction
    # t_maxs shape: (n_directions,)
    t_maxs = np.min(t_values, axis=0)
    
    # find the constraint index for each direction
    # constraint_indices shape: (n_directions,)
    constraint_indices = np.argmin(t_values, axis=0)
    
    # handle the case that no valid t value is found for some directions
    invalid_directions = np.isinf(t_maxs)
    constraint_indices[invalid_directions] = -1
    
    valid_mask = ~np.isinf(t_maxs) & (constraint_indices != -1)
    all_points = current_point_u + (t_maxs + 1e-5).reshape(-1, 1) * directions

    intersection_points = np.full(len(directions), None, dtype=object)

    # find the valid indices
    valid_indices = np.where(valid_mask)[0]

    # assign the valid points
    if len(valid_indices) > 0:
        intersection_points[valid_indices] = list(all_points[valid_indices])

    return intersection_points, constraint_indices


def sample_utilities_batch(dim, sample_u_num, utility_range, base_utility):
    """
    完全批量采样效用向量，无任何循环
    """
    # generate all random directions at once
    n_samples = sample_u_num
    directions = np.random.normal(0, 1, (n_samples, dim))
    directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
    
    # extract the constraint matrix
    constraints = np.array(utility_range)
    b = constraints[:, 0]  # shape: (n_constraints,)
    a = constraints[:, 1:]  # shape: (n_constraints, dim)
    
    # calculate the denominator for all directions shape: (n_constraints, n_samples)
    denom = np.dot(a, directions.T)
    
    # calculate the numerator for all directions shape: (n_constraints, n_samples)
    numer = -(np.dot(a, base_utility.coord) + b).reshape(-1, 1)
    numer = np.broadcast_to(numer, denom.shape)

    # calculate all valid t values shape: (n_constraints, n_samples)
    valid_mask = np.abs(denom) > 1e-10
    t_values = np.full_like(denom, np.inf)
    t_values[valid_mask] = numer[valid_mask] / denom[valid_mask]
    
    # create the mask for positive and negative denominators
    pos_denom_mask = denom > 0
    neg_denom_mask = denom < 0
    
    # calculate t_mins and t_maxs
    pos_values = np.where(pos_denom_mask, t_values, float('-inf'))
    neg_values = np.where(neg_denom_mask, t_values, float('inf'))
    t_mins = np.max(pos_values, axis=0)  # shape: (n_samples,)
    t_maxs = np.min(neg_values, axis=0)  # shape: (n_samples,)
    
    # find the valid directions
    valid_directions = t_mins < t_maxs
    
    # generate t values
    t_samples = np.random.uniform(t_mins[valid_directions], t_maxs[valid_directions])
    
    # generate new point coordinates
    new_coords = base_utility.coord + t_samples.reshape(-1, 1) * directions[valid_directions]
    
    # check all points
    constraints_values = np.dot(a, new_coords.T) + b.reshape(-1, 1)
    valid_points_mask = np.all(constraints_values >= -1e-10, axis=0)
    
    # create the final point objects
    new_points = []
    valid_coords = new_coords[valid_points_mask]
    for coord in valid_coords:
        new_point = Point(dim)
        new_point.coord = coord
        new_points.append(new_point)
    
    return new_points


def sample_utilities_batch1(dim, sample_u_num, utility_range, base_utility, pset):
    """
    完全批量采样效用向量，无任何循环
    """
    # create the final point objects
    new_points = []
    
    # generate all random directions at once
    n_samples = dim * sample_u_num
    directions = np.random.normal(0, 1, (n_samples, dim))
    directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
    
    # extract the constraint matrix
    constraints = np.array(utility_range)
    b = constraints[:, 0]  # shape: (n_constraints,)
    a = constraints[:, 1:]  # shape: (n_constraints, dim)
    
    # calculate the denominator for all directions shape: (n_constraints, n_samples)
    denom = np.dot(a, directions.T)
    
    # calculate the numerator for all directions shape: (n_constraints, n_samples)
    numer = -(np.dot(a, base_utility.coord) + b).reshape(-1, 1)
    numer = np.broadcast_to(numer, denom.shape)

    # calculate all valid t values shape: (n_constraints, n_samples)
    valid_mask = np.abs(denom) > 1e-10
    t_values = np.full_like(denom, np.inf)
    t_values[valid_mask] = numer[valid_mask] / denom[valid_mask]
    
    # create the mask for positive and negative denominators
    pos_denom_mask = denom > 0
    neg_denom_mask = denom < 0
    
    # calculate t_mins and t_maxs
    pos_values = np.where(pos_denom_mask, t_values, float('-inf'))
    neg_values = np.where(neg_denom_mask, t_values, float('inf'))
    t_mins = np.max(pos_values, axis=0)  # shape: (n_samples,)
    t_maxs = np.min(neg_values, axis=0)  # shape: (n_samples,)
    
    # find the valid directions
    valid_directions = t_mins < t_maxs
    
    # generate t values
    t_samples = np.random.uniform(t_mins[valid_directions], t_maxs[valid_directions])
    
    # generate new point coordinates
    new_coords = base_utility.coord + t_samples.reshape(-1, 1) * directions[valid_directions]
    
    # check all points
    constraints_values = np.dot(a, new_coords.T) + b.reshape(-1, 1)
    valid_points_mask = np.all(constraints_values >= -1e-10, axis=0)
    
    valid_coords = new_coords[valid_points_mask]
    for coord in valid_coords:
        new_point = Point(dim)
        new_point.coord = coord
        new_points.append(new_point)
    
    # construct a utility based based on each point in pset (same direction)
    for point in pset.points:
        p = Point(dim)
        p.coord = point.coord
        sum = 0
        for i in range(0, dim):
            sum += p.coord[i]
        p.coord = p.coord / sum
        # check if the point is in the utility range
        valid_mark = True
        for constraint in utility_range:
            if np.dot(constraint[1:], p.coord) + constraint[0] < 1e-10:
                print(f"constraint: {constraint}")
                print(f"dot: {np.dot(constraint[1:], p.coord) + constraint[0]}")
                print(f"p: {p.coord} is not in utility_range")
                valid_mark = False
                break
        if valid_mark:
            new_points.append(p)
    
    return new_points

# LP
def find_center_point(A, dim):
    """
    找到约束区域的中心点
    A: 约束矩阵，每行为 [b, a1, a2, ..., ad]
    dim: 问题维度
    返回：区域中心点坐标
    """
    import cvxpy as cp
    
    x = cp.Variable(dim)  # center point
    r = cp.Variable(1)    # radius
    
    b = A[:, 0]          # offset
    a = A[:, 1:]         # coefficient matrix
    
    constraints = []
    for i in range(len(A)):
        # b + ax >= 0
        # b + ax - r||a|| >= 0
        norm_ai = cp.norm(a[i, :], 2)  # norm of the coefficient vector
        constraints.append(b[i] + a[i, :] @ x - r * norm_ai >= 0)
    
    # maximize the radius
    objective = cp.Maximize(r)
    
    # solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS,  max_iters=1000, abstol=1e-8, reltol=1e-8)
    
    return x.value


def check_constraints_batch_detailed(original_utility_range, center_point):
    coefficients = original_utility_range[:, 1:]
    offsets = original_utility_range[:, 0]
    constraints_values = np.dot(coefficients, center_point) + offsets
    
    # find the violated constraints
    violated_indices = np.where(constraints_values < 1e-10)[0]
    inside_mark = len(violated_indices) == 0
    if not inside_mark:
        print(f"violated constraints number: {len(violated_indices)}")
        print(f"most violated constraint value: {np.min(constraints_values)}")
    
    return inside_mark, constraints_values, violated_indices


# calculate the boundary separately
# ray shooting method
# dimension reduction
def APCMiddle(pset: PointSet, u: Point, epsilon, dataset_name):
    groudtruth_utility = pset.find_top_k(u, 1)[0].dot_prod(u)
    question_threshold = 10
    start_time = time.time()
    num_of_question = 0
    old_num_of_question = 0 # to avoid precision problem
    original_dim = pset.points[0].dim
    projected_points = None
    best = None
    sample_u_num = original_dim * original_dim
    sample_ray_num = 20 * original_dim 
    round_num = 5
    # JL projection
    dim_error = 0.1  # distance error
    projected_dim = int(4 * np.log(len(pset.points)) / (dim_error**2 / 2 - dim_error**3 / 3))

    if projected_dim > original_dim:
        # original utility range
        original_utility_range = []
        points_array = np.array([p.coord for p in pset.points])
        # x >= 0 => 0 <= 0 + xi
        for j in range(original_dim):
            a = np.zeros(original_dim + 1)
            a[j + 1] = 1
            original_utility_range.append(a.tolist())
        # x1 + x2 +... + xd <= 1    ==>     0 <= 1 - x1 - x2 -... - xd
        a = np.zeros(original_dim + 1)
        a[0] = 1.0000000001
        for j in range(original_dim):
            a[j + 1] = -1
        original_utility_range.append(a.tolist())
        original_utility_range = np.array(original_utility_range)
        
        all_considered = False
        sample_points = pset.generate_sample_points(original_dim, sample_u_num)
        print(f"sample_points: {len(sample_points)}")
        best = sample_points[0]

        while len(sample_points) > 1:
            best_changed_mark = True

            # single iteration path construction
            while (best_changed_mark):
                best_changed_mark = False

                # constraints construction
                A = []
                j = 0
                while j < len(sample_points):
                    if best.id != sample_points[j].id:
                        a = None
                        if epsilon <= 0.005:
                            a = best.coord - sample_points[j].coord
                        else:
                            a = best.coord - (1 - epsilon + 0.005) * sample_points[j].coord
                        a = a.tolist()
                        a.insert(0, 0)
                        A.append(a)
                        j += 1
                    else:
                        sample_points.remove(sample_points[j])
                if(len(A) == 0 and len(sample_points) < 1):
                    print(f"A is empty")
                    break
                A = np.vstack([A, original_utility_range])
                
                # find a center point in constraints set A
                center_point = Point(original_dim)
                center_point.coord = best.sampledU
                t = np.dot(A[:,1:], center_point.coord) + A[:,0]
                if np.all(t >= 1e-20):
                    print(f"center_point is in A")
                else:
                    print(f"center_point is Not in A")
                    # generate a utility vector in utility range
                    center_point.coord = best.sampledU
                    while True:
                        t = np.dot(original_utility_range[:,1:], center_point.coord) + original_utility_range[:,0]
                        if np.all(t >= 1e-20):
                            best = pset.find_top_k_batch(center_point, 1)[0]
                            best.sampledU = center_point.coord
                            best_changed_mark = True
                            break
                        #center_point.coord = np.random.uniform(0, 1, original_dim)
                        center_point.coord = find_center_point(original_utility_range, original_dim)
                        center_point.coord = center_point.coord / np.sum(center_point.coord)
                    continue
                    
                '''
                test_set = PointSet()
                test_set.points = sample_points
                test_set.points.append(best)
                p = test_set.find_top_k_batch(center_point, 1)[0]
                print(f"best.id: {best.id}, utility: {best.dot_prod(center_point)}, p.id: {p.id}, utility: {p.dot_prod(center_point)}")
                if best.id != p.id:
                    print(f"best.id: {best.id}, p.id: {p.id}")
                    return
                sample_points.remove(best)
                '''

                center_point = center_point.coord
                # ray shooting method finding the boundary of the polytope
                for _ in range(round_num):
                    t1 = time.time()
                    directions = np.random.normal(0, 1, (sample_ray_num, original_dim))
                    directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
                    # generate unit directions and negative unit directions
                    unit_directions = np.eye(original_dim)
                    neg_unit_directions = -np.eye(original_dim)
                    directions = np.vstack([directions, unit_directions, neg_unit_directions])
                    t_utility_vectors, constraint_indices = calculate_constraints_batch_multi_directions(A, directions, center_point)
                    t2 = time.time()
                    print(f"calculate_constraints_batch_multi_directions time: {t2 - t1}")


                    dir = np.array([u.coord - center_point])
                    dir = dir / np.linalg.norm(dir)
                    _, constraint_indices2 = calculate_constraints_batch_multi_directions(A, dir, center_point)
                    if constraint_indices2[0] != -1 and constraint_indices2[0] < len(sample_points):
                        if sample_points[constraint_indices2[0]].mark == False:
                            print(f"constraint_indices2: {constraint_indices2}, point: {sample_points[constraint_indices2[0]].id}")
                            print("best norm", np.linalg.norm(best.coord))
                            print("sample_points norm", np.linalg.norm(sample_points[constraint_indices2[0]].coord))
                        else:
                            print(f"case 1: constraint_indices2: [], point: ")
                    else:
                        print(f"case 2: constraint_indices2: [], point: {constraint_indices2[0]}")


                    t3 = time.time()
                    # get the valid indices and t values
                    valid_mask = (constraint_indices != -1) & \
                                (constraint_indices < len(sample_points)) & \
                                ~np.array([sample_points[i].mark if i != -1 and i < len(sample_points) else True for i in constraint_indices])
                    valid_ci = constraint_indices[valid_mask]
                    valid_t = t_utility_vectors[valid_mask]
                    # get the unique values and counts, and create the sorted array
                    unique_ci, indices, counts = np.unique(valid_ci, return_index=True, return_counts=True)
                    first_t = valid_t[indices]
                    sort_idx = np.lexsort((unique_ci, -counts))
                    valid_pairs = [[unique_ci[i], first_t[i]] for i in sort_idx]
                    for tt, t_val in valid_pairs:
                        print(f"sample_points[tt[0]].id: {sample_points[tt].id}")
                    t4 = time.time()
                    print(f"get valid_pairs time: {t4 - t3}")

                    for constraint_index, t_val in valid_pairs:
                        num_of_question += 1
                        print(f"Question {num_of_question}: Point {best.id}, Utility {best.dot_prod(u)}, Point {sample_points[constraint_index].id}, Utility {sample_points[constraint_index].dot_prod(u)}")
                        if best.dot_prod(u) < sample_points[constraint_index].dot_prod(u):
                            a = sample_points[constraint_index].coord - best.coord
                            a = a.tolist()
                            a.insert(0, 0)
                            original_utility_range = np.vstack([original_utility_range, a])
                            best.mark = True
                            best = sample_points[constraint_index]
                            best_changed_mark = True
                            best.sampledU = t_val

                            # print the middle result
                            current_best_point_utility = best.dot_prod(u)
                            middle_rr = 1 - current_best_point_utility / groudtruth_utility
                            print(f"current_best_point: {best.id}, current_best_point_utility: {current_best_point_utility}, groudtruth_utility: {groudtruth_utility}, rr: {middle_rr}")
                            best.printMiddleResultToFile("APCEnhance", dataset_name, epsilon, num_of_question, start_time, middle_rr)
                            if num_of_question >= question_threshold:
                                return
                            break
                        else:
                            a = best.coord - sample_points[constraint_index].coord
                            a = a.tolist()
                            a.insert(0, 0)
                            original_utility_range = np.vstack([original_utility_range, a])
                            sample_points[constraint_index].mark = True

                            # print the middle result
                            current_best_point_utility = best.dot_prod(u)
                            middle_rr = 1 - current_best_point_utility / groudtruth_utility
                            print(f"current_best_point: {best.id}, current_best_point_utility: {current_best_point_utility}, groudtruth_utility: {groudtruth_utility}, rr: {middle_rr}")
                            best.printMiddleResultToFile("APCEnhance", dataset_name, epsilon, num_of_question, start_time, middle_rr)
                            if num_of_question >= question_threshold:
                                return
                                 
                    if best_changed_mark:
                        break

            
            if old_num_of_question == num_of_question and not all_considered:
                all_considered = True
                sample_points = []
                for p in pset.points:
                    if p.mark == False:
                        sample_points.append(p)
            elif not all_considered:
                # base_utility = R.find_feasible()
                base_utility = Point(original_dim)
                t1 = time.time()
                base_utility.coord = find_center_point(original_utility_range, original_dim)
                t2 = time.time()
                print(f"find_center_point time: {t2 - t1}")
                samples_utility_possible = []
                samples_utility_possible = sample_utilities_batch(original_dim, sample_u_num, original_utility_range, base_utility)
                print(f"samples_utility_possible: {len(samples_utility_possible)}")

                # stopping condition checking
                sample_points = [best]
                utility_vectors = np.array([u.coord for u in samples_utility_possible])
                all_utility_values = np.dot(utility_vectors, points_array.T)
                max_indices = np.argmax(all_utility_values, axis=1)
                max_values = all_utility_values[np.arange(len(all_utility_values)), max_indices]
                threshold_matrix = (1 - epsilon) * max_values.reshape(-1, 1)
                best_utility_values = (np.dot(utility_vectors, best.coord.T)).reshape(-1, 1)
                candidate_mask = best_utility_values > threshold_matrix + 1e-5
                for i, mask in enumerate(candidate_mask):
                    if mask == False and pset.points[max_indices[i]].mark == False and pset.points[max_indices[i]] not in sample_points:
                        sample_points.append(pset.points[max_indices[i]])
                        pset.points[max_indices[i]].sampledU = samples_utility_possible[i].coord
                if len(sample_points) <= 1:
                    all_considered = True
                    sample_points = []
                    for p in pset.points:
                        if p.mark == False:
                            sample_points.append(p)
                old_num_of_question = num_of_question
            else:
                sample_points = []
            print(f"sample_points: {len(sample_points)}")

        best.printAlgResult("APCEnhance", num_of_question, start_time, 0)
        groudtruth = pset.find_top_k(u, 1)[0]
        rr = 1 - best.dot_prod(u) / groudtruth.dot_prod(u)
        print(f"best.id: {best.id}, best.dot_prod(u): {best.dot_prod(u)}, groudtruth.dot_prod(u): {groudtruth.dot_prod(u)}")
        print("Regret: ", rr)
        best.printToFile("APCEnhance", dataset_name, epsilon, num_of_question, start_time, rr)
        # pset.printFinal(best, num_of_question, u, "APC", dataset_name, epsilon)




def APC(pset: PointSet, u: Point, epsilon, dataset_name, u_r=1, round_num=5):
    start_time = time.time()
    num_of_question = 0
    old_num_of_question = 0 # to avoid precision problem
    original_dim = pset.points[0].dim
    projected_points = None
    best = None
    sample_u_num = u_r * original_dim * original_dim
    sample_ray_num = 20 * original_dim  # 20 * round_num = 100
    # JL projection
    dim_error = 0.1  # distance error
    projected_dim = int(4 * np.log(len(pset.points)) / (dim_error**2 / 2 - dim_error**3 / 3))

    if projected_dim > original_dim:
        points_array = np.array([p.coord for p in pset.points])

        # original utility range
        original_utility_range = []
        # x >= 0 => 0 <= 0 + xi
        for j in range(original_dim):
            a = np.zeros(original_dim + 1)
            a[j + 1] = 1
            original_utility_range.append(a.tolist())
        # x1 + x2 +... + xd <= 1    ==>     0 <= 1 - x1 - x2 -... - xd
        a = np.zeros(original_dim + 1)
        a[0] = 1.0000000001
        for j in range(original_dim):
            a[j + 1] = -1
        original_utility_range.append(a.tolist())
        original_utility_range = np.array(original_utility_range)
        
        all_considered = False
        sample_points = pset.generate_sample_points(original_dim, sample_u_num)
        print(f"sample_points: {len(sample_points)}")
        best = sample_points[0]

        while len(sample_points) > 1:
            best_changed_mark = True

            # single iteration path construction
            while (best_changed_mark):
                best_changed_mark = False

                # constraints construction
                A = []
                j = 0
                while j < len(sample_points):
                    if best.id != sample_points[j].id:
                        a = None
                        if epsilon <= 0.005:
                            a = best.coord - sample_points[j].coord
                        else:
                            a = best.coord - (1 - epsilon + 0.005) * sample_points[j].coord
                        a = a.tolist()
                        a.insert(0, 0)
                        A.append(a)
                        j += 1
                    else:
                        sample_points.remove(sample_points[j])
                if(len(A) == 0 and len(sample_points) < 1):
                    print(f"A is empty")
                    break
                A = np.vstack([A, original_utility_range])
                
                # find a center point in constraints set A
                center_point = Point(original_dim)
                center_point.coord = best.sampledU
                t = np.dot(A[:,1:], center_point.coord) + A[:,0]
                if np.all(t >= 1e-20):
                    print(f"center_point is in A")
                else:
                    print(f"center_point is Not in A")
                    # generate a utility vector in utility range
                    center_point.coord = best.sampledU
                    while True:
                        t = np.dot(original_utility_range[:,1:], center_point.coord) + original_utility_range[:,0]
                        if np.all(t >= 1e-20):
                            best = pset.find_top_k_batch(center_point, 1)[0]
                            best.sampledU = center_point.coord
                            best_changed_mark = True
                            break
                        #center_point.coord = np.random.uniform(0, 1, original_dim)
                        center_point.coord = find_center_point(original_utility_range, original_dim)
                        center_point.coord = center_point.coord / np.sum(center_point.coord)
                    continue
                    
                '''
                test_set = PointSet()
                test_set.points = sample_points
                test_set.points.append(best)
                p = test_set.find_top_k_batch(center_point, 1)[0]
                print(f"best.id: {best.id}, utility: {best.dot_prod(center_point)}, p.id: {p.id}, utility: {p.dot_prod(center_point)}")
                if best.id != p.id:
                    print(f"best.id: {best.id}, p.id: {p.id}")
                    return
                sample_points.remove(best)
                '''

                center_point = center_point.coord
                # ray shooting method finding the boundary of the polytope
                for _ in range(round_num):
                    t1 = time.time()
                    directions = np.random.normal(0, 1, (sample_ray_num, original_dim))
                    directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
                    # generate unit directions and negative unit directions
                    unit_directions = np.eye(original_dim)
                    neg_unit_directions = -np.eye(original_dim)
                    directions = np.vstack([directions, unit_directions, neg_unit_directions])
                    t_utility_vectors, constraint_indices = calculate_constraints_batch_multi_directions(A, directions, center_point)
                    t2 = time.time()
                    print(f"calculate_constraints_batch_multi_directions time: {t2 - t1}")


                    dir = np.array([u.coord - center_point])
                    dir = dir / np.linalg.norm(dir)
                    _, constraint_indices2 = calculate_constraints_batch_multi_directions(A, dir, center_point)
                    if constraint_indices2[0] != -1 and constraint_indices2[0] < len(sample_points):
                        if sample_points[constraint_indices2[0]].mark == False:
                            print(f"constraint_indices2: {constraint_indices2}, point: {sample_points[constraint_indices2[0]].id}")
                            print("best norm", np.linalg.norm(best.coord))
                            print("sample_points norm", np.linalg.norm(sample_points[constraint_indices2[0]].coord))
                        else:
                            print(f"case 1: constraint_indices2: [], point: ")
                    else:
                        print(f"case 2: constraint_indices2: [], point: {constraint_indices2[0]}")


                    t3 = time.time()
                    # get the valid indices and t values
                    valid_mask = (constraint_indices != -1) & \
                                (constraint_indices < len(sample_points)) & \
                                ~np.array([sample_points[i].mark if i != -1 and i < len(sample_points) else True for i in constraint_indices])
                    valid_ci = constraint_indices[valid_mask]
                    valid_t = t_utility_vectors[valid_mask]
                    # get the unique values and counts, and create the sorted array
                    unique_ci, indices, counts = np.unique(valid_ci, return_index=True, return_counts=True)
                    first_t = valid_t[indices]
                    sort_idx = np.lexsort((unique_ci, -counts))
                    valid_pairs = [[unique_ci[i], first_t[i]] for i in sort_idx]
                    for tt, t_val in valid_pairs:
                        print(f"sample_points[tt[0]].id: {sample_points[tt].id}")
                    t4 = time.time()
                    print(f"get valid_pairs time: {t4 - t3}")

                    for constraint_index, t_val in valid_pairs:
                        num_of_question += 1
                        print(f"Question {num_of_question}: Point {best.id}, Utility {best.dot_prod(u)}, Point {sample_points[constraint_index].id}, Utility {sample_points[constraint_index].dot_prod(u)}")
                        if best.dot_prod(u) < sample_points[constraint_index].dot_prod(u):
                            a = sample_points[constraint_index].coord - best.coord
                            a = a.tolist()
                            a.insert(0, 0)
                            original_utility_range = np.vstack([original_utility_range, a])
                            best.mark = True
                            best = sample_points[constraint_index]
                            best_changed_mark = True
                            best.sampledU = t_val
                            break
                        else:
                            a = best.coord - sample_points[constraint_index].coord
                            a = a.tolist()
                            a.insert(0, 0)
                            original_utility_range = np.vstack([original_utility_range, a])
                            sample_points[constraint_index].mark = True
                            
                    if best_changed_mark:
                        break

            
            if old_num_of_question == num_of_question and not all_considered:
                all_considered = True
                sample_points = []
                for p in pset.points:
                    if p.mark == False:
                        sample_points.append(p)
            elif not all_considered:
                # base_utility = R.find_feasible()
                base_utility = Point(original_dim)
                t1 = time.time()
                base_utility.coord = find_center_point(original_utility_range, original_dim)
                t2 = time.time()
                print(f"find_center_point time: {t2 - t1}")
                samples_utility_possible = []
                samples_utility_possible = sample_utilities_batch(original_dim, sample_u_num, original_utility_range, base_utility)
                print(f"samples_utility_possible: {len(samples_utility_possible)}")

                # stopping condition checking
                sample_points = [best]
                utility_vectors = np.array([u.coord for u in samples_utility_possible])
                all_utility_values = np.dot(utility_vectors, points_array.T)
                max_indices = np.argmax(all_utility_values, axis=1)
                max_values = all_utility_values[np.arange(len(all_utility_values)), max_indices]
                threshold_matrix = (1 - epsilon) * max_values.reshape(-1, 1)
                best_utility_values = (np.dot(utility_vectors, best.coord.T)).reshape(-1, 1)
                candidate_mask = best_utility_values > threshold_matrix + 1e-5
                for i, mask in enumerate(candidate_mask):
                    if mask == False and pset.points[max_indices[i]].mark == False and pset.points[max_indices[i]] not in sample_points:
                        sample_points.append(pset.points[max_indices[i]])
                        pset.points[max_indices[i]].sampledU = samples_utility_possible[i].coord
                if len(sample_points) <= 1:
                    all_considered = True
                    sample_points = []
                    for p in pset.points:
                        if p.mark == False:
                            sample_points.append(p)
                old_num_of_question = num_of_question
            else:
                sample_points = []
            print(f"sample_points: {len(sample_points)}")

        best.printAlgResult("APCEnhance", num_of_question, start_time, 0)
        groudtruth = pset.find_top_k(u, 1)[0]
        rr = 1 - best.dot_prod(u) / groudtruth.dot_prod(u)
        print(f"best.id: {best.id}, best.dot_prod(u): {best.dot_prod(u)}, groudtruth.dot_prod(u): {groudtruth.dot_prod(u)}")
        print("Regret: ", rr)
        best.printToFile(f"APCEnhance{u_r}-{round_num}", dataset_name, epsilon, num_of_question, start_time, rr)
        # pset.printFinal(best, num_of_question, u, "APC", dataset_name, epsilon)
    else:
        # projected points
        projected_points = jl_projection(pset.points, projected_dim)
        projected_points_array = np.array([p.coord for p in projected_points.points])
        # projected utility range
        projected_utility_range = []
        for j in range(projected_dim):
            a = np.zeros(projected_dim + 1)
            a[j + 1] = 1
            projected_utility_range.append(a.tolist())
        a = np.zeros(projected_dim + 1)
        a[0] = 1.0000000001
        for j in range(projected_dim):
            a[j + 1] = -1
        projected_utility_range.append(a.tolist())
        projected_utility_range = np.array(projected_utility_range)

        points_array = np.array([p.coord for p in pset.points])
        # original utility range
        original_utility_range = []
        # x >= 0 => 0 <= 0 + xi
        for j in range(original_dim):
            a = np.zeros(original_dim + 1)
            a[j + 1] = 1
            original_utility_range.append(a.tolist())
        # x1 + x2 +... + xd <= 1    ==>     0 <= 1 - x1 - x2 -... - xd
        a = np.zeros(original_dim + 1)
        a[0] = 1.0000000001
        for j in range(original_dim):
            a[j + 1] = -1
        original_utility_range.append(a.tolist())
        original_utility_range = np.array(original_utility_range)
        
        all_considered = False
        sample_points = projected_points.generate_sample_points(projected_dim, sample_u_num)
        print(f"sample_points: {len(sample_points)}")
        best = sample_points[0]

        while len(sample_points) > 1:
            best_changed_mark = True

            # single iteration path construction
            while (best_changed_mark):
                best_changed_mark = False

                # constraints construction
                A = []
                j = 0
                while j < len(sample_points):
                    if best.id != sample_points[j].id:
                        a = None
                        if epsilon <= 0.005:
                            a = best.coord - sample_points[j].coord
                        else:
                            a = best.coord - (1 - epsilon + 0.005) * sample_points[j].coord
                        a = a.tolist()
                        a.insert(0, 0)
                        A.append(a)
                        j += 1
                    else:
                        sample_points.remove(sample_points[j])
                if(len(A) == 0 and len(sample_points) < 1):
                    print(f"A is empty")
                    break
                A = np.vstack([A, projected_utility_range])
                
                # find a center point in constraints set A
                center_point = Point(projected_dim)
                if best.sampledU is not None:
                    center_point.coord = best.sampledU
                else:
                    center_point = Point(projected_dim)
                t = np.dot(A[:,1:], center_point.coord) + A[:,0]
                if np.all(t >= 1e-20):
                    print(f"center_point is in A")
                else:
                    print(f"center_point is Not in A")
                    # generate a utility vector in utility range
                    if best.sampledU is not None:
                        center_point.coord = best.sampledU
                    else:
                        center_point = Point(projected_dim)
                    while True:
                        t = np.dot(projected_utility_range[:,1:], center_point.coord) + projected_utility_range[:,0]
                        if np.all(t >= 1e-20):
                            best = projected_points.find_top_k_batch(center_point, 1)[0]
                            best.sampledU = center_point.coord
                            best_changed_mark = True
                            break
                        #center_point.coord = np.random.uniform(0, 1, original_dim)
                        center_point.coord = find_center_point(projected_utility_range, projected_dim)
                        center_point.coord = center_point.coord / np.sum(center_point.coord)
                    continue

                center_point = center_point.coord
                # ray shooting method finding the boundary of the polytope
                for _ in range(round_num):
                    t1 = time.time()
                    directions = np.random.normal(0, 1, (sample_ray_num, projected_dim))
                    directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
                    # generate unit directions and negative unit directions
                    unit_directions = np.eye(projected_dim)
                    neg_unit_directions = -np.eye(projected_dim)
                    directions = np.vstack([directions, unit_directions, neg_unit_directions])
                    t_utility_vectors, constraint_indices = calculate_constraints_batch_multi_directions(A, directions, center_point)
                    t2 = time.time()
                    print(f"calculate_constraints_batch_multi_directions time: {t2 - t1}")

                    t3 = time.time()
                    # get the valid indices and t values
                    valid_mask = (constraint_indices != -1) & \
                                (constraint_indices < len(sample_points)) & \
                                ~np.array([sample_points[i].mark if i != -1 and i < len(sample_points) else True for i in constraint_indices])
                    valid_ci = constraint_indices[valid_mask]
                    valid_t = t_utility_vectors[valid_mask]
                    # get the unique values and counts, and create the sorted array
                    unique_ci, indices, counts = np.unique(valid_ci, return_index=True, return_counts=True)
                    first_t = valid_t[indices]
                    sort_idx = np.lexsort((unique_ci, -counts))
                    valid_pairs = [[unique_ci[i], first_t[i]] for i in sort_idx]
                    for tt, t_val in valid_pairs:
                        print(f"sample_points[tt[0]].id: {sample_points[tt].id}")
                    t4 = time.time()
                    print(f"get valid_pairs time: {t4 - t3}")

                    for constraint_index, t_val in valid_pairs:
                        num_of_question += 1
                        print(f"Question {num_of_question}: Point {best.id}, Point {sample_points[constraint_index].id}")
                        if pset.points[best.id].dot_prod(u) < pset.points[sample_points[constraint_index].id].dot_prod(u):
                            a = pset.points[sample_points[constraint_index].id].coord - pset.points[best.id].coord
                            a = a.tolist()
                            a.insert(0, 0)
                            original_utility_range = np.vstack([original_utility_range, a])
                            a2 = best.coord - sample_points[constraint_index].coord
                            b2 = (dim_error / 2) * (np.linalg.norm(pset.points[best.id].coord)**2 + 
                                                np.linalg.norm(pset.points[sample_points[constraint_index].id].coord)**2) + dim_error
                            a2 = np.append([b2], a2)
                            projected_utility_range = np.vstack([projected_utility_range, a2])
                            best.mark = True
                            pset.points[best.id].mark = True
                            best = sample_points[constraint_index]
                            best_changed_mark = True
                            best.sampledU = t_val
                            break
                        else:
                            a = pset.points[best.id].coord - pset.points[sample_points[constraint_index].id].coord
                            a = a.tolist()
                            a.insert(0, 0)
                            original_utility_range = np.vstack([original_utility_range, a])
                            a2 = sample_points[constraint_index].coord - best.coord
                            b2 = (dim_error / 2) * (np.linalg.norm(pset.points[sample_points[constraint_index].id].coord)**2 + 
                                                np.linalg.norm(pset.points[best.id].coord)**2) + dim_error
                            a2 = np.append([b2], a2)
                            projected_utility_range = np.vstack([projected_utility_range, a2])
                            sample_points[constraint_index].mark = True
                            pset.points[sample_points[constraint_index].id].mark = True
                            
                    if best_changed_mark:
                        break

            
            if old_num_of_question == num_of_question and not all_considered:
                all_considered = True
                sample_points = []
                for p in pset.points:
                    if p.mark == False:
                        sample_points.append(p)
                projected_utility_range = original_utility_range
                projected_dim = original_dim
                projected_points = pset
                projected_points_array = points_array
                best = pset.points[best.id]
            elif not all_considered:
                # base_utility = R.find_feasible()
                base_utility = Point(projected_dim)
                t1 = time.time()
                base_utility.coord = find_center_point(projected_utility_range, projected_dim)
                t2 = time.time()
                print(f"find_center_point time: {t2 - t1}")
                samples_utility_possible = []
                samples_utility_possible = sample_utilities_batch(projected_dim, sample_u_num, projected_utility_range, base_utility)
                print(f"samples_utility_possible: {len(samples_utility_possible)}")

                # stopping condition checking
                sample_points = [best]
                utility_vectors = np.array([u.coord for u in samples_utility_possible])
                all_utility_values = np.dot(utility_vectors, projected_points_array.T)
                max_indices = np.argmax(all_utility_values, axis=1)
                max_values = all_utility_values[np.arange(len(all_utility_values)), max_indices]
                threshold_matrix = (1 - epsilon) * max_values.reshape(-1, 1)
                best_utility_values = (np.dot(utility_vectors, best.coord.T)).reshape(-1, 1)
                candidate_mask = best_utility_values > threshold_matrix + 1e-5
                for i, mask in enumerate(candidate_mask):
                    if mask == False and projected_points.points[max_indices[i]].mark == False and projected_points.points[max_indices[i]] not in sample_points:
                        sample_points.append(projected_points.points[max_indices[i]])
                        projected_points.points[max_indices[i]].sampledU = samples_utility_possible[i].coord
                if len(sample_points) <= 1:
                    # back to original points
                    all_considered = True
                    sample_points = []
                    for p in pset.points:
                        if p.mark == False:
                            sample_points.append(p)
                    projected_utility_range = original_utility_range
                    projected_dim = original_dim
                    projected_points = pset
                    projected_points_array = points_array
                    best = pset.points[best.id]
                old_num_of_question = num_of_question
            else:
                sample_points = []
            print(f"sample_points: {len(sample_points)}")

        best.printAlgResult("APC", num_of_question, start_time, 0)
        groudtruth = pset.find_top_k(u, 1)[0]
        rr = 1 - best.dot_prod(u) / groudtruth.dot_prod(u)
        print(f"best.id: {best.id}, best.dot_prod(u): {best.dot_prod(u)}, groudtruth.dot_prod(u): {groudtruth.dot_prod(u)}")
        print("Regret: ", rr)
        best.printToFile(f"APC{u_r}-{round_num}", dataset_name, epsilon, num_of_question, start_time, rr)
        # pset.printFinal(best, num_of_question, u, "APC", dataset_name, epsilon)