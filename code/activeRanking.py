import time
import numpy as np
from structure.point_set import PointSet
from structure.hyperplane_set import HyperplaneSet
from structure.hyperplane import Hyperplane
from structure.point import Point

def active_ranking(pset: PointSet, u: Point, epsilon: float, dataset_name) -> int:
    start_time = time.time()
    
    dim = pset.points[0].dim
    num_question = 0
    M = len(pset.points)
    
    # 随机化点集
    np.random.shuffle(pset.points)
    
    # 初始化
    R = HyperplaneSet(dim)
    current = PointSet()
    current.points.append(pset.points[0])
    
    # 按顺序存储所有点
    for i in range(1, M):  # 比较：p_set包含所有点
        if i % 1000 == 0:
            print(i)
        
        num_point = len(current.points)
        place = 0  # 点插入到current_use中的位置
        
        # 找到要询问用户的问题
        for j in range(num_point):
            h = Hyperplane(p1=pset.points[i], p2=current.points[j])

            relation = R.check_relation(h)
            v1 = u.dot_prod(pset.points[i])
            v2 = u.dot_prod(current.points[j])
            # print(v1, v2)
            # 如果相交，计算距离
            if relation == 0:
                num_question += 1
                
                # 比较两个点的效用值
                v1 = u.dot_prod(pset.points[i])
                v2 = u.dot_prod(current.points[j])
                
                if v1 > v2:
                    h = Hyperplane(p1=current.points[j], p2=pset.points[i])
                    R.hyperplanes.append(h)
                    if R.set_ext_pts() == False:
                        R.hyperplanes.pop()
                    break
                else:
                    h = Hyperplane(p1=pset.points[i], p2=current.points[j])
                    R.hyperplanes.append(h)
                    if R.set_ext_pts() == False:
                        R.hyperplanes.pop()
                    place = j + 1

                
            elif relation == -1:
                place = j + 1
            else:
                break
        
        current.points.insert(place, pset.points[i])
        '''
        if len(current.points) > 100:
            for i in range(100):
                print(current.points[i].id, current.points[i].dot_prod(u))
            print("--------------------------------")
        else:
            for i in range(len(current.points)):
                print(current.points[i].id, current.points[i].dot_prod(u))
            print("--------------------------------")
        '''

    # print results
    result = current.points[0]
    groudtruth = pset.find_top_k(u, 1)[0]
    rr = 1 - result.dot_prod(u) / groudtruth.dot_prod(u)
    print("Regret: ", rr)
    result.printAlgResult("ActiveRanking", num_question, start_time, 0)
    result.printToFile("ActiveRanking", dataset_name, epsilon, num_question, start_time, rr)

    return num_question




def active_rankingMiddle(pset: PointSet, u: Point, epsilon: float, dataset_name) -> int:
    start_time = time.time()
    groudtruth = pset.find_top_k(u, 1)[0]
    groudtruth_utility = groudtruth.dot_prod(u)
    dim = pset.points[0].dim
    num_question = 0
    M = len(pset.points)
    question_threshold = 10
    
    # 随机化点集
    np.random.shuffle(pset.points)
    
    # 初始化
    R = HyperplaneSet(dim)
    current = PointSet()
    current.points.append(pset.points[0])
    
    # 按顺序存储所有点
    for i in range(1, M):  # 比较：p_set包含所有点
        if i % 1000 == 0:
            print(i)
        
        num_point = len(current.points)
        place = 0  # 点插入到current_use中的位置
        
        # 找到要询问用户的问题
        for j in range(num_point):
            h = Hyperplane(p1=pset.points[i], p2=current.points[j])

            relation = R.check_relation(h)
            v1 = u.dot_prod(pset.points[i])
            v2 = u.dot_prod(current.points[j])
            # print(v1, v2)
            # 如果相交，计算距离
            if relation == 0:
                num_question += 1
                
                # 比较两个点的效用值
                v1 = u.dot_prod(pset.points[i])
                v2 = u.dot_prod(current.points[j])
                
                if v1 > v2:
                    h = Hyperplane(p1=current.points[j], p2=pset.points[i])
                    R.hyperplanes.append(h)
                    if R.set_ext_pts() == False:
                        R.hyperplanes.pop()
                    
                else:
                    h = Hyperplane(p1=pset.points[i], p2=current.points[j])
                    R.hyperplanes.append(h)
                    if R.set_ext_pts() == False:
                        R.hyperplanes.pop()
                    place = j + 1

                current_best_point = current.points[0]
                current_best_point_utility = current_best_point.dot_prod(u)
                middle_rr = 1 - current_best_point_utility / groudtruth_utility
                print(f"current_best_point: {current_best_point.id}, current_best_point_utility: {current_best_point_utility}, groudtruth_utility: {groudtruth_utility}, rr: {middle_rr}")
                current_best_point.printMiddleResultToFile("ActiveRanking", dataset_name, epsilon, num_question, start_time, middle_rr)
                if num_question >= question_threshold:
                    return

                if v1 > v2:
                    break

                
            elif relation == -1:
                place = j + 1
            else:
                break
        
        current.points.insert(place, pset.points[i])
                


    # print results
    result = current.points[0]
    groudtruth = pset.find_top_k(u, 1)[0]
    rr = 1 - result.dot_prod(u) / groudtruth.dot_prod(u)
    print("Regret: ", rr)
    result.printAlgResult("ActiveRanking", num_question, start_time, 0)
    result.printToFile("ActiveRanking", dataset_name, epsilon, num_question, start_time, rr)

    return num_question



