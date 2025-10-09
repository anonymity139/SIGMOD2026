import time
import numpy as np
import math
from typing import List, Optional
from structure.point_set import PointSet
from structure.hyperplane_set import HyperplaneSet
from structure.hyperplane import Hyperplane
from structure.point import Point
from qpsolvers import solve_qp
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
import cvxpy as cp
import scipy.optimize as opt

Lnum = 50
pi = 3.1415


class Cluster:
    """聚类类"""
    def __init__(self, dim):
        self.center = Point(dim)
        self.h_set = []


class SNode:
    """球形树节点类"""
    def __init__(self, dim):
        self.dim = dim
        self.center = None
        self.angle = 0.0
        self.hyper = []
        self.child = []
        self.is_leaf = False


def find_estimate(V: List[Point]) -> Optional[Point]:
    if len(V) == 0:
        return None
    elif len(V) == 1:
        return Point(dim=V[0].dim, coord=V[0].coord.copy())
    
    dim = V[0].dim
    zero = 0.00000001
    
    # 使用二次规划求解
    # min 1/2 x^T*G*x + g0*x
    n = dim
    
    # 目标函数矩阵 G (单位矩阵)
    G = np.eye(n)
    
    # 目标函数向量 g0 (零向量)
    g0 = np.zeros(n)
    
    # 等式约束 CE*x = ce0 (无等式约束)
    m = 0
    CE = None
    ce0 = None
    
    # 不等式约束 CI*x >= ci0
    p = len(V)
    CI = np.zeros((p, n))
    for i in range(p):
        for j in range(n):
            CI[i][j] = - V[i].coord[j]
    print(f"CI: {CI}")
    
    ci0 = np.full(p, -1.0)
    
    try:
        # 使用qpsolvers求解二次规划问题
        x = solve_qp(P=G, q=g0, G=CI, h=ci0, solver='osqp')
        if x is None:
            x = Point(dim=dim, coord=np.zeros(dim))
            '''
            # average of all points
            for i in range(1, len(V)):
                for j in range(dim):
                    x.coord[j] += V[i].coord[j]
            for i in range(dim):
                x.coord[i] /= len(V)
            '''
            return x
            
        # 将结果转换为Point对象
        estimate = Point(dim=dim, coord=x)
        return estimate
        
    except Exception as e:
        print(f"Quadratic programming error: {e}")
        return None


def find_estimate_cvxpy(V: List[Point]) -> Optional[Point]:
    """
    使用CVXPY求解二次规划问题 - 更精准的凸优化库
    """
    if len(V) == 0:
        return None
    elif len(V) == 1:
        return Point(dim=V[0].dim, coord=V[0].coord.copy())
    
    dim = V[0].dim
    
    # 定义优化变量
    x = cp.Variable(dim)
    
    # 目标函数：minimize ||x||^2
    objective = cp.Minimize(cp.sum_squares(x))
    
    # 约束条件：V[i]^T * x >= -1 for all i
    constraints = []
    for i in range(len(V)):
        constraints.append(V[i].coord @ x >= -1)
    
    # 定义问题
    problem = cp.Problem(objective, constraints)
    
    try:
        # 求解
        problem.solve(verbose=False, solver=cp.OSQP)
        
        if problem.status == "optimal":
            result = Point(dim=dim, coord=x.value)
            return result
        elif problem.status == "infeasible":
            print(f"CVXPY: 问题不可行")
        elif problem.status == "unbounded":
            print(f"CVXPY: 问题无界")
        else:
            print(f"CVXPY: 求解失败，状态: {problem.status}")
            
    except Exception as e:
        print(f"CVXPY error: {e}")
    
    # 回退到平均方法
    estimate = Point(dim=dim, coord=np.zeros(dim))
    for i in range(len(V)):
        estimate.coord += V[i].coord
    estimate.coord /= len(V)
    return estimate


def find_estimate_scipy(V: List[Point]) -> Optional[Point]:
    """
    使用SciPy求解二次规划问题 - 更稳定的数值优化
    """
    if len(V) == 0:
        return None
    elif len(V) == 1:
        return Point(dim=V[0].dim, coord=V[0].coord.copy())
    
    dim = V[0].dim
    
    # 目标函数：minimize 1/2 * x^T * G * x + g^T * x
    G = np.eye(dim)
    g = np.zeros(dim)
    
    # 约束：V[i]^T * x >= -1
    A = np.array([V[i].coord for i in range(len(V))])
    b = np.full(len(V), -1.0)
    
    try:
        # 使用SLSQP求解器
        result = opt.minimize(
            fun=lambda x: 0.5 * x.T @ G @ x + g.T @ x,
            x0=np.zeros(dim),
            method='SLSQP',
            constraints={'type': 'ineq', 'fun': lambda x: A @ x - b},
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            estimate = Point(dim=dim, coord=result.x)
            return estimate
        else:
            print(f"SciPy: 求解失败 - {result.message}")
            
    except Exception as e:
        print(f"SciPy error: {e}")
    
    # 回退到平均方法
    estimate = Point(dim=dim, coord=np.zeros(dim))
    for i in range(len(V)):
        estimate.coord += V[i].coord
    estimate.coord /= len(V)
    return estimate


def find_estimate_quadprog(V: List[Point]) -> Optional[Point]:
    """
    使用quadprog求解二次规划问题 - 专门的高精度QP求解器
    """
    if len(V) == 0:
        return None
    elif len(V) == 1:
        return Point(dim=V[0].dim, coord=V[0].coord.copy())
    
    dim = V[0].dim
    
    # 目标函数：minimize 1/2 * x^T * G * x + g^T * x
    G = np.eye(dim)
    g = np.zeros(dim)
    
    # 约束：V[i]^T * x >= 1
    A = np.array([-V[i].coord for i in range(len(V))])
    b = np.full(len(V), -1.0)
    
    try:
        # 使用quadprog求解器
        x = solve_qp(P=G, q=g, G=A, h=b, solver='quadprog')
        
        if x is not None:
            estimate = Point(dim=dim, coord=x)
            return estimate
        else:
            # average of all points
            estimate = Point(dim=dim, coord=np.zeros(dim))
            for i in range(len(V)):
                estimate.coord += V[i].coord
            estimate.coord /= len(V)
            return estimate
            
    except Exception as e:
        print(f"QuadProg error: {e}")
    
    # 回退到平均方法
    estimate = Point(dim=dim, coord=np.zeros(dim))
    for i in range(len(V)):
        estimate.coord += V[i].coord
    estimate.coord /= len(V)
    return estimate


def hyperplane_normalize(hyper: Hyperplane):
    dim = hyper.dim
    norm_length = np.linalg.norm(hyper.norm)
    if norm_length > 0:
        hyper.norm = hyper.norm / norm_length


def point_normalize(p: Point):
    dim = p.dim
    norm_length = np.linalg.norm(p.coord)
    if norm_length > 0:
        p.coord = p.coord / norm_length
    else:
        print("point normalize failed")


def cosine0(h1: np.ndarray, h2: np.ndarray, dim: int) -> float:
    sum_val = np.dot(h1, h2)
    s_h1 = np.linalg.norm(h1)
    s_h2 = np.linalg.norm(h2)
    
    if s_h1 == 0 or s_h2 == 0:
        return 0.0
    
    return sum_val / (s_h1 * s_h2)


def orthogonality(h1: np.ndarray, h2: np.ndarray, dim: int) -> float:
    value = cosine0(h1, h2, dim)
    if value >= 0:
        return 1 - value
    else:
        return 1 + value


def upper_orthog(n: Point, node: SNode) -> float:
    alpha0 = cosine0(n.coord, node.center.coord, n.dim)  # cos(a)
    alpha0 = math.acos(max(-1, min(1, alpha0)))  # angle
    theta0 = math.acos(max(-1, min(1, node.angle)))
    
    if (alpha0 - theta0) < pi / 2 and (alpha0 + theta0) > pi / 2:
        return 1
    else:
        v1 = abs(math.cos(alpha0 + theta0))
        v2 = abs(math.cos(alpha0 - theta0))
        if v1 < v2:
            return 1 - v1
        else:
            return 1 - v2


def lower_orthog(n: Point, node: SNode) -> float:
    alpha0 = cosine0(n.coord, node.center.coord, n.dim)  # cos(a)
    alpha0 = math.acos(max(-1, min(1, alpha0)))  # angle
    theta0 = math.acos(max(-1, min(1, node.angle)))
    
    if alpha0 < theta0 or (alpha0 + theta0) > pi:
        return 0
    else:
        v1 = abs(math.cos(alpha0 + theta0))
        v2 = abs(math.cos(alpha0 - theta0))
        if v1 > v2:
            return 1 - v1
        else:
            return 1 - v2


def k_means_cosine(hyper: List[Hyperplane]) -> List[Cluster]:
    M = len(hyper)
    if M == 0:
        return []
        
    dim = hyper[0].dim
    seg = M // Lnum
    
    # precompute all hyperplane norms
    hyper_norms = np.array([h.norm for h in hyper])  # (M, dim)
    
    # initialize cluster centers
    clu = []
    for i in range(Lnum):
        if seg * i < M:
            c = Cluster(dim)
            c.center.coord = hyper[seg * i].norm.copy()
            clu.append(c)
    
    # convert to numpy array for batch calculation
    clu_centers = np.array([c.center.coord for c in clu])  # (num_clusters, dim)
    
    shift = 9999
    max_iter = 50
    iter_count = 0
    while shift >= 0.1 and iter_count < max_iter:
        iter_count += 1
        shift = 0
        
        # 计算余弦相似度：cos(θ) = (a·b) / (||a|| × ||b||)
        # 使用更高效的向量化方法
        similarities = cosine_similarity(hyper_norms, clu_centers)  # (M, num_clusters)
        
        # find the most similar cluster for each hyperplane
        assignments = np.argmax(similarities, axis=1)  # (M,)
        
        # update cluster centers
        for i in range(len(clu)):
            # find the hyperplanes belong to the current cluster
            mask = (assignments == i)
            if np.any(mask):
                # calculate the new center (average)
                new_center = np.mean(hyper_norms[mask], axis=0)
                
                # calculate the center shift
                center_shift = np.sum(np.abs(new_center - clu_centers[i]))
                shift += center_shift
                
                # update the center
                clu_centers[i] = new_center
                clu[i].center.coord = new_center.copy()
                
                # update the hyperplane list
                clu[i].h_set = [hyper[j] for j in range(M) if mask[j]]
            else:
                # if the cluster is empty, keep the original center
                clu[i].h_set = []

        print(f"shift: {shift}")

    # delete the cluster with no members
    clu = [c for c in clu if len(c.h_set) > 0]
    
    return clu


def cap_construction(node: SNode):
    V = []
    M = len(node.hyper)
    if M == 0:
        return
        
    dim = node.hyper[0].dim
    for i in range(M):
        pt = Point(dim=dim, coord=node.hyper[i].norm.copy())
        V.append(pt)
    
    node.center = find_estimate_quadprog(V)
    if node.center is not None:
        point_normalize(node.center)
        
        node.angle = cosine0(node.center.coord, node.hyper[0].norm, dim)
        for i in range(1, M):
            angle = cosine0(node.center.coord, node.hyper[i].norm, dim)
            if angle < node.angle:
                node.angle = angle
    else:
        print("node center failed")


def build_spherical_tree(hyper: List[Hyperplane], node: SNode):
    M = len(hyper)
    if M == 0:
        return
        
    dim = hyper[0].dim
    
    # 构建叶子节点
    if M <= 50:
        node.hyper = hyper.copy()
        node.is_leaf = True
        cap_construction(node)
    else:  # 构建内部节点
        # 获得所有聚类
        clu = k_means_cosine(hyper)
        clu_size = len(clu)
        
        if clu_size == 1:
            # 如果只有一个聚类，分割它
            c = Cluster(dim)
            countt = len(clu[0].h_set)
            for j in range(countt // 2):
                c.h_set.append(clu[0].h_set[0])
                clu[0].h_set.pop(0)
            clu.append(c)
            clu_size = len(clu)
        
        # 为每个聚类构建节点
        for i in range(clu_size):
            s_n = SNode(dim)
            build_spherical_tree(clu[i].h_set, s_n)
            node.child.append(s_n)
        
        # 将所有超平面添加到当前节点
        node.hyper = hyper.copy()
        node.is_leaf = False
        cap_construction(node)


def spherical_cap_pruning(q: Point, S: List[SNode]) -> List[SNode]:
    M = len(S)
    if M == 0:
        return []
        
    Q = [S[0]]
    maxLB = lower_orthog(q, S[0])
    
    for i in range(1, M):
        UB = upper_orthog(q, S[i])
        LB = lower_orthog(q, S[i])
        
        if UB > maxLB:
            # 处理Q
            index = 0
            q_size = len(Q)
            while index < q_size:
                UB0 = upper_orthog(q, Q[index])
                if UB0 < LB:
                    Q.pop(index)
                    q_size -= 1
                else:
                    index += 1
            
            # 处理maxLB
            if LB > maxLB:
                maxLB = LB
            
            # 插入到Q中
            q_size = len(Q)
            a = 0
            for a in range(q_size):
                UB0 = upper_orthog(q, Q[a])
                if UB < UB0:
                    break
            Q.insert(a, S[i])
    
    return Q


def orthogonal_search(node: SNode, q: Point, best: Optional[Hyperplane] = None) -> Optional[Hyperplane]:
    if node.is_leaf:
        M = len(node.hyper)
        if best is None:
            ortho_value = 0
        else:
            ortho_value = orthogonality(best.norm, q.coord, q.dim)
            
        for i in range(M):
            v = orthogonality(node.hyper[i].norm, q.coord, q.dim)
            if v > ortho_value:
                ortho_value = v
                best = node.hyper[i]
        return best

    Q = spherical_cap_pruning(q, node.child)
    S = Q.copy()  # 使用列表作为栈
    Q.clear()
    
    while S:
        t = S.pop()
        UB_t = upper_orthog(q, t)
        if best is None or UB_t > orthogonality(best.norm, q.coord, q.dim):
            best = orthogonal_search(t, q, best)
    
    return best

def preference_learning(original_set: PointSet, u: Point, epsilon: float, dataset_name) -> int:
    start_time = time.time()
    

    sample_num = 500
    pset = PointSet()
    if len(original_set.points) < sample_num:
        M = len(original_set.points)
        for i in range(M):
            pset.points.append(original_set.points[i])
        np.random.shuffle(pset.points)
    else:
        np.random.shuffle(original_set.points)
        for i in range(0, sample_num):
            pset.points.append(original_set.points[i])
    
    dim = pset.points[0].dim
    M = len(pset.points)
    accuracy = 0
    de_accuracy = 100
    num_of_question = 0

    # 法向量
    V = []
    for i in range(dim):
        b = Point(dim=dim)
        for j in range(dim):
            if i == j:
                b.coord[j] = 1
            else:
                b.coord[j] = 0
        V.append(b)

    # 为每对点构建超平面
    h_set = []
    for i in range(M):
        for j in range(M):
            if i != j and not pset.points[i].is_same(pset.points[j]):
                h1 = Hyperplane(p1=pset.points[i], p2=pset.points[j])
                hyperplane_normalize(h1)
                h_set.append(h1)
                h2 = Hyperplane(p1=pset.points[j], p2=pset.points[i])
                hyperplane_normalize(h2)
                h_set.append(h2)

    stree_root = SNode(dim)
    build_spherical_tree(h_set, stree_root)

    # 初始化
    estimate_u = find_estimate_quadprog(V)
    if estimate_u is not None:
        point_normalize(estimate_u)
    print(f"estimate_u: {estimate_u.coord}")

    EQN_EPS = 0.00001  # 定义阈值
    while accuracy < 1 - EQN_EPS and de_accuracy > 1e-9:
        num_of_question += 1
        best = orthogonal_search(stree_root, estimate_u)
        
        '''
        value = -1
        for i in range(len(h_set)):
            v = orthogonality(h_set[i].norm, estimate_u.coord, dim)
            if v > value:
                value = v
                best = h_set[i]
        '''

        if best is not None:
            p = best.p1
            q = best.p2
            pt = Point(dim)
            
            # 模拟用户比较结果
            v1 = u.dot_prod(p)
            v2 = u.dot_prod(q)
            if v1 > v2:
                pt.coord = best.norm.copy()
            else:
                pt.coord = -best.norm.copy()

            V.append(pt)
            estimate_u = find_estimate_quadprog(V)
            if estimate_u is not None:
                for i in range(dim):
                    estimate_u.coord[i] = max(0, estimate_u.coord[i])
                point_normalize(estimate_u)
                ac = cosine0(u.coord, estimate_u.coord, dim)
                de_accuracy = abs(ac - accuracy)
                accuracy = ac
                print(f"estimate_u: {estimate_u.coord}")
                print(f"accuracy: {accuracy}, de_accuracy: {de_accuracy}")
                

    # print results
    result = original_set.find_top_k(estimate_u, 1)[0]
    groudtruth = original_set.find_top_k(u, 1)[0]
    rr = 1 - result.dot_prod(u) / groudtruth.dot_prod(u)
    print("Regret: ", rr)
    result.printAlgResult("PrefLearning", num_of_question, start_time, 0)
    result.printToFile("PrefLearning", dataset_name, epsilon, num_of_question, start_time, rr)

    return num_of_question





def preference_learningMiddle(original_set: PointSet, u: Point, epsilon: float, dataset_name) -> int:
    start_time = time.time()
    groudtruth = original_set.find_top_k(u, 1)[0]
    groudtruth_utility = groudtruth.dot_prod(u)
    question_threshold = 10
    
    sample_num = 500
    pset = PointSet()
    if len(original_set.points) < sample_num:
        M = len(original_set.points)
        for i in range(M):
            pset.points.append(original_set.points[i])
        np.random.shuffle(pset.points)
    else:
        np.random.shuffle(original_set.points)
        for i in range(0, sample_num):
            pset.points.append(original_set.points[i])
    
    dim = pset.points[0].dim
    M = len(pset.points)
    accuracy = 0
    de_accuracy = 100
    num_of_question = 0

    # 法向量
    V = []
    for i in range(dim):
        b = Point(dim=dim)
        for j in range(dim):
            if i == j:
                b.coord[j] = 1
            else:
                b.coord[j] = 0
        V.append(b)

    # 为每对点构建超平面
    h_set = []
    for i in range(M):
        for j in range(M):
            if i != j and not pset.points[i].is_same(pset.points[j]):
                h1 = Hyperplane(p1=pset.points[i], p2=pset.points[j])
                hyperplane_normalize(h1)
                h_set.append(h1)
                h2 = Hyperplane(p1=pset.points[j], p2=pset.points[i])
                hyperplane_normalize(h2)
                h_set.append(h2)

    stree_root = SNode(dim)
    build_spherical_tree(h_set, stree_root)

    # 初始化
    estimate_u = find_estimate_quadprog(V)
    if estimate_u is not None:
        point_normalize(estimate_u)
    print(f"estimate_u: {estimate_u.coord}")

    EQN_EPS = 0.00001  # 定义阈值
    while accuracy < 1 - EQN_EPS and de_accuracy > 1e-9:
        num_of_question += 1
        best = orthogonal_search(stree_root, estimate_u)
        
        '''
        value = -1
        for i in range(len(h_set)):
            v = orthogonality(h_set[i].norm, estimate_u.coord, dim)
            if v > value:
                value = v
                best = h_set[i]
        '''

        if best is not None:
            p = best.p1
            q = best.p2
            pt = Point(dim)
            
            # 模拟用户比较结果
            v1 = u.dot_prod(p)
            v2 = u.dot_prod(q)
            if v1 > v2:
                pt.coord = best.norm.copy()
            else:
                pt.coord = -best.norm.copy()

            V.append(pt)
            estimate_u = find_estimate_quadprog(V)
            for i in range(dim):
                estimate_u.coord[i] = max(0, estimate_u.coord[i])
            point_normalize(estimate_u)
            ac = cosine0(u.coord, estimate_u.coord, dim)
            de_accuracy = abs(ac - accuracy)
            accuracy = ac
            print(f"estimate_u: {estimate_u.coord}")
            print(f"accuracy: {accuracy}, de_accuracy: {de_accuracy}")

            current_best_point = original_set.find_top_k(estimate_u, 1)[0]
            current_best_point_utility = current_best_point.dot_prod(u)
            middle_rr = 1 - current_best_point_utility / groudtruth_utility
            print(f"current_best_point: {current_best_point.id}, current_best_point_utility: {current_best_point_utility}, groudtruth_utility: {groudtruth_utility}, rr: {middle_rr}")
            current_best_point.printMiddleResultToFile("PrefLearning", dataset_name, epsilon, num_of_question, start_time, middle_rr)
            if num_of_question >= question_threshold:
                return
                

    # print results
    result = original_set.find_top_k(estimate_u, 1)[0]
    groudtruth = original_set.find_top_k(u, 1)[0]
    rr = 1 - result.dot_prod(u) / groudtruth.dot_prod(u)
    print("Regret: ", rr)
    result.printAlgResult("PrefLearning", num_of_question, start_time, 0)
    result.printToFile("PrefLearning", dataset_name, epsilon, num_of_question, start_time, rr)

    return num_of_question




