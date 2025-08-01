import cvxpy as cp
import numpy as np

# 给定的点 p 和维度 d
p = np.array([1, 2, 3])
d = len(p)

# 超平面系数，示例
A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# 创建优化变量
x = cp.Variable(d)

# 目标函数，最小化负的欧氏距离
objective = cp.Minimize(-cp.norm(x - p))

# 约束条件
constraints = [A @ x <= np.zeros(A.shape[0])]

# 定义并解决问题
problem = cp.Problem(objective, constraints)
problem.solve()

# 输出结果
print("最优解 x =", x.value)
print("最大距离 =", -problem.value)  # 输出的是最大距离，注意取负值
