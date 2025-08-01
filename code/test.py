import cdd as pycdd
from pprint import pprint
from structure.point_set import PointSet
from structure.point import Point
import numpy as np

# 创建测试数据
points = [
    [1, 0, 2, 0],    # 维度 (0,2)
    [1, 0, 3, 0],    # 维度 (0,2)
    [0, 2, 0, 1],    # 维度 (1,3)
    [0, 1, 2, 2],    # 维度 (1,2,3)
    [1, 1, 2, 0],    # 维度 (0,1,2)
]

point_set = PointSet()
for coords in points:
    point_set.add_point(Point(len(coords), coord=np.array(coords)))

# 按非零维度分组
groups = point_set.group_by_nonzero_dims()

# 打印结果
for nonzero_dims, point_group in groups.items():
    print(f"\n非零维度: {nonzero_dims}")
    for point in point_group.points:
        print(point.coord)