import numpy as np
import struct

def read_fvecs(path):
    with open(path, 'rb') as f:
        data = f.read()
    offset = 0
    vectors = []
    while offset < len(data):
        dim = struct.unpack_from('i', data, offset)[0]
        offset += 4
        vec = struct.unpack_from(f'{dim}f', data, offset)
        vectors.append(vec)
        offset += 4 * dim
    vectors = np.array(vectors, dtype=np.float32)
    
    # 对每一维进行归一化
    min_vals = np.min(vectors, axis=0)
    max_vals = np.max(vectors, axis=0)
    
    # 处理可能的除零情况
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # 如果某维度最大值等于最小值，避免除零
    
    normalized_vectors = (vectors - min_vals) / range_vals
    
    # 确保所有值都在[0,1]范围内
    normalized_vectors = np.clip(normalized_vectors, 0, 1)
    
    return normalized_vectors

def fvecs_to_txt(input_path, output_path):
    vectors = read_fvecs(input_path)
    n, d = vectors.shape
    with open(output_path, 'w') as f:
        f.write(f"{n} {d}\n")
        for vec in vectors:
            f.write(" ".join(map(str, vec)) + "\n")

# 示例用法
#fvecs_to_txt("./input/audio.fvecs", "./input/audio.txt")
#fvecs_to_txt("./input/sift1M.fvecs", "./input/sift1M.txt")
#fvecs_to_txt("./input/movielens.fvecs", "./input/movielens.txt")
#fvecs_to_txt("./input/netflix.fvecs", "./input/netflix.txt")
#fvecs_to_txt("./input/deep1M.fvecs", "./input/deep1M.txt")
fvecs_to_txt("./input/yahoomusic.fvecs", "./input/yahoomusic.txt")