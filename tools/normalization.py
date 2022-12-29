import numpy as np

# 实现对输入数据的归一化
def normalization(x):
    """
    :param x:输入的数据维度可能是[N,C,H,W]或[N,m]
    :return: 归一化后的结果
    """
    eps = 1e-5
    if x.ndim > 2:
        mean = np.mean(x, axis=(0, 2, 3))[:, np.newaxis, np.newaxis]
        var = np.var(x, axis=(0, 2, 3))[:, np.newaxis, np.newaxis]
        x = (x - mean) / np.sqrt(var + eps)
    else:
        mean = np.mean(x, axis=1)[:, np.newaxis]
        var = np.var(x, axis=1)[:, np.newaxis] + eps
        x = (x - mean) / np.sqrt(var)

    return x