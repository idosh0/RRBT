import numpy as np

def ConnectIn2D(x_a, x_b):
    path_len = np.linalg.norm(x_b - x_a)
    if path_len > 1:
        path_div = int(path_len)
    else:
        path_div = 5
    segment_length = path_len/path_div
    x_prev = x_a
    x = x_a
    X_nominal = []
    U_nominal = []
    K_nominal = []
    x_dim = x_a.shape[0]
    for i in range(path_div):
        X_nominal.append(x)
        if (x_b - x).all == 0:
            raise ZeroDivisionError
        u = segment_length * (x_b - x) / np.linalg.norm(x_b - x)
        x = x + u
        U_nominal.append(u)
        k = np.eye(x_dim) * 0.5
        K_nominal.append(k)
    return (X_nominal,U_nominal,K_nominal)

def Model2D(x, map):
    A = np.array([[1, 0], [0, 1]])
    B = np.array([[1, 0], [0, 1]])
    C = np.array([[1, 0], [0, 1]])
    Q = np.array([[1,0],[0,1]])*0.1
    if map[int(x[0]), int(x[1])] == 2:
        R = np.eye(2, 2) * 0.01
    else:
        R = np.ones((2, 2)) * np.inf
    return (A, B, C, Q, R)

def cost2D(x_a,x_b):
    return np.linalg.norm(x_a - x_b)
