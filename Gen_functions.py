import numpy as np
import model2d as m2D
import random


def GenerateObs(N_map, N_obs, obs_size):
    C_obs = []
    for i in range(N_obs):
        obs_x = random.randint(0, N_map - 1)
        obs_y = random.randint(0, N_map - 1)
        C_obs.append([(obs_x, obs_y), obs_size])
    return C_obs


def GenerateMap(C_obs, C_meas, N_map):  # botnik
    # input: N_map,C_obs
    # output: N_map*N_map grid with 1=occ and 0=free assigned in self.map
    Map = np.zeros((N_map, N_map))
    for obs in C_obs:
        # C_obs(0) := location
        # C_obs(1) := size
        for i in range(obs[1][0]):
            for j in range(obs[1][1]):
                if obs[0][0] + i < N_map and obs[0][1] + j < N_map:
                    Map[obs[0][0] + i][obs[0][1] + j] = 1
    for meas in C_meas:
        # C_obs(0) := location
        # C_obs(1) := size
        for i in range(meas[1][0]):
            for j in range(meas[1][1]):
                if meas[0][0] + i < N_map and meas[0][1] + j < N_map:
                    if Map[meas[0][0] + i][meas[0][1] + j] != 1:
                        Map[meas[0][0] + i][meas[0][1] + j] = 2
    return Map


def belief_id_gen():
    i = 1
    yield i
    while True:
        i += 1
        yield i


def node_id_gen():
    i = 0
    yield i
    while True:
        i += 1
        yield i


class belief:
    def __init__(self, cov, lamda, cost, parent=None):
        self.vertex = -1
        self.cov = cov
        self.Lamda = lamda
        self.cost = cost
        self.parent = parent

    def set_vertex(self, v):
        self.vertex = v

def test_connect_2D():
    x_a = np.array([1.0, 1.0])
    x_b = np.array([10.0, 3.0])
    (X, U, K) = m2D.ConnectIn2D(x_a, x_b)
    print("pause")


def FindBestPath(rrbt, C_goal):
    X_goal = np.array(C_goal[0])
    r_goal = C_goal[1]
    vertex_in_goal = [n for n, loc_i in rrbt.G.nodes.data('X') if np.linalg.norm(np.array(loc_i) - X_goal) < r_goal]
    best_belief = -1
    best_cost = np.inf
    if not vertex_in_goal:
        return None, None, None, None
    for vertex in vertex_in_goal:
        for n in rrbt.G.nodes[vertex]['N']:
            if rrbt.beliefs[n].cost < best_cost:
                best_belief = n
                best_cost = rrbt.beliefs[n].cost
    n_cur = rrbt.beliefs[best_belief]
    path = [n_cur.vertex]
    while n_cur.parent is not None:
        n_cur = rrbt.beliefs[n_cur.parent]
        path.append(n_cur.vertex)

    return path, vertex_in_goal, best_belief, best_cost