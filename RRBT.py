import networkx as nx
from Gen_functions import *
import numpy as np

class RRBT:
    def __init__(self, _map, x_init, sigma_0, x_dim=2):
        self.belief_counter = belief_id_gen()
        self.node_counter = node_id_gen()
        self.G = nx.DiGraph()
        self.map = _map
        self.N_map = self.map.shape[0]
        self.ObsPointList = [[i, j] for i in range(self.N_map) for j in range(self.N_map) if self.map[i, j] == 1]
        Map = np.copy(_map)
        np.place(Map, Map == 2, 0)
        self.ObsPointList = [(i, j) for j in range(self.N_map) for i in range(self.N_map)
                             if ((0 < j < self.N_map - 1 and (0 < i < self.N_map - 1) and Map[i, j] == 1 and Map[
                i + 1, j] + Map[i, j + 1] + Map[i - 1, j] + Map[i, j - 1] < 4) or
                                 (i == self.N_map - 1 and (Map[i, j] != Map[i - 1, j] or Map[i, j] == 1)) or
                                 (j == self.N_map - 1 and (Map[i, j] != Map[i, j - 1] or Map[i, j] == 1)) or
                                 (i == 0 and (Map[i, j] != Map[i + 1, j] or Map[i, j] == 1)) or
                                 (j == 0 and (Map[i, j] != Map[i, j + 1] or Map[i, j] == 1)))]

        self.x_dim = x_dim
        self.LinearModel = None
        self.ConnectModel = None
        self.F_function = None
        cur_node_counter = next(self.node_counter)
        n_init = belief(sigma_0, np.zeros((x_dim, x_dim)), 0)
        n_init.set_vertex(cur_node_counter)
        self.beliefs = {next(self.belief_counter): n_init}
        N_init = [1]
        self.G.add_node(cur_node_counter, N=N_init, X=x_init, prev=None, tag=0)
        self.radius_scale = 50
        self.Q = []
        self.biased_sample = False
        self.biased_sample_TH = None
        self.biased_sample_goal_X = None
        self.biased_sample_goal_r = None
        self.biased_sample_flag = False
        self.biased_sample_change_rate = None


    def set_biased_sample(self, Goal, TH = 0.2, rate = 0.7):
        self.biased_sample_TH = TH
        self.biased_sample_goal_X  = np.array(Goal[0])
        self.biased_sample_goal_r = Goal[1]
        self.biased_sample = True
        self.biased_sample_change_rate = rate

    def set_model(self, Linmodel, connectmodel):
        self.LinearModel = Linmodel
        self.ConnectModel = connectmodel

    def set_cost(self, f):
        self.F_function = f

    def Connect(self, x_a, x_b):
        (X_nominal, U_nominal, K_nominal) = self.ConnectModel(x_a, x_b)

        for x in X_nominal:
            if self._IsOcc(x):
                self.biased_sample_flag = False
                return None

        if self.biased_sample_flag:
            self.biased_sample_TH *= self.biased_sample_change_rate
            self.biased_sample_flag = False

        return X_nominal, U_nominal, K_nominal

    def Propagate(self, e, n_start):
        b = self.beliefs[n_start]
        cov_0 = b.cov
        Lamda = b.Lamda
        (X_nom, U_nom, K_nom) = e
        Sigma = cov_0
        Lamda = Lamda
        x_a = X_nom[0]
        x_next = None
        for (x, u, K) in zip(X_nom, U_nom, K_nom):
            if not self.CheckConstrain(x, Sigma, Lamda):
                return None
            (A, B, C, Q, R) = self.LinearModel(x, self.map)
            Sigma_bar = A @ Sigma @ np.transpose(A) + Q
            S = C @ Sigma_bar @ np.transpose(C) + R
            L = np.zeros((2, 2)) if np.any(S == np.inf) else Sigma_bar @ np.transpose(C) @ np.linalg.inv(S)
            Sigma = Sigma_bar - L @ C @ Sigma_bar
            AA = A - B @ K
            Lamda = AA @ Lamda @ np.transpose(AA) + L @ C @ Sigma_bar
            x_next = x + B @ u

        x_b = x_next
        if not self.CheckConstrain(x_b, Sigma, Lamda):
            return None
        cost = self.delta_J(x_a, x_b) + b.cost
        return belief(Sigma, Lamda, cost, n_start)

    def AppendBelief(self, v_idx, n_new):
        epsilon = 0.0001  # Need to be fine tuned
        # Soft Domination - with epsilon
        # Hard Domination - w/o  epsilon

        # Check if the belief is softly dominated by any existing belief
        epsilon_mat = epsilon * np.eye(self.x_dim)
        for b_idx in self.G.nodes[v_idx]["N"]:
            b = self.beliefs[b_idx]
            if np.linalg.det(b.cov) < np.linalg.det(n_new.cov + epsilon_mat) and \
                    np.linalg.det(b.Lamda) < np.linalg.det(n_new.Lamda + epsilon_mat) and \
                    b.cost < n_new.cost:
                return False, -1

        # Check if the belief dominates any existing belief and prune them
        for b_idx in self.G.nodes[v_idx]["N"]:
            b = self.beliefs[b_idx]
            if np.linalg.det(n_new.cov) <= np.linalg.det(b.cov) and \
                    np.linalg.det(n_new.Lamda) <= np.linalg.det(b.Lamda) and \
                    n_new.cost <= b.cost:
                # prune this belief from v_idx
                self.G.nodes[v_idx]["N"].remove(b_idx)
                del self.beliefs[b_idx]

        # Add belief to vertex
        n_new.set_vertex(v_idx)
        n_new_idx = next(self.belief_counter)
        self.beliefs[n_new_idx] = n_new
        self.G.nodes[v_idx]['N'].append(n_new_idx)
        return True, n_new_idx

    def Sample(self):
        # input: map
        # output: random free point in the map
        if self.biased_sample:
            if np.random.random_sample(1) < self.biased_sample_TH:
                while True:
                    theta = np.random.random_sample(1) * np.pi * 2
                    r = np.random.random_sample(1) * self.biased_sample_goal_r
                    X_rand = self.biased_sample_goal_X + np.array([np.cos(theta), np.sin(theta)]).T * r
                    X_rand = X_rand[0, :]
                    if not self._IsOcc(X_rand):
                        return X_rand

        while True:
            X_rand = np.random.random_sample(2) * self.map.shape[0]
            if not self._IsOcc(X_rand):
                return X_rand

    def Nearest(self, X):
        loc = [[loc_i[0], loc_i[1]] for n, loc_i in self.G.nodes.data('X')]
        loc = np.array(loc)
        loc_diff = np.subtract(loc, X)
        loc_dist = np.sqrt(loc_diff[:, 0] ** 2 + loc_diff[:, 1] ** 2)
        return np.argmin(loc_dist)

    def Near(self, X):
        loc = [[loc_i[0], loc_i[1]] for n, loc_i in self.G.nodes.data('X')]
        loc = np.array(loc)
        loc_diff = np.subtract(loc, X)
        loc_dist = np.sqrt(loc_diff[:, 0] ** 2 + loc_diff[:, 1] ** 2)
        nn = self.G.number_of_nodes()
        rho = (np.log(nn) / nn) ** (1 / self.x_dim) * self.radius_scale
        return np.nonzero(loc_dist < rho)[0]

    def _IsOcc(self, X):
        if self.map[int(X[0]), int(X[1])] == 1:
            return True
        else:
            return False

    def add_all_belief(self, v_id):
        return [(v_id, n.id) for n in self.G.nodes[v_id]['N']]

    def delta_J(self, x_a, x_b):
        return self.F_function(x_a, x_b)

    def CheckConstrain(self, X, Sigma, Lamda):
        delta = 0.3
        if self._IsOcc(X):  # Cell is Occ with Obs
            return False
        if self.IsThereAnObsInCovCircle(X, Sigma, Lamda):
            return False
        return True

    def IsThereAnObsInCovCircle(self, X, Sigma, Lamda):
        radius_mat = Sigma + Lamda
        radius = np.diag(radius_mat).max()
        obs_loc_diff = np.subtract(self.ObsPointList, X)
        loc_dist = obs_loc_diff[:, 0] ** 2 + obs_loc_diff[:, 1] ** 2
        if np.any(loc_dist < radius):
            return True
        return False

    def AddVertex(self):
        x_rand = self.Sample()

        v_nearest_idx = self.Nearest(x_rand)
        e_nearest_path = self.Connect(self.G.nodes[v_nearest_idx]['X'], x_rand)
        if e_nearest_path is None:
            return False
        N_iter = self.G.nodes[v_nearest_idx]['N']
        possible_edge = False
        for n in N_iter:
            if self.Propagate(e_nearest_path, n):
                possible_edge = True
                break
        if not possible_edge:
            return False

        new_vertex_id = next(self.node_counter)
        V_near = self.Near(x_rand)
        self.G.add_node(new_vertex_id, N=[], X=x_rand, prev=None, tag=0)

        if V_near.size == 0:
            e_nearest_path_rev = self.Connect(x_rand, self.G.nodes[v_nearest_idx]['X'])
            self.G.add_edge(v_nearest_idx, new_vertex_id, path=e_nearest_path)
            self.G.add_edge(new_vertex_id, v_nearest_idx, path=e_nearest_path_rev)
            self.Q = list(set().union(self.Q, self.G.nodes[v_nearest_idx]['N']))
        else:
            for v_near_idx in V_near:
                e_near_path = self.Connect(self.G.nodes[v_near_idx]['X'], x_rand)
                e_near_path_rev = self.Connect(x_rand, self.G.nodes[v_near_idx]['X'])
                if e_near_path is None or e_near_path_rev is None:
                    return False
                self.G.add_edge(v_near_idx, new_vertex_id, path=e_near_path)
                self.G.add_edge(new_vertex_id, v_near_idx, path=e_near_path_rev)
                #            Q.extend(self.G.nodes[v_near_idx]['N'])
                self.Q = list(set().union(self.Q, self.G.nodes[v_near_idx]['N']))
        while self.Q:
            n_q = self.Q.pop(0)
            if n_q not in self.beliefs:
                continue
            for v_neighbor_idx in self.G.neighbors(self.beliefs[n_q].vertex):
                e_neighbor_path = self.G.edges[(v_neighbor_idx, self.beliefs[n_q].vertex)]['path']
                n_new = self.Propagate(e_neighbor_path, n_q)
                if not n_new:
                    continue
                Append_flag, n_new_idx = self.AppendBelief(v_neighbor_idx, n_new)
                if Append_flag:
                    self.Q = list(set().union(self.Q, [n_new_idx]))
        return True
