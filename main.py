from RRBT import RRBT
from Gen_functions import *
import numpy as np
import model2d as m2D
from Plotter import Plotter


if __name__ == "__main__":

    # running parameters
    M = 1500
    x_dim = 2
    sigma_0 = np.eye(x_dim) * 10
    x_init = np.array([5, 45])

    # build the map
    N_map = 100
    N_obs = 15
    obs_size = (15, 10)
    #C_obs = GenerateObs(N_map, N_obs, obs_size)
    C_obs = [[(0, 50), (45, 10)], [(55, 50), (45, 10)]]
    N_meas = 10
    #C_meas = GenerateObs(N_map, N_meas, obs_size)
    C_meas = [[(0, 0), (100, 10)]]
    C_goal = [(85, 85), 5]
    Map = GenerateMap(C_obs, C_meas, N_map)

    # init the RRBT class
    rrbt = RRBT(Map, x_init, sigma_0)
    f_connect = lambda x_a, x_b: m2D.ConnectIn2D(x_a, x_b)
    f_model = lambda x, map: m2D.Model2D(x, map)
    f_cost = lambda x_a, x_b: m2D.cost2D(x_a, x_b)
    rrbt.set_model(f_model, f_connect)
    rrbt.set_cost(f_cost)

    # init Ploter class
    myplot = Plotter()
    myplot.enable_video()

    i = 0
    while i < M:
        if not rrbt.AddVertex():
            print("Failed to add vertex")
            continue
        i = i + 1
        (path, vertex_in_goal, best_belief, best_cost) = FindBestPath(rrbt, C_goal)
        myplot.Dostep(C_obs, C_meas, C_goal, rrbt, path, N_map)
        print(i)


myplot.generate_video()
(best_path, vertex_in_goal, best_belief, best_cost) = FindBestPath(rrbt, C_goal)
myplot.add_obstacles(C_obs)
myplot.add_MeasArea(C_meas)
myplot.add_GoalArea(C_goal)
myplot.add_node_edges(rrbt.G)
myplot.show_sol(rrbt.G, path)
myplot.show_graph(N_map)


print("Finished")