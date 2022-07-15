from RRBT import RRBT
from Gen_functions import *
import numpy as np
import model2d as m2D
from Plotter import Plotter


if __name__ == "__main__":

    # running parameters
    M = 150
    x_dim = 2
    sigma_0 = np.eye(x_dim) * 15
    x_init = np.array([5, 45])

    # build the map
    N_map = 100
    N_obs = 15
    obs_size = (15, 10)
    # C_obs = GenerateObs(N_map, N_obs, obs_size)
    C_obs = [[(0, 50), (43, 10)], [(55, 50), (45, 10)]]
    N_meas = 10
    # C_meas = GenerateObs(N_map, N_meas, obs_size)
    C_meas = [[(0, 0), (100, 10)]]
    C_goal = [(85, 85), 5]
    Map = GenerateMap(C_obs, C_meas, N_map)


    # model
    f_connect = lambda x_a, x_b: m2D.ConnectIn2D(x_a, x_b)
    f_model = lambda x, map: m2D.Model2D(x, map)
    f_cost = lambda x_a, x_b: m2D.cost2D(x_a, x_b)


    # init the RRBT class without biased sample
    rrbt_wo_BS = RRBT(Map, x_init, sigma_0)
    rrbt_wo_BS.set_model(f_model, f_connect)
    rrbt_wo_BS.set_cost(f_cost)

    # init Ploter class
    myplot_wo_BS = Plotter('without_biased_sampling',1)
    myplot_wo_BS.enable_video()

    # init the RRBT class with biased sample
    rrbt_BS = RRBT(Map, x_init, sigma_0)
    rrbt_BS.set_model(f_model, f_connect)
    rrbt_BS.set_cost(f_cost)
    rrbt_BS.set_biased_sample(C_goal, 0.1, 0.2)

    # init Ploter class
    myplot_BS = Plotter('with_biased_sampling', 2)
    myplot_BS.enable_video()


    i = 0
    j = 0
    while i < M or j < M:
        if i < M:
            if not rrbt_wo_BS.AddVertex():
                print("without biased: Failed to add vertex")
            else:
                i = i + 1
                (path, vertex_in_goal, best_belief, best_cost) = FindBestPath(rrbt_wo_BS, C_goal)
                myplot_wo_BS.Dostep(C_obs, C_meas, C_goal, rrbt_wo_BS, path, N_map)
                print('without biased:' + str(i))
        if j < M:
            if not rrbt_BS.AddVertex():
                print("with biased: Failed to add vertex")
            else:
                j = j + 1
                (path, vertex_in_goal, best_belief, best_cost) = FindBestPath(rrbt_BS, C_goal)
                myplot_BS.Dostep(C_obs, C_meas, C_goal, rrbt_BS, path, N_map)
                print('with biased:' + str(j))

myplot_wo_BS.generate_video()
(best_path, vertex_in_goal, best_belief, best_cost) = FindBestPath(rrbt_wo_BS, C_goal)
myplot_wo_BS.add_obstacles(C_obs)
myplot_wo_BS.add_MeasArea(C_meas)
myplot_wo_BS.add_GoalArea(C_goal)
myplot_wo_BS.add_node_edges(rrbt_wo_BS.G)
myplot_wo_BS.show_sol(rrbt_wo_BS.G, best_path)
myplot_wo_BS.show_graph(N_map)

myplot_BS.generate_video()
(best_path, vertex_in_goal, best_belief, best_cost) = FindBestPath(rrbt_BS, C_goal)
myplot_BS.add_obstacles(C_obs)
myplot_BS.add_MeasArea(C_meas)
myplot_BS.add_GoalArea(C_goal)
myplot_BS.add_node_edges(rrbt_BS.G)
myplot_BS.show_sol(rrbt_BS.G, best_path)
myplot_BS.show_graph(N_map)

print("Finished")
