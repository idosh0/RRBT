# -*- coding: utf-8 -*-


from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import numpy as np
import networkx as nx
# mpl.use('TkAgg')
plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\idosh\\Downloads\\ffmpeg-5.0.1-essentials_build\\' \
                                        'ffmpeg-5.0.1-essentials_build\\bin\\ffmpeg.exe'

class Plotter:
    def __init__(self, name='', fig_num=1):
        self.fig = plt.figure(fig_num)
        self.ax = self.fig.subplots()
        self.write_video = False
        self.frames = None
        self.counter = 0
        self.run_name = name
        self.fig_num = fig_num

    def enable_video(self):
        self.write_video = True
        self.frames = []

    def add_obstacles(self, obstacles):

        for obstacle in obstacles:
            x = obstacle[0][0]
            y = obstacle[0][1]
            dx = obstacle[1][0]
            dy = obstacle[1][1]
            obs = np.array([[x     , y     ],\
                            [x + dx, y     ],\
                            [x + dx, y + dy],\
                            [x     , y + dy]])
            self.ax.add_patch(plt.Polygon(obs, color='black'))

    def add_MeasArea(self, MeasArea):

        for meas in MeasArea:
            x = meas[0][0]
            y = meas[0][1]
            dx = meas[1][0]
            dy = meas[1][1]
            meas_points = np.array([[x, y],
                            [x + dx, y],
                            [x + dx, y + dy],
                            [x, y + dy]])
            self.ax.add_patch(plt.Polygon(meas_points, color='red', alpha = 0.5))

    def add_GoalArea(self, GoalArea):
        X_goal = GoalArea[0]
        radius = GoalArea[1]
        self.ax.add_patch(plt.Circle(X_goal, radius, color='green', alpha = 0.5))


    def add_node_edges(self, G: nx.DiGraph):

        points = [[loc_i[0], loc_i[1]] for n, loc_i in G.nodes.data('X')]
        edges = [[points[u][0], points[v][0], points[u][1], points[v][1]]for u, v in G.edges] # , G[u][v]['tag']
        for i,edge in enumerate(edges):
            # print('plot',i+1,'/',len(prm))
            self.ax.add_line(plt.Line2D(edge[0:2],edge[2:4],1,alpha=0.3))
        
        # plot the nodes after the edges so they appear above them
        plt.scatter(np.array(points)[:,0], np.array(points)[:,1], 4, color='red')
        # plt.scatter(-5,99,4,color='green')
        # for i, p in enumerate(np.array(points)):
        #     self.ax.annotate(i, (p[0], p[1]))

    def add_sol(self, sol):
        for i,node in enumerate(sol):
            if node != sol[-1]:
                next_node = sol[i+1]
                for j,edge in enumerate(node.edges):
                    if next_node == edge[0]: 
                        line = node.edges_lines[j]
                        self.ax.add_line(plt.Line2D(list(line.coords.xy[0]),list(line.coords.xy[1]),
                                                    1,color='red'))
    def show_sol(self, G: nx.DiGraph, path):
        if not path:
            return
        points = [G.nodes[n]['X'] for n in path]
        points = np.array(points)
        edges = [[points[i][0], points[i+1][0], points[i][1], points[i+1][1]] for i in range(points.shape[0] - 1)]  # , G[u][v]['tag']
        for i, edge in enumerate(edges):
            # print('plot',i+1,'/',len(prm))
            self.ax.add_line(plt.Line2D(edge[0:2], edge[2:4], 3, alpha=0.3, color='red'))

        # plot the nodes after the edges so they appear above them
        plt.scatter(points[:, 0], points[:, 1], 6, color='hotpink' )
        # plt.scatter(-5,99,4,color='green')
        # for i, p in enumerate(np.array(points)):
        #     self.ax.annotate(i, (p[0], p[1]))

    def show_graph(self, n_map): #,N_nodes, thd, edges, avg_node_degree):
        self.ax.autoscale()
        self.ax.relim([0, n_map])
        self.ax.relim([0, n_map])
        self.fig.show(self.fig_num)


    def save_courrent(self,n_map):
        if not self.write_video:
            return
        plt.figure(self.fig_num)
        self.ax.autoscale()
        self.ax.relim([0, n_map])
        self.ax.relim([0, n_map])
        plt.axis('off')
        plt.savefig(self.run_name + '_frames/op' + str(self.counter) + '.png')
        self.counter += 1
        self.ax.clear()

    def generate_video(self):
        if not self.write_video:
            return
        opfig = plt.figure()
        for i in range(self.counter):
            im = plt.imread(self.run_name + '_frames/op' + str(i) + '.png')
            a = plt.imshow(im, animated=True)
            plt.axis('off')
            self.frames.append([a])
        ani = animation.ArtistAnimation(opfig, self.frames, interval=50, blit=True,
                                        repeat_delay=1000)
        videowriter = animation.FFMpegWriter(fps=5)
        ani.save('C:\\Users\\idosh\\Documents\\ANP_Nav_Data\\Project\\' + self.run_name + 'movie.mp4', writer=videowriter)

    def Dostep(self, C_obs, C_meas, C_goal, rrbt, path, N_map):
        plt.figure(self.fig_num)
        self.add_obstacles(C_obs)
        self.add_MeasArea(C_meas)
        self.add_GoalArea(C_goal)
        self.add_node_edges(rrbt.G)
        self.show_sol(rrbt.G, path)
        self.save_courrent(N_map)

