import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import community as community_louvain
import networkx as nx
import pandas as pd


class EvoLouvain():

    def __init__(self):
        self.modularities_ = [] # Modularities per timesteps

    def showPlot(self, current_gen, time, partition, modularity, alpha,
                 save_plot = None):
        # Plot partitions
        plt.scatter(current_gen[:,0], current_gen[:,1],
                    c = list(partition.values()), cmap = 'tab20')

        # Create graph title with params
        title = 'Louvain Generation-{g} Alpha-{a} Modularity-{m}'.format(g = time,
                                                                         a = alpha,
                                                                         m = "%.5f" % modularity)
        plt.title(title)
        if not save_plot == None:
            # Save fig as PNG
            plt.savefig('{p}{t}.png'.format(p=save_plot, t = title))
        plt.show() # Show fig


    def applySmoothing(self, mat1, mat2, alpha):
        mat1, mat2 = (1-alpha)*mat1, alpha*mat2
        ysize, xsize = mat2.shape # Get shape of second matrix-1
        mat1[0:ysize, 0:xsize] += mat2 # Add elements of matrices on index (0,0)
        return mat1

    def sparsify(self, dist):
        dist[(dist < 10) & (dist != 0)] = 3
        dist[(dist > 10) & (dist < 20)] = 2
        dist[(dist > 20) & (dist < 30)] = 1
        dist[dist > 30] = 0
        return dist

    def callLouvain(self, X, times, alpha, show_mod = False, plot_gens = None,
                    save_plot = None):


        previous_gen = None
        seen_times = []

        t_intervals = np.unique(times)

        if plot_gens == None:
            # Default generations to plot
            plot_gens = [int(np.median(t_intervals)/2), # Quarter 1
                         int(np.median(t_intervals)), # Middle
                         t_intervals[-1]] # End

        for time in t_intervals:
            seen_times.append(time)

            current_gen = X.loc[X['Time'].isin(seen_times)]
            current_gen = current_gen.iloc[:,[0,1]].values

            # Calculate the distance matrix of points
            dist_cs = pairwise_distances(current_gen)
            # Remove distant links and scale
            dist_cs = self.sparsify(dist_cs)

            if not time == 0:
                # Time must be > 1 to consider history
                dist_cs = self.applySmoothing(dist_cs, previous_gen, alpha)

            # Get NetworkX graph
            G = nx.from_numpy_matrix(dist_cs)
            # Find the best partition at time step
            partition = community_louvain.best_partition(G)

            # Find the modularity
            modularity = 0
            try:
                modularity = community_louvain.modularity(partition, G)
            except(ValueError) as e:
                # If G has incomplete linkage (due to sparsification)
                # it will throw a ValueError
                print(e)
                print("Can't compute modularity for this partition.")

            if (show_mod):
                print('Generation {g} modularity: {m}'
                      .format(g = self.current_time, m = modularity))

            # Save modularities for visualization
            self.modularities_.append(modularity)
            seen_times.append(time)

            # Save previous generation as histroy
            previous_gen = dist_cs

            # Plotting
            if time in plot_gens:
               self.showPlot(current_gen, time, partition, modularity, alpha,
                             save_plot=save_plot)