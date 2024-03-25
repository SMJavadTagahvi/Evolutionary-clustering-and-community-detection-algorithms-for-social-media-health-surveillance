#From the paper:
#Evolutionary Clustering and 
#Community Detection Algorithms for Social Media Health Surveillance

#Kyle Spurlock, Tanner Bogart, Heba Elgazzar
#2020

# Version 2 of an Evolutionary Aadaptation of DBSCAN clustering algorithm.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances

class EvoDBSCAN(DBSCAN):
    """
        Implementation of Evolutionary DBSCAN with dynamic radius measure.
    
        Notes:
        ---------------------
            Acts as a wrapper for sci-kit learn DBSCAN class.
  
        Vars:
        ---------------------
            -eps: Radius measure for finding neighbourhood of core point
            -min_samples: Minimum number of neighbours to form a core point
            -clusters_gen: [] stores cluster count per generation
            -noise_gen: [] stores noise count per time generation
            -eps_gen: [] stores eps parameter per time generation
            
        ---------------------
        Sci-kit Learn Specific Vars:
            -metric: Distance metric (manhattan, euclidean, etc.)
            -metric_params: Sci-kit learn metric parameters
            -algorithm: Algorithm used to compute pointwise distances
            -leaf_size: Specific to BallTree or cKDTree algorithm
            -p: Power of Minkowski metric
            -n_jobs: Number of parallel jobs
    """     
    
    def __init__(self, min_samples=5,*,
                 eps=0,
                 metric='euclidean',
                 algorithm='auto',
                 leaf_size=30,
                 p=None,
                 n_jobs=None):
        
        super().__init__(min_samples=min_samples,
                         eps=eps,
                         metric=metric,
                         algorithm=algorithm,
                         leaf_size=leaf_size,
                         p = p,
                         n_jobs=n_jobs)
        
        self.clusters_gen_ = []
        self.noise_gen_ = []
        self.eps_gen_ = []
        

    def showPlot(self, current_gen, time, labels, noise, alpha,
                 save_plot=None):
        """Performs plotting of clusters at generation"""
        # Plot clusters
        plt.scatter(current_gen[:, 0], current_gen[:, 1],
                    c = labels, cmap = 'tab20')
        # Plot noise
        plt.scatter(current_gen[labels == -1, 0], current_gen[labels == -1, 1],
                    c = 'black') 
        
        # Create graph title with params
        title = 'DBSCAN Generation-{t} Alpha-{a} Noise-{n}'.format(t= time,
                                                                   a= alpha,
                                                                   n= noise)
        plt.title(title)
        if not save_plot == None:
            # Save fig as PNG
            plt.savefig('{p}{t}.png'.format(p=save_plot, t = title))
        plt.show() # Show fig
        
    
    def callDBSCAN(self, X, times, alpha, beta=1, show_eps=False,
                   plot_gens=None, save_plot=None): 
        """
            Perform evolutionary DBSCAN clustering.
        
            Args:
            ---------------------
                -X: Dataframe of tabular data samples
                -times: List containing times for X samples
                -alpha: Parameter used to modulate epsilon by snapshot vs. history
                -beta: Optional scaling param for alpha
                -show_eps: Verbose for comptued epsilon value
                -plot: Generations to show as plots
                
            Returns:
            ---------------------
                -None
        """        
        
        previous_gen = None # Holds the previous generations distance matrix
        seen_times = [] # Accumulates the seen time steps
        
        t_intervals = np.unique(times)
        
        if plot_gens == None:
            # Default generations to plot
            plot_gens = [int(np.median(t_intervals)/2), # Quarter 1
                         int(np.median(t_intervals)), # Middle
                         t_intervals[-1]] # End
        
        # Loop through each time step
        for time in t_intervals:
            seen_times.append(time) 

            current_gen = X.loc[X['Time'].isin(seen_times)] 
            current_gen = current_gen.iloc[:,[0,1]].values
            
            current_snap = X.loc[X['Time'] == time]
            
            # Compute pairwise distances
            snap_distances = pairwise_distances(current_snap)
            
            # Calculating episolon based on current snapshot or history
            if time == 0:
                # No history cost to consider
                self.eps = (np.median(np.unique(snap_distances))/beta)
            else:
                # Determine radius based on history and current snapshot
                self.eps = ((((1-alpha)*
                              np.median(np.unique(snap_distances)))/beta)
                            + (alpha*
                               np.median(np.unique(previous_gen)))/beta)
            
            # If there is currently no valid distances, make radius very small
            if self.eps == 0: 
                self.eps = 1e-5
            
            if show_eps:
                print('Generation {g} epsilon: {e}'
                      .format(g = self.current_time, e = self.eps))
            
            # Determine cluster labels and noise
            labels = self.fit_predict(current_gen) 
            noise = list(labels).count(-1)
            
            # Save noise and cluster count for each generation
            self.clusters_gen_.append(len(set(labels))
                                      - (1 if -1 in labels else 0))
            self.noise_gen_.append(noise)
            
            # Saving previous generation as the history
            previous_gen = pairwise_distances(current_gen) 
            
            # Plotting 
            if time in plot_gens: 
                self.showPlot(current_gen, time, labels, noise, alpha,
                              save_plot=save_plot)
                
        return None

    def callSTATIC(self, X, beta, save_plot = None):
        """"Normal DBSCAN implementation"""
        X = X.iloc[:,[0,1]].values
        
        # Compute pairwise distances
        dist_cg = pairwise_distances(X)
        
        # Compute radius measure
        self.eps = np.median(np.unique(dist_cg))/beta
    
        # Determine cluster labels and noise
        labels = self.fit_predict(X)
        noise = list(labels).count(-1)
        
        #Plotting
        self.showPlot(X, time=None, labels=labels, noise=noise, alpha=None,
                      save_plot=save_plot)