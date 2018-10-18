import pandas as pd


def k_means(data, k, random_seed=1, num_iters=10,plot=True):
    #INPUT: N x d data array, k number of clusters, number of iterations, boolean plot.
    #OUTPUT: N x 1 array of cluster assignments.

    ### STUDENT TASK ###
    #step 1
    #centroids = ...
    # YOUR CODE HERE
    raise NotImplementedError()

    #loop for steps 2 and 3
    for i in range(num_iters):
        ### STUDENT TASK ###
        #step 2
        #clusters = ...
        # YOUR CODE HERE
        raise NotImplementedError()

        #plotting
        if plot==True and i<3:
            plotting(data,centroids,clusters)

        ### STUDENT TASK ###
        #step 3
        #centroids = ...
        # YOUR CODE HERE
        raise NotImplementedError()

    return centroids,clusters


df = pd.read_csv("data.csv")
data = df.as_matrix()

# Step 3.4
centroids, clusters = k_means(data, 2)
print("The final cluster mean values are:",centroids)