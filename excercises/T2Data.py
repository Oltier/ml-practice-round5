# import the needed libraries
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

## Choosing nice colors for plot
# if you want to plot for k>3, extend these lists of colors
cmpd = ['orangered', 'dodgerblue', 'springgreen']
cmpcent = ['red', 'darkblue', 'limegreen']

# read in data from the csv file
df = pd.read_csv("data.csv")
data = df.values

# display first 5 rows, to get a feeling for the data
display(df.head(5))


def plotting(data, centroids=None, clusters=None):
    # this function will later on be used for plotting the clusters and centroids. But now we use it to just make a scatter plot of the data
    # Input: the data as an array, cluster means (centroids), cluster assignemnts in {0,1,...,k-1}
    # Output: a scatter plot of the data in the clusters with cluster means
    plt.figure(figsize=(5.75, 5.25))
    plt.style.use('ggplot')
    plt.title("Data")
    plt.xlabel("feature $x_1$: customers' age")
    plt.ylabel("feature $x_2$: money spent during visit")

    alp = 0.5  # data alpha
    dt_sz = 20  # data point size
    cent_sz = 130  # centroid sz

    if centroids is None and clusters is None:
        plt.scatter(data[:, 0], data[:, 1], s=dt_sz, alpha=alp, c=cmpd[0])
    if centroids is not None and clusters is None:
        plt.scatter(data[:, 0], data[:, 1], s=dt_sz, alpha=alp, c=cmpd[0])
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=cent_sz, c=cmpcent)
    if centroids is not None and clusters is not None:
        plt.scatter(data[:, 0], data[:, 1], c=[cmpd[i] for i in clusters], s=dt_sz, alpha=alp)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", c=cmpcent, s=cent_sz)
    if centroids is None and clusters is not None:
        plt.scatter(data[:, 0], data[:, 1], c=[cmpd[i] for i in clusters], s=dt_sz, alpha=alp)

    plt.show()

    return centroids


# plot the data
plotting(data)

