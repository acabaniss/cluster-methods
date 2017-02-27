# cluster-methods
Statistical methods for dealing with clusters, primarily for applications in archaeology such as ceramic assemblages and survey data. This is neither intended to be complete nor comprehensive, but merely a) code I am using that b) I feel is at a stage where it is usable for others.

## The Gap Statistic.
The Gap Statistic is defined and published by Tibshirani et al. 2001. This implementation in in python and depends on numpy and scipy (particularly scipy.cluster.vq) for the calculations; the plot function is basic and only attempts to recreate the graph in Tibshirani et al., and uses matplotlib.pyplot.

### Citation:
Tibshirani, R.; Walther, G. & Hastie, T. Estimating the number of clusters in a data set via the gap statistic. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 2001, 63, 411-423
