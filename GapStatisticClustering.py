#Gap statistic

import numpy as np
import pandas as pd
import scipy as sp
import scipy.cluster.vq as vq
import matplotlib.pyplot as plt
import os

def clean(features):
    """
    Sanitize the dataset for analysis by centering the means to 0 and creating unit variance
    """
    means = features.mean(axis=0)
    sd = features.std(axis=0)
    cleaned = features
    for j in range(features.shape[1]):
        cleaned.T[j] = [(i-means[j])/sd[j] for i in cleaned.T[j]]
    return (cleaned,means,sd)
    
def random_points(x):
    """
    Generate points within the minimum bounding box defined by pca
    The matrix must alraedy be mean centered and variance 1
    """
    N,M = x.shape
    cv = np.cov(x.T)
    u,s,v = np.linalg.svd(cv)
    ranges = np.dot(x,v)
    mins = ranges.min(axis=0)
    maxs = ranges.max(axis=0)
    newdata = np.zeros((N,M))
    for j in range(M):
        runif = np.random.uniform(size=N)
        newdata.T[j] = runif*maxs[j] + (1.-runif)*mins[j]
    newdata = np.dot(newdata,v.T)
    return newdata

def euclid(x,y):
    """
    Shortcut to euclidian distance
    """
    return sp.spatial.distance.euclidean(x,y)

def within_cluster_sum_squares(data,centroid,label,distance=euclid):
    """
    Calculate the within cluster sum of squares
    """
    K, M = centroid.shape
    N = data.shape[0]
    dr = np.zeros(K)
    nr = np.zeros(K)
    wk = 0
    for k in range(K):
        which = [i for i in range(N) if label[i]==k]
        if len(which) != 0:
            nr[k] = len(which)
            for i in which:
                for j in which:
                    dr[k] += distance(data[i],data[j])
            wk += (1./(2*nr[k]))*dr[k]
    return wk, nr, dr

def mean_log_wcss(data,ranges,samples=100,distance=euclid):
    """
    Calculate the expected value of the log of the within cluster sum of squares (wcss) from randomized data
    """
    wcss = np.zeros(len(ranges)) #Output: the mean of the log of the within cluster sum of squares
    wcss2 = np.zeros(len(ranges))
    for r in range(samples):
        newdata = random_points(data)
        newdata, newmeans, newstd= clean(newdata)
        for k in ranges:
            centroid, label = vq.kmeans2(newdata,k,minit='points')
            wk, nr, dr = within_cluster_sum_squares(newdata,centroid,label,distance)
            wcss[[i for i in range(len(ranges)) if ranges[i]==k]] += np.log(wk)
            wcss2[[i for i in range(len(ranges)) if ranges[i]==k]] += np.log(wk)**2
    wcss = [1.*x/samples for x in wcss]
    wcss2 = [1.*x/samples for x in wcss2]
    wcssvar = [wcss2[i]-wcss[i]**2 for i in range(len(wcss))]
    error = np.sqrt(1+ 1./samples)*np.sqrt(wcssvar) #MAKE SURE THIS WORKS
    return wcss, error
            
def gap_statistic(data,ranges,samples=100,distance=euclid):
    """
    The gap statistic itself
    data - the data that has already been processed and recentered
    ranges - number of clusters to check over
    samples - number of artificial datasets to generate. Thousands or more appear best.
    distance - the distance function to use. Euclidean by default.
    """
    wcss, error = mean_log_wcss(data,ranges,samples,distance)
    actual = np.zeros(len(ranges))
    for k in ranges:
        w = [i for i in range(len(ranges)) if ranges[i] == k]
        centroid, label = vq.kmeans2(data,k,minit='points')
        wk, nr, dr = within_cluster_sum_squares(data,centroid,label,distance)
        actual[w] = np.log(wk)
    return (actual, wcss, error)      

def plot_gap_statistic(data,ranges,actual,wcss,error):
    """
    Reproduce the plot from Tibshirani et al. 2001.
    data - the raw data
    ranges - the ranges of clusters examined
    actual, wcss, error - output from gap_statistic in order
    """
    #Define the figure
    plt.figure(1)
    #Subplot one is a plot of the first 2 dimensions of the data
    plt.subplot(2,2,1)
    plt.plot(data.T[0],data.T[1],'bo')
    plt.title('Data')
    plt.xlabel('X1')
    plt.ylabel('X2')
    
    #Subplot 2 is the elbow graph
    plt.subplot(2,2,2)
    plt.plot(ranges,np.exp(actual))
    plt.xlabel('Number of clusters')
    plt.ylabel('Within cluster sum of squares ($W_k$)')
    plt.title('"Elbow" graph')
    
    #Subplot 3 is the within cluster sum of squares
    plt.subplot(2,2,3)
    wcline, = plt.plot(ranges,wcss,'b-')
    aline, = plt.plot(ranges,actual,'r-')
    plt.title('Expected vs. observed $log(W_k)$ statistic')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within cluster sum of squares ($log(W_k)$)')
    plt.legend((wcline,aline),('Expected $log(W_k)$ from uniform','Observed $log(W_k)$ from data'),loc=0)
    
    #Subplot 4 is the actual gap statistic with error
    plt.subplot(2,2,4)
    plt.errorbar(ranges,wcss-actual,yerr = error)
    plt.xlabel('Number of clusters')
    plt.ylabel('Gap statistic with error')    
    plt.title('Gap statistic')
    
    plt.show()

if __name__ == '__main__':
    ##Generate some artificial data with two cluisters
    features  = np.random.multivariate_normal([-3,-3],[[2,0],[0,2]],size=20)
    features = np.concatenate((features,np.random.multivariate_normal([6,4],[[2,0],[0,2]],size=30)))
    # plot to examine
    plt.plot(features.T[0],features.T[1],'bo')
    
    # STEP 1: clean the data
    cleaned, means, sd = clean(features)
    #centroid, label = vq.kmeans2(cleaned,2,minit='points')
    #within_cluster_sum_squares(cleaned,centroid,label)
    
    ranges = range(1,20)
    actual, wcss, error = gap_statistic(cleaned,ranges,500,euclid) #you need very large numbers, likely in the thousands
    
    export = pd.DataFrame({'k' : ranges, 'actual' : actual, 'wcss' : wcss, 'error' : error})
    os.chdir('c:\cases\southitaly\\analysis\\clusters')
    export.to_csv('SurveyCookingWaresKMeans.csv')
    #plt.plot(wcss,'b-')
    #plt.plot(actual,'r-')
    #plt.plot(wcss-actual,'b-')
    
    plt.figure(1)
    plt.subplot(2,2,1)
    plt.plot(features.T[0],features.T[1],'bo')
    plt.title('Data')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.subplot(2,2,2)
    plt.plot(ranges,np.exp(actual))
    plt.xlabel('Number of clusters')
    plt.ylabel('Within cluster sum of squares ($W_k$)')
    plt.title('"Elbow" graph')
    plt.subplot(2,2,3)
    wcline, = plt.plot(ranges,wcss,'b-')
    aline, = plt.plot(ranges,actual,'r-')
    plt.title('Expected vs. observed $log(W_k)$ statistic')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within cluster sum of squares ($log(W_k)$)')
    plt.legend((wcline,aline),('Expected $log(W_k)$ from uniform','Observed $log(W_k)$ from data'),loc=0)
    plt.subplot(2,2,4)
    plt.errorbar(ranges,wcss-actual,yerr = error)
    plt.xlabel('Number of clusters')
    plt.ylabel('Gap statistic with error')    
    plt.title('Gap statistic')
    
    plt.plot(centroid.T[0],centroid.T[1],'g^')
    
    plt.plot(cleaned.T[0],cleaned.T[1],'bo')
    plt.plot(newdata.T[0],newdata.T[1],'ro')
