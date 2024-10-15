'''
An example of PCA

PCA is a techniwuq used to analyse and simplify data. It is a statistical method that is used to reduce the 
dimensionality of the data set. It does this by transforming the data into a new coordinate system in which the 
data points are uncorrelated. This new coordinate system is called the principal component space, and the axes are 
called the principal components.

A system of neurons could be considered as a data set where each neuron's riring rate is a feature. PCA could 
be used to reduce the dimensionality of the data set by finding the principal components of the data set. This
would allow us to simplify the data set and make it easier to analyse.
'''

import numpy as np

def PCA(data):
    '''
    Perform PCA on the data set
    
    Parameters:
    data: numpy array
        The data set to perform PCA on
        
    Returns:
    numpy array
        The data set in the new coordinate system
    '''
    
    # Calculate the mean of the data set
    mean = np.mean(data, axis=0)
    
    # Subtract the mean from the data set
    data = data - mean
    
    # Calculate the covariance matrix of the data set
    # This is a measure of how much the variables in the data set change together
    covariance_matrix = np.cov(data.T)
    
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    # The eigenvectors are the principal components of the data set
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    # Sort the eigenvectors by the eigenvalues
    # This is done so that the eigenvectors are in descending order of importance
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Project the data set onto the eigenvectors
    # This is done by taking the dot product of the data set and the eigenvectors
    data_pca = np.dot(data, eigenvectors)

    # The data set in the new coordinate system whcih is the principal component space
    # THe dimensions of the data set have been reduced to the number of principal components
    
    return data_pca