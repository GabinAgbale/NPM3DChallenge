#
#
#      0=============================0
#      |    TP4 Point Descriptors    |
#      0=============================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#

 
# Import numpy package and name it "np"
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# FLANN for fast point matching
import cv2 as cv

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from tqdm import tqdm


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#    


def PCA(points):

    N = points.shape[0]
    # Compute PCA on a given point neighborhood
    g = np.mean(points, axis=0)
    centered_cloud = (points - g).T
    M = 1/N * centered_cloud @ centered_cloud.T
    eigenvalues, eigenvectors = np.linalg.eigh(M)


    return eigenvalues, eigenvectors



def compute_local_PCA(query_points, cloud_points, method='radius', 
                      radius=0.5, k=30, leaf_size=500):

    d = query_points.shape[1]
    # tree search on query_points
    tree = KDTree(cloud_points, leaf_size=leaf_size)

    if method=='radius':
        indices = tree.query_radius(query_points, r=radius, 
                                    return_distance=False)
    else:
        indices = tree.query(query_points, k=k, 
                             return_distance=False)

    all_eigenvalues = np.zeros((query_points.shape[0], d))
    all_eigenvectors = np.zeros((query_points.shape[0], d, d))

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    for i, n in enumerate(indices): 
        all_eigenvalues[i], all_eigenvectors[i] = PCA(cloud_points[n])

    return all_eigenvalues, all_eigenvectors



def compute_local_PCA_flann(query_points, cloud_points, method='radius', 
                      radius=0.5, k=30):

    d = query_points.shape[1]
    
    # flann
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50) 
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(query_points, cloud_points, k=k)


    all_eigenvalues = np.zeros((query_points.shape[0], d))
    all_eigenvectors = np.zeros((query_points.shape[0], d, d))

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    # for i, n in enumerate(indices): 
    #     all_eigenvalues[i], all_eigenvectors[i] = PCA(cloud_points[n])

    return matches




def compute_features(query_points, cloud_points,radius=0.5):

    N = query_points.shape[0]
    num_features = 8
    eps = 1e-5
    all_eigenvalues, all_eigenvectors = compute_local_PCA(query_points, cloud_points,method='nn',
                                                          k=30)

    features = np.zeros((N,num_features))

    features[:,0] = 2 * np.arcsin(np.abs(np.dot(all_eigenvectors[:,:,0], np.array([1,0,0]))))/np.pi
    features[:,1] = 1 - all_eigenvalues[:,1]/(all_eigenvalues[:,2] + eps)
    features[:,2] = (all_eigenvalues[:,1] - all_eigenvalues[:,0])/(all_eigenvalues[:,2] + eps)
    features[:,3] = all_eigenvalues[:,0] / (all_eigenvalues[:,2] + eps)
    features[:,4] = (np.prod(all_eigenvalues, axis=1))**1/3
    features[:,5] = -np.sum(all_eigenvalues * np.log(all_eigenvalues + eps), axis=1)
    features[:,6] = np.sum(all_eigenvalues, axis=1)
    features[:,7] = all_eigenvalues[:,0]/(features[:,6] + eps)


    return features


def compute_2d_features(query_points, cloud_points, radius=0.5):
    N = query_points.shape[0]
    
    num_features = 6
    eps = 1e-5
    all_features = []
    N = query_points.shape[0]
    for axe in range(3):
        # project the cloud
        axes = [i for i in range(3) if i != axe]
        query_points2d = query_points[:,axes]
        cloud_points2d = cloud_points[:,axes]
        
        all_eigenvalues, all_eigenvectors = compute_local_PCA(query_points2d, cloud_points2d, method='nn',
                                                              k=30)

        features = np.zeros((N,num_features))
        features[:,0] = 2 * np.arcsin(np.abs(np.dot(all_eigenvectors[:,:,0], np.array([1,0]))))/np.pi
        features[:,1] = 1 - all_eigenvalues[:,0]/(all_eigenvalues[:,1] + eps)
        features[:,2] = (np.prod(all_eigenvalues, axis=1))**1/2
        features[:,3] = -np.sum(all_eigenvalues * np.log(all_eigenvalues + eps), axis=1)
        features[:,4] = np.sum(all_eigenvalues, axis=1)
        features[:,5] = all_eigenvalues[:,0]/(features[:,4] + eps)

        all_features.append(features)
    
    return np.concatenate(all_features, axis=1)











# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # PCA verification
    # ****************
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        eigenvalues, eigenvectors = PCA(cloud)

        # Print your result
        print(eigenvalues)

        # Expected values :
        #
        #   [lambda_3; lambda_2; lambda_1] = [ 5.25050177 21.7893201  89.58924003]
        #
        #   (the convention is always lambda_1 >= lambda_2 >= lambda_3)
        #

		
    # Normal computation
    # ******************
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        
        leaf_size = 50
        t1 = time.time()
        # Compute PCA on the whole cloud
        all_eigenvalues, all_eigenvectors = compute_local_PCA(cloud, cloud, radius=0.5, leaf_size=leaf_size)
        normals = all_eigenvectors[:, :, 0]
        print(f'Leaf size {leaf_size} | {cloud.shape[0]} points | computed in {np.round(time.time() - t1, 1)} sec')

        # Save cloud with normals
        #write_ply('../Lille_street_small_normals.ply', (cloud, normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])



    if False:
        # Load cloud as a [N x 3] matrix
        filename='MiniLille2'
        cloud_path = f'../data/Benchmark_MVA/training/{filename}.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        verticality, linearity, planarity, sphericity = compute_features(cloud, cloud, radius=0.5)

        # Save cloud with normals
        # write_ply(f'../{filename}.ply', [cloud, verticality, linearity, planarity, sphericity], ['x', 'y', 'z', 'v', 'l', 'p', 's'])

    

    if True:
        
        for filename in ['MiniLille2', 'MiniLille1', 'MiniParis1']:
            cloud_path = f'../data/Benchmark_MVA/training/{filename}.ply'
            cloud_ply = read_ply(cloud_path)
            cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
            
            
            dl = 0.5

        cloud = cloud[::300]

        features = compute_2d_features(cloud, cloud, radius=0.5)
      


        # # Save cloud with normals
        # write_ply(f'../{filename}_sampled_features.ply', [cloud, verticality, linearity, planarity, sphericity], ['x', 'y', 'z', 'v', 'l', 'p', 's'])



        # verticality1, linearity1, planarity1, sphericity1 = compute_features(cloud, cloud, radius=0.5)

        # # Save cloud with normals
        # write_ply(f'../{filename}_features.ply', [cloud, verticality1, linearity1, planarity1, sphericity1], ['x', 'y', 'z', 'v', 'l', 'p', 's'])