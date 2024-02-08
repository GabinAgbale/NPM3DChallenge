"""
https://arxiv.org/pdf/1808.00495.pdf%20https://arxiv.org/pdf/1612.00593.pdf
https://ethz.ch/content/dam/ethz/special-interest/baug/igp/photogrammetry-remote-sensing-dam/documents/pdf/timo-jan-isprs2016.pdf
"""


# Import functions to read and write ply files
from ply import write_ply, read_ply

import numpy as np 

# Import functions for features computation
from descriptors import compute_features, compute_2d_features

# Import time package
import time

from os import listdir
from os.path import exists, join

from tqdm import tqdm




def grid_subsampling(points, dl=1):

    """
    Given a voxel size, subsample the cloud with one point (barycenter) per voxel
    dl : voxel size -> dl x dl x dl
    """

    # determine the voxel associated to each point, as 3 dimensions
    # vector giving the position of the voxel in the grid
    voxels = (points / dl).astype(int)
    
    # create a hash number to each voxel for faster retrieval
    hash_ids = np.array([hash(tuple(v)) for v in voxels])
    unique_hashes = np.unique(hash_ids)
    print(f'---Grid Subsampling on {points.shape[0]} ---')
    print(f'---Voxel size: {dl} || {len(unique_hashes)} Voxels---')

    sampled_points = np.zeros((len(unique_hashes), 3))
    sampled_colors = np.zeros((len(unique_hashes), 3))
    sampled_labels = np.zeros(len(unique_hashes))
    
    # iterate over unique hashes (unique voxel) and compute barycenter
    for i, h in enumerate(tqdm(unique_hashes)):
        indices = (hash_ids == h)
        pts_in_voxel = points[indices]
        sampled_points[i] = np.mean(pts_in_voxel, axis=0)

    return sampled_points



class MultiscaleFeaturesExtractor:
    """
    Class that computes multiscale features from point clouds
    """

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, path, voxel_sizes):
        """
        Initiation method called when an object of this class is created. This is where you can define parameters
        """

        # Neighborhood radius
        self.radius = 0.5

        # Number of training points per class
        self.num_per_class = 500

        # Classification labels
        self.label_names = {0: 'Unclassified',
                            1: 'Ground',
                            2: 'Building',
                            3: 'Poles',
                            4: 'Pedestrians',
                            5: 'Cars',
                            6: 'Vegetation'}
        
        self.voxel_sizes = voxel_sizes
        
        self.path = path

    # Methods
    # ------------------------------------------------------------------------------------------------------------------
        




    def extract_training(self):
            """
            This method extract features/labels of a subset of the training points. It ensures a balanced choice between
            classes.
            :param path: path where the ply files are located.
            :return: features and labels
            """

            # Get all the ply files in data folder
            ply_files = [f for f in listdir(self.path) if f.endswith('.ply')]

            training_labels = np.empty((0,))


            all_features = []


            # Loop over each training cloud
            
            for i, file in enumerate(ply_files):
                print(f'Loading cloud {file}...')
                # Load Training cloud
                cloud_ply = read_ply(join(self.path, file))
                points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
                labels = cloud_ply['class']

                # Initiate training indices array
                training_inds = np.empty(0, dtype=np.int32)

                # Loop over each class to choose training points
                for label, name in self.label_names.items():

                    # Do not include class 0 in training
                    if label == 0:
                        continue

                    # Collect all indices of the current class
                    label_inds = np.where(labels == label)[0]

                    # If you have not enough indices, just take all of them
                    if len(label_inds) <= self.num_per_class:
                        training_inds = np.hstack((training_inds, label_inds))

                    # If you have more than enough indices, choose randomly
                    else:
                        random_choice = np.random.choice(len(label_inds), self.num_per_class, replace=False)
                        training_inds = np.hstack((training_inds, label_inds[random_choice]))

                # Gather chosen points
                training_points = points[training_inds, :]
                training_labels = np.hstack((training_labels, labels[training_inds]))
                
                print(f'Computing features on the full cloud...')
                # Compute features for the points of the chosen indices 
                features3d = compute_features(training_points, points, self.radius)
                features2d = compute_2d_features(training_points, points, self.radius)

                cloud_features = np.hstack((features3d, features2d))

                # Compute features for every subsampled cloud, determined by the voxel size of the grid
                sampled_cloud = np.copy(points)
                
                print('Computing multiscale features...')
                for dl in self.voxel_sizes:
                    sampled_cloud = grid_subsampling(sampled_cloud, dl)

                    features3d = compute_features(training_points, sampled_cloud, self.radius)
                    features2d = compute_2d_features(training_points, sampled_cloud, self.radius)

                    cloud_features = np.hstack((cloud_features, np.hstack((features3d, features2d))))
                
                all_features.append(cloud_features)
        
            
            return np.concatenate(all_features, axis=0), training_labels



    def extract_test(self, path):
        """
        This method extract features of all the test points.
        :param path: path where the ply files are located.
        :return: features
        """

        # Get all the ply files in data folder
        ply_files = [f for f in listdir(path) if f.endswith('.ply')]

        # Initiate arrays
        test_features = np.empty((0, 4))

        # Loop over each training cloud
        for i, file in enumerate(ply_files):

            # Load Training cloud
            cloud_ply = read_ply(join(path, file))
            points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

            # Compute features only one time and save them for further use
            #
            #   WARNING : This will save you some time but do not forget to delete your features file if you change
            #             your features. Otherwise you will not compute them and use the previous ones
            #

            # Name the feature file after the ply file.
            feature_file = file[:-4] + '_features.npy'
            feature_file = join(path, feature_file)

            # If the file exists load the previously computed features
            if exists(join(path, feature_file)):
                features = np.load(feature_file)

            # If the file does not exist, compute the features (very long) and save them for future use
            else:
                features = compute_features(points, points, self.radius)
                features = np.column_stack(features)
        

            # Concatenate features of several clouds
            # (For this minichallenge this is useless as the test set contains only one cloud)
            test_features = np.vstack((test_features, features))

        return test_features



if __name__ == "__main__":

    path = '../data/Benchmark_MVA/training'
    mse = MultiscaleFeaturesExtractor(path, [10])
    features, labels = mse.extract_training()


    
