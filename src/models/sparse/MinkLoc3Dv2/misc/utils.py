# Adapted from https://github.com/jkarolczak/ligand-classification
# Warsaw University of Technology

import os
import configparser
import time
import numpy as np

from cfg import read_config


class ModelParams:
    def __init__(self, model_params_path):
        config = read_config(model_params_path)
        self.model_params_path = model_params_path
        self.model = "MinkLoc3Dv2"
        self.output_dim = config['output_dim']      # Size of the final descriptor
        self.input_dim = config['input_dim']
        #######################################################################
        # Model dependent
        #######################################################################

        self.coordinates = config['coordinates']
        assert self.coordinates in ['polar', 'cartesian'], f'Unsupported coordinates: {self.coordinates}'

        # Use cosine similarity instead of Euclidean distance
        # When Euclidean distance is used, embedding normalization is optional
        self.normalize_embeddings = config['normalize_embeddings']  # False

        # Size of the local features from backbone network (only for MinkNet based models)
        self.feature_size = config['feature_size']
        if 'planes' in config:
            self.planes = tuple([int(e) for e in config['planes']])
        else:
            self.planes = tuple([32, 64, 64])

        if 'layers' in config:
            self.layers = tuple([int(e) for e in config['layers']])
        else:
            self.layers = tuple([1, 1, 1])

        self.num_top_down = config['num_top_down']
        self.conv0_kernel_size = config['conv0_kernel_size']
        self.block = config['block']
        self.pooling = config['pooling']

    def print(self):
        print('Model parameters:')
        param_dict = vars(self)
        for e in param_dict:
            print('{}: {}'.format(e, param_dict[e]))

        print('')
