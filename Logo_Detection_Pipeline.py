"""
   Logo Detection Pipeline: Module taking care of the data pipeline for logo detection for Adomate
"""
from __future__ import print_function
import luigi
import os
import argparse
import sys
import ast
import subprocess
import random
import time
import pickle
from argparse import Namespace

import Modules.logo_detection.keras_retinanet.bin
from Modules.logo_detection.keras_retinanet.preprocessing.csv_generator import CSVGenerator
from Modules.logo_detection.keras_retinanet.utils.anchors import make_shapes_callback
from Modules.logo_detection.keras_retinanet.utils.config import read_config_file, parse_anchor_parameters
from Modules.logo_detection.keras_retinanet.utils.transform import random_transform_generator


"""
Data Ingestion Layer: Layer taking care of the data ingestion/loading operations for the logo detection
Modules.

"""

class DataIngestionLogoDetection(luigi.Task):
    """
    Luigi task to create image data generators for logo detection model.
    """
    # Reading Command Line Parameters
    args_list = luigi.Parameter()
    random_transform = luigi.Parameter()



    def create_generators(self, args, preprocess_image=None):

        """ Create generators for training and validation.

        Args
            args             : parseargs object containing configuration for generators.
            preprocess_image : Function that preprocesses an image for the network.
        """

        common_args = {
            'batch_size'       : args.batch_size,
            'config'           : args.config,
            'image_min_side'   : args.image_min_side,
            'image_max_side'   : args.image_max_side,
            # 'preprocess_image' : preprocess_image,
        }


        self.random_transform = bool(self.random_transform)
        # create random transform generator for augmenting training data
        if self.random_transform == True:
            print("&&&& Creating random Transform Generator &&&&&&&")
            transform_generator = random_transform_generator(
                min_rotation=-0.1,
                max_rotation=0.1,
                min_translation=(-0.1, -0.1),
                max_translation=(0.1, 0.1),
                min_shear=-0.1,
                max_shear=0.1,
                min_scaling=(0.9, 0.9),
                max_scaling=(1.1, 1.1),
                flip_x_chance=0.5,
                flip_y_chance=0.5,
            )
        else:
            transform_generator = random_transform_generator(flip_x_chance=0.5)


        if args.dataset_type == 'csv':
            train_generator = CSVGenerator(
                args.annotations,
                args.classes,
                transform_generator = transform_generator,
                **common_args
            )

            if args.val_annotations:
                validation_generator = CSVGenerator(
                    args.val_annotations,
                    args.classes,
                    **common_args
                )
            else:
                validation_generator = None
        else:
            raise ValueError('Invalid data type received: {}'.format(args.dataset_type))
        print("Returning both the generated errors")
        return train_generator, validation_generator


    def parse_args(self, arguments):
        """
         Parse the arguments.
        """
        parser = argparse.ArgumentParser(description='Simple script for Data Ingestion into a RetinaNet network.')
        subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
        subparsers.required = True

        def csv_list(string):
            return string.split(',')

        oid_parser = subparsers.add_parser('oid')
        oid_parser.add_argument('main_dir', help='Path to dataset directory.')
        oid_parser.add_argument('--version',  help='The current dataset version is v4.', default='v4')
        oid_parser.add_argument('--labels-filter',  help='A list of labels to filter.', type=csv_list, default=None)
        oid_parser.add_argument('--annotation-cache-dir', help='Path to store annotation cache.', default='.')
        oid_parser.add_argument('--parent-label', help='Use the hierarchy children of this label.', default=None)

        csv_parser = subparsers.add_parser('csv')
        csv_parser.add_argument('--annotations', help='Path to CSV file containing annotations for training.')
        csv_parser.add_argument('--classes', help='Path to a CSV file containing class label mapping.')
        csv_parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).')

        parser.add_argument('--batch-size',       help='Size of the batches.', default=1, type=int)
        parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=50)
        parser.add_argument('--steps',            help='Number of steps per epoch.', type=int, default=10000)
        parser.add_argument('--no-evaluation',    help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
        parser.add_argument('--freeze-backbone',  help='Freeze training of backbone layers.', action='store_true')
        parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
        parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
        parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
        parser.add_argument('--config',           help='Path to a configuration parameters .ini file.')
        parser.add_argument('--weighted-average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')
        parser.add_argument('--compute-val-loss', help='Compute validation loss during training', dest='compute_val_loss', action='store_true')

        # Fit generator arguments
        parser.add_argument('--workers', help='Number of multiprocessing workers. To disable multiprocessing, set workers to 0', type=int, default=1)
        parser.add_argument('--max-queue-size', help='Queue length for multiprocessing workers in fit generator.', type=int, default=10)

        return parser.parse_args(arguments)

    def run(self):
        self.args_listv1 = self.args_list.strip('][')
        self.args_listv2 = self.args_listv1.split(',')
        self.args = self.parse_args(self.args_listv2)
        self.train_generator, self.validation_generator = self.create_generators(self.args, 'resnet50')
        print("Train Generator Properties :", self.train_generator.__len__())
        print("Test Generator Properties: ", self.validation_generator.__len__())
        print("The type of generators", type(self.train_generator))

        target_file_path = './Results/logo_detection_validation_data_generators.pickle'
        outFile_validation = open(target_file_path, 'wb')
        pickle.dump(self.validation_generator, outFile_validation)
        outFile = open(self.output().path, 'wb')
        pickle.dump(list(self.train_generator), outFile)


    def output(self):
        return luigi.LocalTarget('./Results/logo_detection_train_data_generators.pickle')


"""
   Data Preprocessing Layer: Module taking care of the data Preprocessing Operations of the different
   Modules.
"""

class DataPreprocessingLogoDetection(luigi.Task):
    """
    Luigi Task to handle the preprocessing of Image Detection Model workflow.
    """
    args_list_v1 = luigi.Parameter()
    random_transform_v1 = luigi.Parameter()
    data_text = "Logo Detection Pre-processing completed!!!!"
    train_pickle_dump_path = './Results/logo_detection_train_data_generators.pickle'
    validation_pickle_dump_path = './Results/logo_detection_validation_data_generators.pickle'

    def requires(self):
        return [DataIngestionLogoDetection(args_list = self.args_list_v1, random_transform = self.random_transform_v1)]

    # Doesn't Serve any purpose as of now.
    def run(self):
        pickle_off_train_validator = open(self.train_pickle_dump_path,"rb")
        train_genrator_v1 = pickle.load(pickle_off_train_validator)

        pickle_off_test_validator = open(self.validation_pickle_dump_path,"rb")
        validation_genrator_v1 = pickle.load(pickle_off_test_validator)

        with self.output().open('w') as f:
            f.write(self.data_text)

    def output(self):
        return luigi.LocalTarget('./Results/Preprocessing_completion.txt')


"""
   Data Inference Layer: Module taking care of the data training operations of the different
   Modules.
"""

# class ModelTrainingLogoDetection(luigi.Task):
#     """
#     Luigi Task to handle the Model Training of Image Detection Model workflow.
#     """
#     data_text_v1 = luigi.Parameter()
#
#     def requires(self):
#         return [DataPreprocessingLogoDetection(data_text=self.data_text_v1)]
#
#     def run(self):
#         print("Succesfully Done!!")
#         with self.output().open('w') as out_file:
#             out_file.write(self.data_text_v1)
#
#     def output(self):
#         return luigi.LocalTarget('./Results/sample1.txt')
