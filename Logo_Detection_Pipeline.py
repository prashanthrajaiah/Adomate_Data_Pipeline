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
from Modules.logo_detection.keras_retinanet.utils.keras_version import check_keras_version


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
        # if self.random_transform == True:
        #     print("&&&& Creating random Transform Generator &&&&&&&")
        #     transform_generator = random_transform_generator(
        #         min_rotation=-0.1,
        #         max_rotation=0.1,
        #         min_translation=(-0.1, -0.1),
        #         max_translation=(0.1, 0.1),
        #         min_shear=-0.1,
        #         max_shear=0.1,
        #         min_scaling=(0.9, 0.9),
        #         max_scaling=(1.1, 1.1),
        #         flip_x_chance=0.5,
        #         flip_y_chance=0.5,
        #     )
        # else:
        #     transform_generator = random_transform_generator(flip_x_chance=0.5)


        if args.dataset_type == 'csv':
            train_generator = CSVGenerator(
                args.annotations,
                args.classes,
                #transform_generator = transform_generator,
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
   Model Training Layer: Module taking care of the data training operations of the Logo detection module.
"""

class ModelTrainingLogoDetection(luigi.Task):
    """
    Luigi Task to handle the Model Training of Image Detection Model workflow.
    """
    args_list_v2 = luigi.Parameter()
    random_transform_v2 = luigi.Parameter()

    data_text_v2 = "Model Training Phase has been Succesfully completed"
    self.args = self.parse_args(self.args_listv2)

    def requires(self):
        return [DataPreprocessingLogoDetection(args_list_v1=self.data_text_v2, random_transform_v1=self.random_transform_v2)]

    def run(self):
        # create object that stores backbone information
        backbone = models.backbone(args.backbone)
        self.args_listv3 = self.args_list_v2.strip('][')
        self.args_listv4 = self.args_listv3.split(',')
        self.args = self.parse_args(self.args_listv4)
        backbone = models.backbone(self.args.backbone)

        #Making sure that the keras is of required minimum versionself
        check_keras_version()

        # Setting up session for GPU
        def get_session():
            """ Construct a modified tf session.
            """
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            return tf.Session(config=config)

        # optionally choose specific GPU
        if self.args.gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu
        keras.backend.tensorflow_backend.set_session(get_session())


        # create the model
        if args.snapshot is not None:
            print('Loading model, this may take a second...')
            model            = models.load_model(args.snapshot, backbone_name=args.backbone)
            training_model   = model
            anchor_params    = None
            if args.config and 'anchor_parameters' in args.config:
                anchor_params = parse_anchor_parameters(args.config)
            prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)
        else:
            weights = args.weights
            # default to imagenet if nothing else is specified
            if weights is None and args.imagenet_weights:
                weights = backbone.download_imagenet()

            print('Creating model, this may take a second...')
            model, training_model, prediction_model = create_models(
                backbone_retinanet=backbone.retinanet,
                num_classes=train_generator.num_classes(),
                weights=weights,
                multi_gpu=args.multi_gpu,
                freeze_backbone=args.freeze_backbone,
                lr=args.lr,
                config=args.config
            )

        # this lets the generator compute backbone layer shapes using the actual backbone model
        if 'vgg' in args.backbone or 'densenet' in args.backbone:
            train_generator.compute_shapes = make_shapes_callback(model)
            if validation_generator:
                validation_generator.compute_shapes = train_generator.compute_shapes

        csv_log = CSVLogger("training_coco_weights.log", separator=',', append= True)

        # create the callbacks
        callbacks = create_callbacks(
            model,
            training_model,
            prediction_model,
            validation_generator,
            csv_log,
            args
        )

        # Use multiprocessing if workers > 0
        if args.workers > 0:
            use_multiprocessing = True
        else:
            use_multiprocessing = False

        if not args.compute_val_loss:
            validation_generator = None


        # start training
        print("Started Training !!!!")
        return training_model.fit_generator(
            generator=train_generator,
            steps_per_epoch=args.steps,
            epochs=args.epochs,
            verbose=1,
            callbacks=callbacks,
            workers=args.workers,
            use_multiprocessing=use_multiprocessing,
            max_queue_size=args.max_queue_size,
            validation_data=validation_generator
        )

        print("Succesfully Done!!")
        with self.output().open('w') as out_file:
            out_file.write(self.data_text_v2)

    def output(self):
        return luigi.LocalTarget('./Results/samplemodeltraining.txt')


"""
Model Inference Layer : Module taking care of the inference part of the logo detection module.
"""

class ModelInferenceLogoDetection(luigi.Task):
    """
    Luigi Task to handle inference part of the logo detection model.
    """
    arg_list_inference = luigi.Parameter()
    rand_transform_inference = luigi.Parameter()

    def requires(self):
        return [ModelTrainingLogoDetection(args_list_v2=self.arg_list_inference, random_transform_v2=self.rand_transform_inference)]

    # The GPU id to use, usually either "0" or "1";
    os.environ["CUDA_VISIBLE_DEVICES"]="0";

    #PASSING DEFAULT ARGUMENTS
    def parse_args(args):
        # construct the argument parse and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", required=True,help="path to input image")
        return ap.parse_args(args)

    def process(img):
        # load the input image (in BGR order), clone it, and preprocess it
        image = read_image_bgr(img)
        output_image = image.copy()
        image = preprocess_image(image)
        (image, scale) = resize_image(image)
        image = np.expand_dims(image, axis=0)

        # detect objects in the input image and correct for the image scale
        (boxes, scores, labels) = model.predict_on_batch(image)
        boxes /= scale
        boxes=boxes.astype('int')

        #Creatung the Logo dataframe from the model and retaining rows where confidence is high
        Logo=pd.DataFrame(zip(boxes[0], scores[0], labels[0]),columns=['bbox','conf','label'])
        Logo=Logo[Logo['conf'] >.50]
        Logo['label']=Logo['label'].apply(lambda x:LABELS[x])

        return Logo,output_image

    def op(ocr,Logo):
        #Iou over Logo and cor and creating final results dataframe
        if ocr is not None :

            output=pd.DataFrame(columns=['bbox', 'conf', 'label', 'bbox_ocr', 'conf_ocr', 'text','class'])
            inter=0

            for index,logo_row in Logo.iterrows():

                for i,ocr_row in ocr.iterrows():

                    iou =bb_intersection_over_union(logo_row['bbox'],ocr_row['bbox_ocr'])

                    if iou > .60:
                        inter +=1
                        #If ocr['text'] in Logo['label']and also if mention present in Logo['label']
                        if re.search(ocr_row['text'], logo_row['label'], re.IGNORECASE) and "Mention" in logo_row['label']:

                            prob =0.2 * logo_row['conf'] + 0.8 * ocr_row['conf_ocr']/100
                            output =output.append(logo_row.to_frame().T.reset_index(drop=True).join(ocr_row.to_frame().T.reset_index(drop=True)),ignore_index=True,sort=False)
                            output.loc[output.index[-1],'class']=logo_row['label']
                            output.loc[output.index[-1],'prob'] = prob
                        else:
                            #Creating a prob based on both logo and ocr and giving a weihted prob and both the labels
                            prob =0.8 * logo_row['conf'] + 0.2 * ocr_row['conf_ocr']/100
                            output =output.append(logo_row.to_frame().T.reset_index(drop=True).join(ocr_row.to_frame().T.reset_index(drop=True)),ignore_index=True,sort=False)
                            output.loc[output.index[-1],'class']=logo_row['label'] + '/' + ocr_row['text']+ '_' +'Mention'
                            output.loc[output.index[-1],'prob'] = prob

                if inter >0:
                    #If intersections detected
                    result=pd.concat([output,Logo],ignore_index=True,sort=False)
                    try:
                        result=result[~result['bbox'].duplicated()]
                    except:
                        pass
                    result.loc[result['class'].isnull(),'class'] = result['label']
                    result.loc[result['prob'].isnull(),'prob'] = result['conf']
                else :
                    #if no intersections appned the Logo and ocr
                    ocr.columns=Logo.columns
                    result =Logo.append(ocr)
                    result.columns = ['bbox','prob','class']
        else :
            #if no ocr just use the LOGO
            result =Logo.copy()
            result.columns = ['bbox','prob','class']

        return result


    def run(luigi.Task):
        model = '../../inference_models/inference_micro_intell_add.h5'
        labels = '../../Data/Microsoft_Data/retinanet_classes_add.csv'
        confidence = 0.55
        # load the class label mappings
        LABELS = open(labels).read().strip().split("\n")
        LABELS = {int(L.split(",")[1]): L.split(",")[0] for L in LABELS}

        # load the model from disk
        model = models.load_model(model, backbone_name="resnet50")
        args=parse_args(args)
        print('y')
        ocr = get_ocr(args.image)
        Logo,output_image=process(args.image)
        result=op(ocr,Logo)
        print(result)

    def complete(self):
        print("******* The data pipelining has been successfully completed")
