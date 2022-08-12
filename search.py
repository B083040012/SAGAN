from re import search
import time, logging, supernet
import numpy as np
import tensorflow as tf
from ASAGA import ASAGA_Searcher
from file_loader import File_Loader
from tensorflow.python.keras import backend as K

class Search():
    def __init__(self, dataset_type, config):
        
        self.dataset_type = dataset_type
        self.config = config

    def set_logger(self, test_dir):
        """
        Initialize logger
        """
        formatter = logging.Formatter('%(asctime)s %(message)s')
        file_handler = logging.FileHandler(test_dir + self.config["file"]["log_path"] + "Architecture_searching.log")
        stream_handler = logging.StreamHandler()
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger = logging.getLogger('architecture_searching')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        self.logger = logging.getLogger('architecture_searching')
        self.test_dir = test_dir

    def load_data(self, logger):
        """
        Load validation dataset
        """
        my_data_loader = File_Loader(self.dataset_type, self.config, limit_dataset = True)
        if self.dataset_type == "station":
            att_cnn, att_flow, att_lstm, att_weather, short_cnn, short_flow, short_lstm, short_weather, short_poi, y = my_data_loader.sample(datatype = "validation")
            self.val_data = [att_cnn, att_flow, att_lstm, att_weather, short_cnn, short_flow, short_lstm, short_weather, short_poi]
            self.val_label = y

            logger.info("[Agent] shapes of each station-level inputs for operation Search")
            logger.info("att_cnn: {0}, {1}".format(len(att_cnn), att_cnn[0].shape))
            logger.info("att_flow: {0}, {1}".format(len(att_flow), att_flow[0].shape))
            logger.info("att_lstm: {0}, {1}".format(len(att_lstm), att_lstm[0].shape))
            logger.info("att_weather: {0}, {1}".format(len(att_weather), att_weather[0].shape))
            logger.info("short_cnn: {0}, {1}".format(len(short_cnn), short_cnn[0].shape))
            logger.info("short_flow: {0}, {1}".format(len(short_flow), short_flow[0].shape))
            logger.info("short_lstm: {0}".format(short_lstm.shape))
            logger.info("short_weather: {0}".format(short_weather.shape))
            logger.info("short poi: {0}".format(short_poi.shape))
            logger.info("y: {0}".format(y.shape))

        elif self.dataset_type == "region":
            att_cnn, att_flow, att_lstm, att_weather, short_cnn, short_flow, short_lstm, short_weather, y = my_data_loader.sample(datatype = "validation")
            self.val_data = [att_cnn, att_flow, att_lstm, att_weather, short_cnn, short_flow, short_lstm, short_weather]
            self.val_label = y

            logger.info("[Agent] shapes of each region-level inputs for operation Search")
            logger.info("att_cnn: {0}, {1}".format(len(att_cnn), att_cnn[0].shape))
            logger.info("att_flow: {0}, {1}".format(len(att_flow), att_flow[0].shape))
            logger.info("att_lstm: {0}, {1}".format(len(att_lstm), att_lstm[0].shape))
            logger.info("att_weather: {0}, {1}".format(len(att_weather), att_weather[0].shape))
            logger.info("short_cnn: {0}, {1}".format(len(short_cnn), short_cnn[0].shape))
            logger.info("short_flow: {0}, {1}".format(len(short_flow), short_flow[0].shape))
            logger.info("short_lstm: {0}".format(short_lstm.shape))
            logger.info("short_weather: {0}".format(short_weather.shape))
            logger.info("y: {0}".format(y.shape))

    def load_model(self):
        """
        Load pretrained supernet
            1. loading subclass supernet model
            2. loading weight from pretrained best supernet
            * run functions eagerly
        """
        # create session, unlimit gpu and run eagerly
        K.clear_session()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        K.set_session(tf.compat.v1.Session(config=config))
        tf.config.run_functions_eagerly(True)
        
        # loading supernet model
        self.logger.info("[Architecture Searching] loading supernet...")
        if self.dataset_type == "station":
            self.model = supernet.Station_Supernet_Subclass_Model(self.config)
        elif self.dataset_type == 'region':
            self.model = supernet.Region_Supernet_Subclass_Model(self.config)
        self.logger.info("[Architecture Searching] supernet model loading complete")

        # loading pretrained weight
        self.logger.info("[Architecture Searching] loading pretrained best supernet weight...")
        weight_file_path = self.test_dir + self.config["file"]["model_path"] + 'best_supernet_weight'
        # expect partial for clean warning: only use partial variables in model when predicting (ex: optimizer)
        self.model.load_weights(weight_file_path).expect_partial()
        self.logger.info("[Architecture Searching] pretrained weight loading complete")

    def search(self):
        """
        Searching best architecture by searching strategy
            1. searching architecture by searcher
            2. saving the searched architecture
        """
        # searching architecture and get the loss value
        self.logger.info("[Architecture Searching] searching start...")
        start_time = time.time()
        searcher = ASAGA_Searcher(self.config, self.logger, self.model, self.val_data, self.val_label, self.dataset_type)
        searched_architecture, loss = searcher.search_architecture(self.val_data, self.val_label)
        end_time = time.time()
        self.logger.info("[Architecture Searching] searching complete")
        self.logger.info("[Architecture Searching] total searching time: {0} min, architecture loss: {1}".format((end_time - start_time) / 60, loss))

        # saving searched architecture
        architecture_file_path = self.test_dir + self.config["file"]["model_path"] + "searched_choice_list"
        for index in range(len(searched_architecture)):
            searched_architecture[index] = searched_architecture[index].tolist()
        self.logger.info("[Architecture Searching] searched architecture: {0}".format(searched_architecture))
        searched_architecture = np.array(searched_architecture)
        np.save(architecture_file_path, searched_architecture)
        self.logger.info("[Architecture Searching] architecture saved")

        del self.model
        K.clear_session()

    def process(self, test_dir):
        self.set_logger(test_dir)
        self.load_model()
        self.search()