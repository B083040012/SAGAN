import logging, retrain_network
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K
from file_loader import File_Loader
from criterion import eval_lstm, eval_together

class Evaluate_Architecture():
    def __init__(self, dataset_type, config):
        """
        Initialize parameters
        """
        self.dataset_type = dataset_type
        self.config = config
        self.optimizer = config["training"]["architecture_retrain"]["optimizer"]
        self.loss = config["training"]["architecture_retrain"]["loss"]

        # used for denormalized
        target = config["dataset"][dataset_type]["pred_target"]
        if target != "volume" and target != "flow":
            print("invalid target")
            raise Exception
        if dataset_type == "station":
            self.label_max = config["dataset"]["station"]["single_volume_test_max"] if \
                target == "volume" else config["dataset"]["station"]["single_flow_test_max"]
        elif dataset_type == 'region':
            if target != "volume":
                print("invalid target for region-level")
                raise Exception
            else:
                self.label_max = config["dataset"]["region"]["volume_test_max"]

        # normalize thershold
        self.threshold_denormalize = config["dataset"][dataset_type]["threshold"]
        self.threshold = config["dataset"][dataset_type]["threshold"] / self.label_max

    def set_logger(self, test_dir):
        """
        Initialize logger and test_dir
        """
        formatter = logging.Formatter('%(asctime)s %(message)s')
        file_handler = logging.FileHandler(test_dir + self.config["file"]["log_path"] + "Evaluate_Architecture.log")
        stream_handler = logging.StreamHandler()
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger = logging.getLogger('evaluate_architecture' + test_dir)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        self.logger = logging.getLogger('evaluate_architecture' + test_dir)
        self.test_dir = test_dir

    def load_data(self, logger):
        """
        Loading test data depend on the dataset_type
        """
        my_data_loader = File_Loader(self.dataset_type, self.config)
        if self.dataset_type == "station":
            att_cnn, att_flow, att_lstm, att_weather, short_cnn, short_flow, short_lstm, short_weather, y = my_data_loader.sample(datatype = "test")
            self.test_data = att_cnn + att_flow + att_lstm + att_weather + short_cnn + short_flow + [short_lstm, ] + [short_weather, ]
            self.test_label = y

            self.feature_vec_len = short_lstm.shape[-1]
            self.nbhd_size = short_cnn[0].shape[1]
            self.nbhd_type = short_cnn[0].shape[-1]
            self.flow_type = short_flow[0].shape[-1]
            self.weather_type = short_weather.shape[-1]

            # if label are all below threshold, end the program
            valid_label = np.sum(self.test_label > self.threshold)
            if valid_label == 0:
                self.logger.info("[Evaluate Architecture] invalid label !!")
                raise Exception

            logger.info("[Agent] shapes of each station-level inputs for operation Evaluate_Architecture")
            logger.info("att_cnn: {0}, {1}".format(len(att_cnn), att_cnn[0].shape))
            logger.info("att_flow: {0}, {1}".format(len(att_flow), att_flow[0].shape))
            logger.info("att_lstm: {0}, {1}".format(len(att_lstm), att_lstm[0].shape))
            logger.info("att_weather: {0}, {1}".format(len(att_weather), att_weather[0].shape))
            logger.info("short_cnn: {0}, {1}".format(len(short_cnn), short_cnn[0].shape))
            logger.info("short_flow: {0}, {1}".format(len(short_flow), short_flow[0].shape))
            logger.info("short_lstm: {0}".format(short_lstm.shape))
            logger.info("short_weather: {0}".format(short_weather.shape))
            logger.info("y: {0}".format(y.shape))

        elif self.dataset_type == "region":
            att_cnn, att_flow, att_lstm, att_weather, short_cnn, short_flow, short_lstm, short_weather, y = my_data_loader.sample(datatype = "test")
            self.test_data = att_cnn + att_flow + att_lstm + att_weather + short_cnn + short_flow + [short_lstm, ] + [short_weather, ]
            self.test_label = y

            self.feature_vec_len = short_lstm.shape[-1]
            self.nbhd_size = short_cnn[0].shape[1]
            self.nbhd_type = short_cnn[0].shape[-1]
            self.flow_type = short_flow[0].shape[-1]
            self.weather_type = short_weather.shape[-1]

            logger.info("[Agent] shapes of each region-level inputs for operation Evaluate Architecture")
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
        Loading model for evaluating
        """
        # create session and unlimit gpu
        K.clear_session()
        tf_config = tf.compat.v1.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        K.set_session(tf.compat.v1.Session(config = tf_config))

        self.logger.info("[Evaluate Architecture] loading network for evaluating...")
        if self.dataset_type == "station":
            # loading model form weight
            modeler = retrain_network.Station_Functional_Model(self.config)
            searched_choice_file_path = self.test_dir + self.config["file"]["model_path"] + "searched_choice_list.npy"
            searched_choice=np.load(open(searched_choice_file_path, "rb"), allow_pickle = True)
            self.model = modeler.func_model(nas_choice = searched_choice, feature_vec_len = self.feature_vec_len, \
                nbhd_size = self.nbhd_size, nbhd_type = self.nbhd_type, flow_type = self.flow_type, \
                weather_type = self.weather_type, optimizer = self.optimizer, loss = self.loss, metrics = [])
            
            retrained_model_weight_file_path = self.test_dir + self.config["file"]["model_path"] + "retrained_final_model_weight"
            self.model.load_weights(retrained_model_weight_file_path)

            # loading model directly from h5 file
            # retrained_model_file_path = self.test_dir + self.config["file"]["model_path"] + "retrained_final_model.h5"
            # self.model = tf.keras.models.load_model(retrained_model_file_path)

        elif self.dataset_type == "region":
            # loading model form weight
            modeler = retrain_network.Region_Functional_Model(self.config)
            searched_choice_file_path = self.test_dir + self.config["file"]["model_path"] + "searched_choice_list.npy"
            searched_choice=np.load(open(searched_choice_file_path, "rb"), allow_pickle = True)
            self.model = modeler.func_model(nas_choice = searched_choice, feature_vec_len = self.feature_vec_len, \
                nbhd_size = self.nbhd_size, nbhd_type = self.nbhd_type, flow_type = self.flow_type, \
                weather_type = self.weather_type, optimizer = self.optimizer, loss = self.loss, metrics = [])
            
            retrained_model_weight_file_path = self.test_dir + self.config["file"]["model_path"] + "retrained_final_model_weight"
            self.model.load_weights(retrained_model_weight_file_path)

            # loading model directly from h5 file
            # retrained_model_file_path = self.test_dir + self.config["file"]["model_path"] + "retrained_final_model.h5"
            # self.model = tf.keras.models.load_model(retrained_model_file_path)
        self.logger.info("[Evaluate Architecture] loading retained network complete")

    def evaluate_architecture(self):
        """
        Evaluate retrained architecture
        criterion: RMSE & MAPE
            1. without denormalized
            2. with denormalized
            3. delete model and clear session
        """
        self.logger.info("[Evaluate Architecture Without Denormalzied] evaluating start...")
        test_pred = self.model.predict(x = self.test_data)
        self.logger.info("[Evaluate Architecture Without Denormalzied] evaluating threshold: {0}".format(self.threshold))
        total_loss_rmse, total_loss_mape = eval_together(self.test_label, test_pred, self.threshold)
        (prmse, pmape), (drmse, dmape) = eval_lstm(self.test_label, test_pred, self.threshold)
        self.logger.info("[Evaluate Architecture Without Denormalize] pickup rmse = {0}, pickup mape = {1}%\n\
            dropoff rmse = {2}, dropoff mape = {3}%".format(prmse, pmape * 100, drmse, dmape * 100))
        self.logger.info("[Evaluate Architecture Without Denormalize] total_rmse = {0}, total_mape = {1}".format(total_loss_rmse, total_loss_mape * 100))
        
        self.logger.info("[Evaluate Architecture With Denormalized] evaluating start...")
        test_pred *= self.label_max
        test_label = self.test_label * self.label_max
        self.logger.info("[Evaluate Architecture] evaluating threshold: {0}".format(self.threshold_denormalize))
        total_loss_rmse, total_loss_mape = eval_together(test_label, test_pred, self.threshold_denormalize)
        (prmse, pmape), (drmse, dmape) = eval_lstm(test_label, test_pred, self.threshold_denormalize)
        self.logger.info("[Evaluate Architecture With Denormalize] pickup rmse = {0}, pickup mape = {1}%\n\
            dropoff rmse = {2}, dropoff mape = {3}%".format(prmse, pmape * 100, drmse, dmape * 100))
        self.logger.info("[Evaluate Architecture With Denormalize] total_rmse = {0}, total_mape = {1}".format(total_loss_rmse, total_loss_mape * 100))

        self.logger.info("[Evaluate Architecture] evaluate retrained model complete")
        del self.model
        K.clear_session()

    def process(self, test_dir):
        self.set_logger(test_dir)
        self.load_model()
        self.evaluate_architecture()