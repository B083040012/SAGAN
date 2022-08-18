import logging, time, retrain_network
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from file_loader import File_Loader
from criterion import eval_together, eval_lstm

try:
    import keras
    from keras.callbacks import ModelCheckpoint, CSVLogger
except:
    from tensorflow import keras
    from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

def get_model_memory_usage(batch_size, model):
    import numpy as np
    # try:
    #     from keras import backend as K
    # except:
    #     from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes

# custom for early stop
# for more details: https://keras.io/api/callbacks/early_stopping/
class CustomStopper(keras.callbacks.EarlyStopping):
    # add argument for starting epoch
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', start_epoch=40):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)

class Architecture_Retrain():
    def __init__(self, dataset_type, config):
        """
        Initialize
            1. initialize parameters
        """
        self.dataset_type = dataset_type
        self.config = config
        self.batch_size = config["training"]["architecture_retrain"]["batch_size"]
        self.max_epochs = config["training"]["architecture_retrain"]["max_epochs"]
        self.validation_split = config["training"]["architecture_retrain"]["validation_split"]
        self.optimizer = config["training"]["architecture_retrain"]["optimizer"]
        self.loss = config["training"]["architecture_retrain"]["loss"]

    def set_logger(self, test_dir):
        """
        Initialize logger and test directory
        """
        formatter = logging.Formatter('%(asctime)s %(message)s')
        file_handler = logging.FileHandler(test_dir + self.config["file"]["log_path"] + "Architecture_retraining.log")
        stream_handler = logging.StreamHandler()
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger = logging.getLogger('architecture_retraining' + test_dir)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        self.logger = logging.getLogger('architecture_retraining' + test_dir)
        self.test_dir = test_dir

    def load_data(self, logger):
        """
        Loading dataset from agent
        """
        my_data_loader = File_Loader(self.dataset_type, self.config)
        if self.dataset_type == "station":
            att_cnn, att_flow, att_lstm, att_weather, short_cnn, short_flow, short_lstm, short_weather, y = my_data_loader.sample(datatype = "train")
            self.test_data = att_cnn + att_flow + att_lstm + att_weather + short_cnn + short_flow + [short_lstm, ] + [short_weather, ]
            self.test_label = y

            self.feature_vec_len = short_lstm.shape[-1]
            self.nbhd_size = short_cnn[0].shape[1]
            self.nbhd_type = short_cnn[0].shape[-1]
            self.flow_type = short_flow[0].shape[-1]
            self.weather_type = short_weather.shape[-1]

            logger.info("[Agent] shapes of each station-level inputs for operation Architecture Retrain")
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
            att_cnn, att_flow, att_lstm, att_weather, short_cnn, short_flow, short_lstm, short_weather, y = my_data_loader.sample(datatype = "train")
            self.test_data = att_cnn + att_flow + att_lstm + att_weather + short_cnn + short_flow + [short_lstm, ] + [short_weather, ]
            self.test_label = y

            self.feature_vec_len = short_lstm.shape[-1]
            self.nbhd_size = short_cnn[0].shape[1]
            self.nbhd_type = short_cnn[0].shape[-1]
            self.flow_type = short_flow[0].shape[-1]
            self.weather_type = short_weather.shape[-1]

            logger.info("[Agent] shapes of each region-level inputs for operation Architecture Retrain")
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
        Load network for retraining
            * ignore checkpoint due to the error message on tf 1.x
            1. setting tensorflow flags
            2. loading model
        """
        # create session and unlimit gpu
        K.clear_session()
        tf_config = tf.compat.v1.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        K.set_session(tf.compat.v1.Session(config = tf_config))

        self.logger.info("[Architecture Retrain] loading network for retraining...")
        if self.dataset_type == "station":
            modeler = retrain_network.Station_Functional_Model(self.config)
            searched_choice_file_path = self.test_dir + self.config["file"]["model_path"] + "searched_choice_list.npy"
            searched_choice=np.load(open(searched_choice_file_path, "rb"), allow_pickle = True)
            self.model = modeler.func_model(nas_choice = searched_choice, feature_vec_len = self.feature_vec_len, \
                nbhd_size = self.nbhd_size, nbhd_type = self.nbhd_type, flow_type = self.flow_type, \
                weather_type = self.weather_type, optimizer = self.optimizer, loss = self.loss, metrics = [])
        elif self.dataset_type == "region":
            modeler = retrain_network.Region_Functional_Model(self.config)
            searched_choice_file_path = self.test_dir + self.config["file"]["model_path"] + "searched_choice_list.npy"
            searched_choice=np.load(open(searched_choice_file_path, "rb"), allow_pickle = True)
            self.model = modeler.func_model(nas_choice = searched_choice, feature_vec_len = self.feature_vec_len, \
                nbhd_size = self.nbhd_size, nbhd_type = self.nbhd_type, flow_type = self.flow_type, \
                weather_type = self.weather_type, optimizer = self.optimizer, loss = self.loss, metrics = [])
        self.logger.info("[Architecture Retrain] loading network complete")

    def retrain(self):
        """
        Retrain process
            1. define callbacks
            2. fit the model and record time
            3. saved the final weight and model
            4. delete model and clear session
        """
        # define early stop
        early_stop = CustomStopper(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', start_epoch=40)

        # define csv logger
        csv_logger = CSVLogger(self.test_dir + self.config["file"]["log_path"] + 'architecture_retraining_step.csv', append=True, separator=';')

        self.logger.info("[Architecture Retrain] start retraining architecture...")
        start_time = time.time()
        self.model.fit(x = self.test_data, y = self.test_label, batch_size = self.batch_size, validation_split = self.validation_split, \
            epochs = self.max_epochs, callbacks = [early_stop, csv_logger])
        end_time = time.time()
        self.logger.info("[Architecture Retrain] architecture retrain complete")
        self.logger.info("[Architecture Retrain] total training time: {0} min".format((end_time - start_time) / 60))
        
        # save model and layer weights
        self.model.save_weights(self.test_dir + self.config["file"]["model_path"] + "retrained_final_model_weight")
        self.logger.info("[Architecture Retrain] retrained model weight saved")
        self.model.save(self.test_dir + self.config["file"]["model_path"] + "retrained_final_model.h5")
        self.logger.info("[Architecture Retrain] retrained model saved")

        self.logger.info("model summary in Architecture Retrain")
        self.model.summary(print_fn=self.logger.info)
        self.logger.info("memory_usage of model: {0}".format(get_model_memory_usage(64, self.model)))

        # delete model and clear session
        del self.model
        K.clear_session()

    def process(self, test_dir):
        self.set_logger(test_dir)
        self.load_model()
        self.retrain()