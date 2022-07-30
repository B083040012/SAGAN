import keras, logging, time, supernet
import tensorflow as tf
from file_loader import File_Loader
from tensorflow.python.keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, ProgbarLogger

# custom for early stop
# for more details: https://keras.io/api/callbacks/early_stopping/
class CustomStopper(keras.callbacks.EarlyStopping):
    # add argument for starting epoch
    def __init__(self, monitor = 'val_loss', min_delta = 0.000005, patience = 10, verbose = 1, mode = 'min', start_epoch = 40):
        super().__init__(monitor = monitor, min_delta = min_delta, patience = patience, verbose = verbose, mode = mode)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs = None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)

class Supernet_Training():
    def __init__(self, dataset_type, config):

        """
        Constructor of Supernet_Training
            1. initiliize parameters
        """
        # initialize parameters
        self.dataset_type = dataset_type
        self.config = config

        self.batch_size = config["training"]["supernet_training"]["batch_size"]
        self.validation_split = config["training"]["supernet_training"]["validation_split"]
        self.max_epochs = config["training"]["supernet_training"]["max_epochs"]
        self.optimizer = config["training"]["supernet_training"]["optimizer"]
        self.loss = config["training"]["supernet_training"]["loss"]

    def set_logger(self, test_dir):
        """
        Initialize logger
        """
        formatter = logging.Formatter('%(asctime)s %(message)s')
        file_handler = logging.FileHandler(test_dir + self.config["file"]["log_path"] + "Supernet_training.log")
        stream_handler = logging.StreamHandler()
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger = logging.getLogger('supernet_training')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        self.logger = logging.getLogger('supernet_training')
        self.test_dir = test_dir

    def load_data(self, logger):
        """
        Load dataset from agent
            * if ram cannot handle, maybe need to load from supernet_training object
            * can be loaded from the training_model method
        """
        my_data_loader = File_Loader(self.dataset_type, self.config)
        if self.dataset_type == "station":
            att_cnn, att_flow, att_lstm, att_weather, short_cnn, short_flow, short_lstm, short_weather, short_poi, y = my_data_loader.sample(datatype = "train")
            self.train_data = [att_cnn, att_flow, att_lstm, att_weather, short_cnn, short_flow, short_lstm, short_weather, short_poi]
            self.train_label = y

            logger.info("[Agent] shapes of each station-level inputs for operation Supernet_Training")
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
            att_cnn, att_flow, att_lstm, att_weather, short_cnn, short_flow, short_lstm, short_weather, y = my_data_loader.sample(datatype = "train")
            self.train_data = [att_cnn, att_flow, att_lstm, att_weather, short_cnn, short_flow, short_lstm, short_weather]
            self.train_label = y

            logger.info("[Agent] shapes of each region-level inputs for operation Supernet_Training")
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
        Load Supernet
            * run functions eagerly: if disables it,
                    the supernet will not choose randomly in each batch when training
        """
        # create session, unlimit gpu and run eagerly
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        K.set_session(tf.compat.v1.Session(config=config))
        tf.config.run_functions_eagerly(True)
        # tf.data.experimental.enable.debug_mode()

        self.logger.info("[Supernet Training] loading supernet...")
        if self.dataset_type == 'station':
            self.model = supernet.Station_Supernet_Subclass_Model(self.config)
        elif self.dataset_type == 'region':
            self.model = supernet.Region_Supernet_Subclass_Model(self.config)
        self.logger.info("[Supernet Training] supernet loading complete")
        self.model.compile(optimizer = self.optimizer, loss = self.loss, metrics = [])
        self.logger.info("[Supernet Training] supernet compile complete")

    def train_supernet(self):
        """
        Training supernet & record training time
            1. set checkpoint for saving model of best validation loss
            2. training & record time
            3. saving final supernet
            4. delete model and clear session
        """
        # define checkpoint for best supernet weight
        cpt_filepath=self.test_dir + self.config["file"]["model_path"] + 'best_supernet_weight'
        checkpoint = ModelCheckpoint(cpt_filepath, monitor='val_loss', verbose=1, \
            save_best_only=True, save_weights_only=True, mode='auto')

        # define csv logger
        csv_logger = CSVLogger(self.test_dir + self.config["file"]["log_path"] + 'supernet_training_step.csv', append=True, separator=';')
        
        self.logger.info("[Supernet Training] start training supernet...")
        start_time = time.time()
        self.model.fit(x = self.train_data, y = self.train_label, batch_size = self.batch_size, validation_split = self.validation_split, \
            epochs = self.max_epochs, callbacks = [checkpoint, csv_logger])
        end_time = time.time()
        self.logger.info("[Supernet Training] supernet training complete")
        self.logger.info("[Supernet Training] total training time: {0} min".format((end_time-start_time) / 60))
        self.model.save_weights(self.test_dir + self.config["file"]["model_path"] + "final_supernet_weight")
        self.logger.info("[Supernet Training] final supernet weight saved")

        # delete model and clear session
        del self.model
        K.clear_session()

    def process(self, test_dir):
        self.set_logger(test_dir)
        self.load_model()
        self.train_supernet()