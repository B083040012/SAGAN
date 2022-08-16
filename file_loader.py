from cProfile import label
import numpy as np
import random

class File_Loader():
    def __init__(self, dataset_type, config, limit_dataset = False):
        """
        Initilalize parameters
        """
        self.dataset_type = dataset_type
        self.config = config
        self.limit_dataset = limit_dataset
        self.att_lstm_num = config["dataset"][self.dataset_type]["att_lstm_num"]
        self.long_term_lstm_seq_len = config["dataset"][self.dataset_type]["long_term_lstm_seq_len"]
        self.short_term_lstm_seq_len = config["dataset"][self.dataset_type]["short_term_lstm_seq_len"]
        self.hist_feature_daynum = config["dataset"][self.dataset_type]["hist_feature_daynum"]
        self.last_feature_num = config["dataset"][self.dataset_type]["last_feature_num"]
        self.lstm_nbhd_size = config["dataset"][self.dataset_type]["lstm_nbhd_size"]
        self.timeslot_daynum = config["dataset"][self.dataset_type]["timeslot_daynum"]
        self.validation_ratio = config["dataset"][self.dataset_type]["validation_ratio"]
        self.pred_target = config["dataset"][self.dataset_type]["pred_target"]

        if dataset_type == 'region':
            self.cnn_nbhd_size = config["dataset"][dataset_type]["cnn_nbhd_size"]

    def load_train_station(self):
        """
        Load train data from station-level dataset
            shape of each original data:
                1. volume: timeslot * station * type (start / end from timeslot) * grid_x * grid_y
                2. flow: timeslot * station * type (in / outflow) * grid_x * grid_y
                3. single volume (volume for single station): timeslot * station * type (start / end)
                4. single flow (flow for single station): timeslot * station * type (start / end)
                * single flow will be loaded only if it's pred_target
                * datasets above will be normalized when loading
                4. weather: timeslot * type
                5. poi: station * type * grid_x * grid_y
        """
        self.volume_train = np.load(self.config["file"]["station"]["volume_train"]) / self.config["dataset"]["station"]["volume_train_max"]
        self.flow_train = np.load(self.config["file"]["station"]["flow_train"]) / self.config["dataset"]["station"]["flow_train_max"]
        self.single_volume_train = np.load(self.config["file"]["station"]["single_volume_train"]) / self.config["dataset"]["station"]["single_volume_train_max"]
        if self.pred_target == "flow":
            self.single_flow_train = np.load(self.config["file"]["station"]["single_flow_train"]) / self.config["dataset"]["station"]["single_flow_train_max"]
        self.weather_train = np.load(self.config["file"]["station"]["weather_train"])
        self.poi_data = np.load(self.config["file"]["station"]["poi_data"])
        if self.limit_dataset == False:
            self.start_date = self.config["dataset"]["station"]["start_date_train"]
            self.end_date = self.config["dataset"]["station"]["end_date_train"]
            self.start_hour = self.config["dataset"]["station"]["start_hour_train"]
            self.end_hour = self.config["dataset"]["station"]["end_hour_train"]
        else:
            print("limit_dataset")
            self.start_date = self.config["dataset"]["station"]["limit_start_date_train"]
            self.end_date = self.config["dataset"]["station"]["limit_end_date_train"]
            self.start_hour = self.config["dataset"]["station"]["limit_start_hour_train"]
            self.end_hour = self.config["dataset"]["station"]["limit_end_hour_train"]

    def load_test_station(self):
        """
        Load test data from station-level dataset
        """
        self.volume_test = np.load(self.config["file"]["station"]["volume_test"]) / self.config["dataset"]["station"]["volume_test_max"]
        self.flow_test = np.load(self.config["file"]["station"]["flow_test"]) / self.config["dataset"]["station"]["flow_test_max"]
        self.single_volume_test = np.load(self.config["file"]["station"]["single_volume_test"]) / self.config["dataset"]["station"]["single_volume_test_max"]
        if self.pred_target == "flow":
            self.single_flow_test = np.load(self.config["file"]["station"]["single_flow_test"]) / self.config["dataset"]["station"]["single_flow_test_max"]
        self.weather_test = np.load(self.config["file"]["station"]["weather_test"])
        self.poi_data = np.load(self.config["file"]["station"]["poi_data"])
        if self.limit_dataset == False:
            self.start_date = self.config["dataset"]["station"]["start_date_test"]
            self.end_date = self.config["dataset"]["station"]["end_date_test"]
            self.start_hour = self.config["dataset"]["station"]["start_hour_test"]
            self.end_hour = self.config["dataset"]["station"]["end_hour_test"]
        else:
            print("limit_dataset")
            self.start_date = self.config["dataset"]["station"]["limit_start_date_test"]
            self.end_date = self.config["dataset"]["station"]["limit_end_date_test"]
            self.start_hour = self.config["dataset"]["station"]["limit_start_hour_test"]
            self.end_hour = self.config["dataset"]["station"]["limit_end_hour_test"]

    def load_train_region(self):
        """
        Load train data from region-level dataset
            shape of each original data:
                1. volume: timeslot * station * grid_x * grid_y * type (start / end from region)
                2. flow: timeslot * station * type (in / outflow) * grid_x * grid_y (start region) * grid_x * grid_y (end region)
                * datasets above will be normalized when loading
                3. weather: timeslot * type
        """
        self.volume_train = np.load(self.config["file"]["region"]["volume_train"]) / self.config["dataset"]["region"]["volume_train_max"]
        self.flow_train = np.load(self.config["file"]["region"]["flow_train"]) / self.config["dataset"]["region"]["flow_train_max"]
        if self.pred_target == "flow":
            print("invalid target for region-level dataset !!")
            raise Exception
        self.weather_train = np.load(self.config["file"]["region"]["weather_train"])
        if self.limit_dataset == False:
            self.start_date = self.config["dataset"]["region"]["start_date_train"]
            self.end_date = self.config["dataset"]["region"]["end_date_train"]
            self.start_hour = self.config["dataset"]["region"]["start_hour_train"]
            self.end_hour = self.config["dataset"]["region"]["end_hour_train"]
        else:
            print("limit_dataset")
            self.start_date = self.config["dataset"]["region"]["limit_start_date_train"]
            self.end_date = self.config["dataset"]["region"]["limit_end_date_train"]
            self.start_hour = self.config["dataset"]["region"]["limit_start_hour_train"]
            self.end_hour = self.config["dataset"]["region"]["limit_end_hour_train"]

    def load_test_region(self):
        """
        Load test data from region-level dataset
        """
        self.volume_test = np.load(self.config["file"]["region"]["volume_test"]) / self.config["dataset"]["region"]["volume_test_max"]
        self.flow_test = np.load(self.config["file"]["region"]["flow_test"]) / self.config["dataset"]["region"]["flow_test_max"]
        if self.pred_target == "flow":
            print("invalid target for region-level dataset !!")
            raise Exception
        self.weather_test = np.load(self.config["file"]["region"]["weather_test"])
        if self.limit_dataset == False:
            self.start_date = self.config["dataset"]["region"]["start_date_test"]
            self.end_date = self.config["dataset"]["region"]["end_date_test"]
            self.start_hour = self.config["dataset"]["region"]["start_hour_test"]
            self.end_hour = self.config["dataset"]["region"]["end_hour_test"]
        else:
            print("limit_dataset")
            self.start_date = self.config["dataset"]["region"]["limit_start_date_test"]
            self.end_date = self.config["dataset"]["region"]["limit_end_date_test"]
            self.start_hour = self.config["dataset"]["region"]["limit_start_hour_test"]
            self.end_hour = self.config["dataset"]["region"]["limit_end_hour_test"]
    
    def sample_station(self, datatype):
        """
        Sampling data from station dataset
            1. loading data depends on datatype
            2. initialize the inputs for sampling
            3. determine the sampling time range
                (inputs data size = time range * station number)
            4. start sampling
            5. encapsulate numpy array
        """

        if self.long_term_lstm_seq_len % 2 !=1:
            print("long_term_lstm_seq_len must be odd !!")
            raise Exception
        
        # loading data depends on datatype
        if datatype == "train" or datatype == "validation":
            self.load_train_station()
            volume_data = self.volume_train
            flow_data = self.flow_train
            weather_data = self.weather_train
            poi_data = self.poi_data
            single_volume_data = self.single_volume_train
            if self.pred_target == "flow":
                single_flow_data = self.single_flow_train
            print("train single volume shape: {0}".format(single_volume_data.shape))
        elif datatype == "test":
            self.load_test_station()
            volume_data = self.volume_test
            flow_data = self.flow_test
            weather_data = self.weather_test
            poi_data = self.poi_data
            single_volume_data = self.single_volume_test
            if self.pred_target == "flow":
                single_flow_data = self.single_flow_test
            print("test single volume shape: {0}".format(single_volume_data.shape))
        else:
            print("Please select 'train', 'validation', or 'test'")
            raise Exception

        # initialize short term features & label
        short_nbhd_features = []
        short_flow_features = []
        short_weather_features = []
        short_lstm_features = []
        poi_features = []
        labels = []
        for ts in range(self.short_term_lstm_seq_len):
            short_nbhd_features.append([])
            short_flow_features.append([])

        # initialize long term features
        att_nbhd_features = []
        att_flow_features = []
        att_weather_features = []
        att_lstm_features = []
        for att in range(self.att_lstm_num):
            att_nbhd_features.append([])
            att_flow_features.append([])
            att_weather_features.append([])
            att_lstm_features.append([])
            for ts in range(self.long_term_lstm_seq_len):
                att_nbhd_features[att].append([])
                att_flow_features[att].append([])

        # calculate start and end index of lstm nbhd feature depends on pre-determined size
        lstm_nbhd_center_grid = (volume_data.shape[3] - 1) / 2
        lstm_nbhd_start_size = int(lstm_nbhd_center_grid - ((self.lstm_nbhd_size - 1) / 2))
        lstm_nbhd_end_size = int(lstm_nbhd_center_grid - ((self.lstm_nbhd_size - 1) / 2) + 1)
        
        # initialize sampling time range
        time_start = (self.hist_feature_daynum + self.att_lstm_num) * self.timeslot_daynum + self.long_term_lstm_seq_len

        time_range_list = sorted([hour + 48 * date for hour in range(self.start_hour, self.end_hour) \
            for date in range(self.start_date, self.end_date) if hour + 48 * date >= time_start])
        if datatype == 'validation':
            time_range_list = sorted(random.sample(time_range_list, int(len(time_range_list) * self.validation_ratio)))

        # sampling
        print("time interval length: {0}".format(len(time_range_list)))
        for index, t in enumerate(time_range_list):
            if index % 100 == 0:
                print("Now sampling at {0}th timeslots.".format(index))
            for station_idx in range(0, volume_data.shape[1]):
                """
                poi features
                    size: 5 (grid_x) * 5 (gird_y) * 10 (poi_type) 
                    # since the poi features have identical timeslot for each station, 
                      short and long term all have same poi_features
                """
                poi_features.append(poi_data[station_idx, :, :, 0:6])

                """
                short-term features
                including:
                    1. nbhd_features
                    2. flow_features
                    3. lstm_features
                    4. weather_features
                    5. poi_features
                        # notice that poi features has no time interval yet (same time for each station)
                """
                short_term_lstm_samples = []
                short_term_weather_samples = []
                for seqn in range(self.short_term_lstm_seq_len):
                    # real_t from (t - short_term_lstm_seq_len) to (t-1)
                    real_t = t - (self.short_term_lstm_seq_len - seqn)

                    """
                    short-term nbhd features
                        size: 7 (nbhd_size) * 7 (nbhd_size) * 2 (start / end)
                    """
                    nbhd_feature = np.zeros((volume_data.shape[3], volume_data.shape[4], volume_data.shape[2]))
                    nbhd_feature[:, :, 0] = volume_data[real_t, station_idx, 0, :, :]
                    nbhd_feature[:, :, 1] = volume_data[real_t, station_idx, 1, :, :]
                    short_nbhd_features[seqn].append(nbhd_feature)

                    """
                    short-term flow features
                        size: 7 (nbhd_size) * 7 (nbhd_size) * 4 (type)
                        including:
                            1. curr outflow
                            2. curr inflow
                            3. last outflow
                            4. last inflow
                            * curr: start and end in same timeslot
                            * last: start and end in different timeslot (not done yet)
                    """
                    flow_feature_curr_out = flow_data[real_t, station_idx, 0, :, :]
                    flow_feature_curr_in = flow_data[real_t, station_idx, 1, :, :]
                    flow_feature_last_out_to_curr = flow_data[real_t - 1, station_idx, 0, :, :]
                    flow_feature_curr_in_from_last = flow_data[real_t - 1, station_idx, 1, :, :]

                    flow_feature = np.zeros(flow_feature_curr_in.shape+(4,))

                    flow_feature[:, :, 0] = flow_feature_curr_out
                    flow_feature[:, :, 1] = flow_feature_curr_in
                    flow_feature[:, :, 2] = flow_feature_last_out_to_curr
                    flow_feature[:, :, 3] = flow_feature_curr_in_from_last

                    short_flow_features[seqn].append(flow_feature)

                    # """
                    # short-term poi features
                    #     size: 10*10*10
                    #     including:
                    #         10 types of poi
                    # """
                    # poi_features.append(poi_data[station_idx])

                    """
                    short-term lstm features
                        size: sum of the size below (after flatten)
                        including:
                            1. volume feature: lstm_nbhd_size * lstm_nbhd_size * volume_type (start / end)
                            2. last feature: last_feature_num * volume_type (start / end)
                            3. hist feature: hist_feature_num * volume_type (start / end)
                    
                    short-term weather features
                        size: do not have fixed size (depends on the time of dataset)
                        including:
                            1. temparature
                            2. dew point
                            3. humidity
                            4. wind speed
                            5. wind gust
                            6. pressure
                            7. precip
                            8 ~ shape[-1]. one hot encoding for weather type
                    """
                    # volume feature
                    # nbhd_feature = np.zeros((volume_data.shape[3], volume_data.shape[4], volume_data.shape[2]))
                    nbhd_feature = np.zeros((self.lstm_nbhd_size, self.lstm_nbhd_size, volume_data.shape[2]))
                    nbhd_feature[:, :, 0] = volume_data[real_t, station_idx, 0, lstm_nbhd_start_size:lstm_nbhd_end_size, lstm_nbhd_start_size:lstm_nbhd_end_size]
                    nbhd_feature[:, :, 1] = volume_data[real_t, station_idx, 1, lstm_nbhd_start_size:lstm_nbhd_end_size, lstm_nbhd_start_size:lstm_nbhd_end_size]
                    nbhd_feature = nbhd_feature.flatten()
                    
                    # last feature
                    # last_feature = np.zeros((volume_data.shape[3], volume_data.shape[4], volume_data.shape[2]))
                    # last_feature = np.zeros((self.lstm_nbhd_size, self.lstm_nbhd_size, volume_data.shape[2]))
                    # last_feature[:, :, 0] = volume_data[real_t - last_feature_num, station_idx, 0, lstm_nbhd_start_size:lstm_nbhd_end_size, lstm_nbhd_start_size:lstm_nbhd_end_size]
                    # last_feature[:, :, 1] = volume_data[real_t - last_feature_num, station_idx, 1, lstm_nbhd_start_size:lstm_nbhd_end_size, lstm_nbhd_start_size:lstm_nbhd_end_size]
                    last_feature = np.zeros((self.last_feature_num, volume_data.shape[2]))
                    last_feature[:, 0] = single_volume_data[real_t - self.last_feature_num: real_t, station_idx, 0]
                    last_feature[:, 1] = single_volume_data[real_t - self.last_feature_num: real_t, station_idx, 1]
                    last_feature = last_feature.flatten()

                    # hist feature
                    # hist_feature = np.zeros((7, volume_data.shape[3], volume_data.shape[4], volume_data.shape[2]))
                    # hist_feature = np.zeros((7, self.lstm_nbhd_size, self.lstm_nbhd_size, volume_data.shape[2]))
                    # hist_feature[:, :, :, 0] = volume_data[real_t - hist_feature_daynum*self.timeslot_daynum: real_t: self.timeslot_daynum,\
                    #     station_idx, 0, lstm_nbhd_start_size:lstm_nbhd_end_size, lstm_nbhd_start_size:lstm_nbhd_end_size]
                    # hist_feature[:, :, :, 1] = volume_data[real_t - hist_feature_daynum*self.timeslot_daynum: real_t: self.timeslot_daynum,\
                    #     station_idx, 1, lstm_nbhd_start_size:lstm_nbhd_end_size, lstm_nbhd_start_size:lstm_nbhd_end_size]
                    hist_feature = np.zeros((self.hist_feature_daynum, volume_data.shape[2]))
                    hist_feature[:, 0] = single_volume_data[real_t - self.hist_feature_daynum*self.timeslot_daynum: real_t: self.timeslot_daynum, station_idx, 0]
                    hist_feature[:, 1] = single_volume_data[real_t - self.hist_feature_daynum*self.timeslot_daynum: real_t: self.timeslot_daynum, station_idx, 1]
                    hist_feature = hist_feature.flatten()

                    feature_vec = np.concatenate((hist_feature, last_feature))
                    feature_vec = np.concatenate((feature_vec, nbhd_feature))

                    short_term_lstm_samples.append(feature_vec)
                    short_term_weather_samples.append(weather_data[real_t])

                short_lstm_features.append(np.array(short_term_lstm_samples))
                short_weather_features.append(np.array(short_term_weather_samples))

                """
                long-term features
                including:
                    1. att_nbhd_features
                    2. att_flow_features
                    3. att_lstm_features
                    4. att_weather_features
                    5. poi_features (same as short term poi features)
                """
                for att_lstm_cnt in range(self.att_lstm_num):

                    long_term_lstm_samples = []
                    long_term_weather_samples = []

                    """
                    range of att_t:
                    for target timeslot t,
                        1. att_t first reach (att_lstm_num - att_lstm_cnt) days before (same time)
                            --> t - (att_lstm_num - att_lstm_cnt) * self.timeslot_daynum
                        2. for each day, sample the time from 
                            (long_term_lstm_seq_len / 2) before target time ~ (long_term_lstm_seq_len / 2) after target time
                    for example, if (att_lstm_num, long_term_lstm_seq_len) = (3, 3), target time = (day 4, timeslot 9), then att_t samples
                            day 1: timeslot 8 ~ 10
                            day 2: timeslot 8 ~ 10
                            day 3: timeslot 8 ~ 10
                    """
                    att_t = int(t - (self.att_lstm_num - att_lstm_cnt) * self.timeslot_daynum + (self.long_term_lstm_seq_len - 1) / 2 + 1)

                    for seqn in range(self.long_term_lstm_seq_len):
                        real_t = att_t - (self.long_term_lstm_seq_len - seqn)

                        """
                        long term nbhd features
                            size: 7 (grid_x) * 7 (grid_y) * 2 (start / end)
                        """
                        nbhd_feature = np.zeros((volume_data.shape[3], volume_data.shape[4], volume_data.shape[2]))
                        nbhd_feature[:, :, 0] = volume_data[real_t, station_idx, 0, :, :]
                        nbhd_feature[:, :, 1] = volume_data[real_t, station_idx, 1, :, :]
                        att_nbhd_features[att_lstm_cnt][seqn].append(nbhd_feature)

                        """
                        long-term flow features
                            size: 7 (grid_x) * 7 (grid_y) * 4 (type)
                            including:
                                1. curr outflow
                                2. curr inflow
                                3. last outflow
                                4. last inflow
                            * curr: start and end in same timeslot
                            * last: start and end in different timeslot (not done yet)
                        """
                        flow_feature_curr_out = flow_data[real_t, station_idx, 0, :, :]
                        flow_feature_curr_in = flow_data[real_t, station_idx, 1, :, :]
                        flow_feature_last_out_to_curr = flow_data[real_t, station_idx, 0, :, :]
                        flow_feature_curr_in_from_last = flow_data[real_t, station_idx, 1, :, :]

                        flow_feature = np.zeros(flow_feature_curr_in.shape+(4,))

                        flow_feature[:, :, 0] = flow_feature_curr_out
                        flow_feature[:, :, 1] = flow_feature_curr_in
                        flow_feature[:, :, 2] = flow_feature_last_out_to_curr
                        flow_feature[:, :, 3] = flow_feature_curr_in_from_last

                        att_flow_features[att_lstm_cnt][seqn].append(flow_feature)

                        # """
                        # long-term poi features
                        #     size: 10*10*10
                        #     including:
                        #         10 types of poi
                        # """
                        # poi_att_features[att_lstm_cnt].append(poi_data[station_idx])

                        """
                        long-term lstm features
                            size: sum of the size below (after flatten)
                            including:
                                1. volume feature: lstm_nbhd_size * lstm_nbhd_size * volume_type (start / end)
                                2. last feature: last_feature_num * volume_type (start / end)
                                3. hist feature: hist_feature_num * volume_type (start / end)
                        
                        long-term weather features
                            size: do not have fixed size (depends on the time of dataset)
                            including:
                                1. temparature
                                2. dew point
                                3. humidity
                                4. wind speed
                                5. wind gust
                                6. pressure
                                7. precip
                                8 ~ shape[-1]. one hot encoding for weather type
                        """
                        # volume feature
                        # nbhd_feature = np.zeros((volume_data.shape[3], volume_data.shape[4], volume_data.shape[2]))
                        nbhd_feature = np.zeros((self.lstm_nbhd_size, self.lstm_nbhd_size, volume_data.shape[2]))
                        nbhd_feature[:, :, 0] = volume_data[real_t, station_idx, 0, lstm_nbhd_start_size:lstm_nbhd_end_size, lstm_nbhd_start_size:lstm_nbhd_end_size]
                        nbhd_feature[:, :, 1] = volume_data[real_t, station_idx, 1, lstm_nbhd_start_size:lstm_nbhd_end_size, lstm_nbhd_start_size:lstm_nbhd_end_size]
                        nbhd_feature = nbhd_feature.flatten()

                        # last feature
                        # last_feature = np.zeros((volume_data.shape[3], volume_data.shape[4], volume_data.shape[2]))
                        # last_feature = np.zeros((self.lstm_nbhd_size, self.lstm_nbhd_size, volume_data.shape[2]))
                        # last_feature[:, :, 0] = volume_data[real_t - self.last_feature_num, station_idx, 0, lstm_nbhd_start_size:lstm_nbhd_end_size, lstm_nbhd_start_size:lstm_nbhd_end_size]
                        # last_feature[:, :, 1] = volume_data[real_t - self.last_feature_num, station_idx, 1, lstm_nbhd_start_size:lstm_nbhd_end_size, lstm_nbhd_start_size:lstm_nbhd_end_size]
                        # last_feature = last_feature.flatten()
                        last_feature = np.zeros((self.last_feature_num, volume_data.shape[2]))
                        last_feature[:, 0] = single_volume_data[real_t - self.last_feature_num: real_t, station_idx, 0]
                        last_feature[:, 1] = single_volume_data[real_t - self.last_feature_num: real_t, station_idx, 1]
                        last_feature = last_feature.flatten()

                        # hist feature
                        # hist_feature = np.zeros((7, volume_data.shape[3], volume_data.shape[4], volume_data.shape[2]))
                        # hist_feature = np.zeros((7, self.lstm_nbhd_size, self.lstm_nbhd_size, volume_data.shape[2]))
                        # hist_feature[:, :, :, 0] = volume_data[real_t - self.hist_feature_daynum*self.timeslot_daynum: real_t: self.timeslot_daynum,\
                        #     station_idx, 0, lstm_nbhd_start_size:lstm_nbhd_end_size, lstm_nbhd_start_size:lstm_nbhd_end_size]
                        # hist_feature[:, :, :, 1] = volume_data[real_t - self.hist_feature_daynum*self.timeslot_daynum: real_t: self.timeslot_daynum,\
                        #     station_idx, 1, lstm_nbhd_start_size:lstm_nbhd_end_size, lstm_nbhd_start_size:lstm_nbhd_end_size]
                        hist_feature = np.zeros((self.hist_feature_daynum, volume_data.shape[2]))
                        hist_feature[:, 0] = single_volume_data[real_t - self.hist_feature_daynum*self.timeslot_daynum: real_t: self.timeslot_daynum, station_idx, 0]
                        hist_feature[:, 1] = single_volume_data[real_t - self.hist_feature_daynum*self.timeslot_daynum: real_t: self.timeslot_daynum, station_idx, 1]
                        hist_feature = hist_feature.flatten()
                        hist_feature = hist_feature.flatten()

                        feature_vec = np.concatenate((hist_feature, last_feature))
                        feature_vec = np.concatenate((feature_vec, nbhd_feature))

                        long_term_lstm_samples.append(feature_vec)
                        long_term_weather_samples.append(weather_data[real_t])
                    att_lstm_features[att_lstm_cnt].append(np.array(long_term_lstm_samples))
                    att_weather_features[att_lstm_cnt].append(np.array(long_term_weather_samples))
                
                """
                label
                    size: 2
                    pred_target == "volume":
                        1. start volume
                        2. end volume
                    pred_target == "flow":
                        1. inflow
                        2. outflow
                """
                if self.pred_target == "volume":
                    labels.append(single_volume_data[t, station_idx, :].flatten())
                elif self.pred_target == "flow":
                    labels.append(single_flow_data[t, station_idx, :].flatten())

        for i in range(self.short_term_lstm_seq_len):
            short_nbhd_features[i] = np.array(short_nbhd_features[i])
            short_flow_features[i] = np.array(short_flow_features[i])
        short_lstm_features = np.array(short_lstm_features)
        weather_features = np.array(short_weather_features)
        poi_features = np.array(poi_features)
        
        output_nbhd_att_features = []
        output_flow_att_features = []
        # output_poi_att_features = []
        for i in range(self.att_lstm_num):
            att_lstm_features[i] = np.array(att_lstm_features[i])
            att_weather_features[i] = np.array(att_weather_features[i])
            for j in range(self.long_term_lstm_seq_len):
                att_nbhd_features[i][j] = np.array(att_nbhd_features[i][j])
                att_flow_features[i][j] = np.array(att_flow_features[i][j])
                # poi_att_features[i][j] = np.array(poi_att_features[i][j])
                output_nbhd_att_features.append(att_nbhd_features[i][j])
                output_flow_att_features.append(att_flow_features[i][j])
                # output_poi_att_features.append(poi_att_features[i][j])
        labels = np.array(labels)

        return output_nbhd_att_features, output_flow_att_features, att_lstm_features, att_weather_features,\
            short_nbhd_features, short_flow_features, short_lstm_features, weather_features, poi_features,\
            labels

    def sample_region(self, datatype):
        if self.long_term_lstm_seq_len % 2 != 1:
            print("Att-lstm seq_len must be odd!")
            raise Exception

        if datatype == "train" or datatype == "validation":
            self.load_train_region()
            data = self.volume_train
            flow_data = self.flow_train
            weather_data = self.weather_train
        elif datatype == "test":
            self.load_test_region()
            data = self.volume_test
            flow_data = self.flow_test
            weather_data = self.weather_test
        else:
            print("Please select 'train', 'validation', or 'test'")
            raise Exception


        cnn_att_features = []
        lstm_att_features = []
        flow_att_features = []
        weather_att_features = []
        for i in range(self.att_lstm_num):
            lstm_att_features.append([])
            weather_att_features.append([])
            cnn_att_features.append([])
            flow_att_features.append([])
            for j in range(self.long_term_lstm_seq_len):
                cnn_att_features[i].append([])
                flow_att_features[i].append([])

        cnn_features = []
        flow_features = []
        weather_features = []
        for i in range(self.short_term_lstm_seq_len):
            cnn_features.append([])
            flow_features.append([])
        short_term_lstm_features = []
        labels = []

        # initialize sampling time range
        time_start = (self.hist_feature_daynum + self.att_lstm_num) * self.timeslot_daynum + self.long_term_lstm_seq_len

        time_range_list = sorted([hour + 48 * date for hour in range(self.start_hour, self.end_hour) \
            for date in range(self.start_date, self.end_date) if hour + 48 * date >= time_start])
        if datatype == 'validation':
            time_range_list = sorted(random.sample(time_range_list, int(len(time_range_list) * self.validation_ratio)))
        
        volume_type = data.shape[-1]

        print("time interval length: {0}".format(len(time_range_list)))
        for index, t in enumerate(time_range_list):
            if index%100 == 0:
                print("Now sampling at {0}th timeslots.".format(index))
            for x in range(data.shape[1]):
                for y in range(data.shape[2]):
                    
                    #sample common (short-term) lstm
                    short_term_lstm_samples = []
                    short_term_weather_samples = []
                    for seqn in range(self.short_term_lstm_seq_len):
                        # real_t from (t - short_term_lstm_seq_len) to (t-1)
                        real_t = t - (self.short_term_lstm_seq_len - seqn)

                        #cnn features, zero_padding
                        cnn_feature = np.zeros((2*self.cnn_nbhd_size+1, 2*self.cnn_nbhd_size+1, volume_type))
                        #actual idx in data
                        for cnn_nbhd_x in range(x - self.cnn_nbhd_size, x + self.cnn_nbhd_size + 1):
                            for cnn_nbhd_y in range(y - self.cnn_nbhd_size, y + self.cnn_nbhd_size + 1):
                                #boundary check
                                if not (0 <= cnn_nbhd_x < data.shape[1] and 0 <= cnn_nbhd_y < data.shape[2]):
                                    continue
                                #get features
                                cnn_feature[cnn_nbhd_x - (x - self.cnn_nbhd_size), cnn_nbhd_y - (y - self.cnn_nbhd_size), :] = data[real_t, cnn_nbhd_x, cnn_nbhd_y, :]
                        cnn_features[seqn].append(cnn_feature)

                        #flow features, 4 type
                        flow_feature_curr_out = flow_data[0, real_t, x, y, :, :]
                        flow_feature_curr_in = flow_data[0, real_t, :, :, x, y]
                        flow_feature_last_out_to_curr = flow_data[1, real_t - 1, x, y, :, :]
                        #real_t - 1 is the time for in flow in longflow1
                        flow_feature_curr_in_from_last = flow_data[1, real_t - 1, :, :, x, y]

                        flow_feature = np.zeros(flow_feature_curr_in.shape+(4,))
                        
                        flow_feature[:, :, 0] = flow_feature_curr_out
                        flow_feature[:, :, 1] = flow_feature_curr_in
                        flow_feature[:, :, 2] = flow_feature_last_out_to_curr
                        flow_feature[:, :, 3] = flow_feature_curr_in_from_last
                        #calculate local flow, same shape cnn
                        local_flow_feature = np.zeros((2*self.cnn_nbhd_size+1, 2*self.cnn_nbhd_size+1, 4))
                        #actual idx in data
                        for cnn_nbhd_x in range(x - self.cnn_nbhd_size, x + self.cnn_nbhd_size + 1):
                            for cnn_nbhd_y in range(y - self.cnn_nbhd_size, y + self.cnn_nbhd_size + 1):
                                #boundary check
                                if not (0 <= cnn_nbhd_x < data.shape[1] and 0 <= cnn_nbhd_y < data.shape[2]):
                                    continue
                                #get features
                                local_flow_feature[cnn_nbhd_x - (x - self.cnn_nbhd_size), cnn_nbhd_y - (y - self.cnn_nbhd_size), :] = flow_feature[cnn_nbhd_x, cnn_nbhd_y, :]
                        flow_features[seqn].append(local_flow_feature)

                        #lstm features
                        # nbhd feature, zero_padding
                        nbhd_feature = np.zeros((2*self.lstm_nbhd_size+1, 2*self.lstm_nbhd_size+1, volume_type))
                        #actual idx in data
                        for nbhd_x in range(x - self.lstm_nbhd_size, x + self.lstm_nbhd_size + 1):
                            for nbhd_y in range(y - self.lstm_nbhd_size, y + self.lstm_nbhd_size + 1):
                                #boundary check
                                if not (0 <= nbhd_x < data.shape[1] and 0 <= nbhd_y < data.shape[2]):
                                    continue
                                #get features
                                nbhd_feature[nbhd_x - (x - self.lstm_nbhd_size), nbhd_y - (y - self.lstm_nbhd_size), :] = data[real_t, nbhd_x, nbhd_y, :]
                        nbhd_feature = nbhd_feature.flatten()

                        #last feature
                        last_feature = data[real_t - self.last_feature_num: real_t, x, y, :].flatten()

                        #hist feature
                        hist_feature = data[real_t - self.hist_feature_daynum*self.timeslot_daynum: real_t: self.timeslot_daynum, x, y, :].flatten()

                        feature_vec = np.concatenate((hist_feature, last_feature))
                        feature_vec = np.concatenate((feature_vec, nbhd_feature))

                        short_term_lstm_samples.append(feature_vec)
                        short_term_weather_samples.append(weather_data[real_t])
                    short_term_lstm_features.append(np.array(short_term_lstm_samples))
                    weather_features.append(np.array(short_term_weather_samples))

                    #sample att-lstms
                    for att_lstm_cnt in range(self.att_lstm_num):
                        
                        #sample lstm at att loc att_lstm_cnt
                        long_term_lstm_samples = []
                        long_term_weather_samples = []
                        # get time att_t, move forward for (att_lstm_num - att_lstm_cnt) day, then move back for ([long_term_lstm_seq_len / 2] + 1)
                        # notice that att_t-th timeslot will not be sampled in lstm
                        # e.g., **** (att_t - 3) **** (att_t - 2) (yesterday's t) **** (att_t - 1) **** (att_t) (this one will not be sampled)
                        # sample att-lstm with seq_len = 3
                        att_t = t - (self.att_lstm_num - att_lstm_cnt) * self.timeslot_daynum + (self.long_term_lstm_seq_len - 1) / 2 + 1
                        att_t = int(att_t)
                        #att-lstm seq len
                        for seqn in range(self.long_term_lstm_seq_len):
                            # real_t from (att_t - self.long_term_lstm_seq_len) to (att_t - 1)
                            real_t = att_t - (self.long_term_lstm_seq_len - seqn)

                            #cnn features, zero_padding
                            cnn_feature = np.zeros((2*self.cnn_nbhd_size+1, 2*self.cnn_nbhd_size+1, volume_type))
                            #actual idx in data
                            for cnn_nbhd_x in range(x - self.cnn_nbhd_size, x + self.cnn_nbhd_size + 1):
                                for cnn_nbhd_y in range(y - self.cnn_nbhd_size, y + self.cnn_nbhd_size + 1):
                                    #boundary check
                                    if not (0 <= cnn_nbhd_x < data.shape[1] and 0 <= cnn_nbhd_y < data.shape[2]):
                                        continue
                                    #get features
                                    # import ipdb; ipdb.set_trace()
                                    cnn_feature[cnn_nbhd_x - (x - self.cnn_nbhd_size), cnn_nbhd_y - (y - self.cnn_nbhd_size), :] = data[real_t, cnn_nbhd_x, cnn_nbhd_y, :]
                            cnn_att_features[att_lstm_cnt][seqn].append(cnn_feature)

                            #flow features, 4 type
                            flow_feature_curr_out = flow_data[0, real_t, x, y, :, :]
                            flow_feature_curr_in = flow_data[0, real_t, :, :, x, y]
                            flow_feature_last_out_to_curr = flow_data[1, real_t - 1, x, y, :, :]
                            #real_t - 1 is the time for in flow in longflow1
                            flow_feature_curr_in_from_last = flow_data[1, real_t - 1, :, :, x, y]

                            flow_feature = np.zeros(flow_feature_curr_in.shape+(4,))
                            
                            flow_feature[:, :, 0] = flow_feature_curr_out
                            flow_feature[:, :, 1] = flow_feature_curr_in
                            flow_feature[:, :, 2] = flow_feature_last_out_to_curr
                            flow_feature[:, :, 3] = flow_feature_curr_in_from_last
                            #calculate local flow, same shape cnn
                            local_flow_feature = np.zeros((2*self.cnn_nbhd_size+1, 2*self.cnn_nbhd_size+1, 4))
                            #actual idx in data
                            for cnn_nbhd_x in range(x - self.cnn_nbhd_size, x + self.cnn_nbhd_size + 1):
                                for cnn_nbhd_y in range(y - self.cnn_nbhd_size, y + self.cnn_nbhd_size + 1):
                                    #boundary check
                                    if not (0 <= cnn_nbhd_x < data.shape[1] and 0 <= cnn_nbhd_y < data.shape[2]):
                                        continue
                                    #get features
                                    local_flow_feature[cnn_nbhd_x - (x - self.cnn_nbhd_size), cnn_nbhd_y - (y - self.cnn_nbhd_size), :] = flow_feature[cnn_nbhd_x, cnn_nbhd_y, :]
                            flow_att_features[att_lstm_cnt][seqn].append(local_flow_feature)

                            #att-lstm features
                            # nbhd feature, zero_padding
                            nbhd_feature = np.zeros((2*self.lstm_nbhd_size+1, 2*self.lstm_nbhd_size+1, volume_type))
                            #actual idx in data
                            for nbhd_x in range(x - self.lstm_nbhd_size, x + self.lstm_nbhd_size + 1):
                                for nbhd_y in range(y - self.lstm_nbhd_size, y + self.lstm_nbhd_size + 1):
                                    #boundary check
                                    if not (0 <= nbhd_x < data.shape[1] and 0 <= nbhd_y < data.shape[2]):
                                        continue
                                    #get features
                                    nbhd_feature[nbhd_x - (x - self.lstm_nbhd_size), nbhd_y - (y - self.lstm_nbhd_size), :] = data[real_t, nbhd_x, nbhd_y, :]
                            nbhd_feature = nbhd_feature.flatten()

                            #last feature
                            last_feature = data[real_t - self.last_feature_num: real_t, x, y, :].flatten()

                            #hist feature
                            hist_feature = data[real_t - self.hist_feature_daynum*self.timeslot_daynum: real_t: self.timeslot_daynum, x, y, :].flatten()

                            feature_vec = np.concatenate((hist_feature, last_feature))
                            feature_vec = np.concatenate((feature_vec, nbhd_feature))

                            long_term_lstm_samples.append(feature_vec)
                            long_term_weather_samples.append(weather_data[real_t])
                        lstm_att_features[att_lstm_cnt].append(np.array(long_term_lstm_samples))
                        weather_att_features[att_lstm_cnt].append(np.array(long_term_weather_samples))

                    #label
                    labels.append(data[t, x , y, :].flatten())          

        output_nbhd_att_features = []
        output_flow_att_features = []
        for i in range(self.att_lstm_num):
            lstm_att_features[i] = np.array(lstm_att_features[i])
            weather_att_features[i] = np.array(weather_att_features[i])
            for j in range(self.long_term_lstm_seq_len):
                cnn_att_features[i][j] = np.array(cnn_att_features[i][j])
                flow_att_features[i][j] = np.array(flow_att_features[i][j])
                output_nbhd_att_features.append(cnn_att_features[i][j])
                output_flow_att_features.append(flow_att_features[i][j])
        
        for i in range(self.short_term_lstm_seq_len):
            cnn_features[i] = np.array(cnn_features[i])
            flow_features[i] = np.array(flow_features[i])
        short_term_lstm_features = np.array(short_term_lstm_features)
        weather_features = np.array(weather_features)
        labels = np.array(labels)

        return output_nbhd_att_features, output_flow_att_features, lstm_att_features, weather_att_features, \
            cnn_features, flow_features, short_term_lstm_features, weather_features, labels

    def sample(self, datatype):
        if self.dataset_type == "station":
            return self.sample_station(datatype)
        elif self.dataset_type == "region":
            return self.sample_region(datatype)