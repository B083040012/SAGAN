import attention
import numpy as np
import tensorflow as tf

try:
    import keras
    from keras.models import Model
    from keras.layers import Dense, Activation, ReLU, PReLU, Input, Conv2D, Reshape, Flatten, Concatenate, LSTM, MaxPooling2D, AveragePooling2D
except:
    from tensorflow import keras
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Activation, ReLU, PReLU, Input, Conv2D, Reshape, Flatten, Concatenate, LSTM, MaxPooling2D, AveragePooling2D

class Station_Functional_Model:
    def __init__(self, config):

        self.short_term_lstm_seq_len = config["dataset"]["station"]["short_term_lstm_seq_len"]
        self.long_term_lstm_seq_len = config["dataset"]["station"]["long_term_lstm_seq_len"]
        self.att_lstm_num = config["dataset"]["station"]["att_lstm_num"]
        self.cnn_flat_size = config["model"]["station"]["cnn_flat_size"]
        self.lstm_out_size = config["model"]["station"]["lstm_out_size"]
        self.output_shape_num = config["model"]["station"]["output_shape_num"]

    def conv2d_layer(self, choice, layer_name):
        return Conv2D(filters = 64, kernel_size = (choice + 1, choice + 1), padding="same", name = layer_name+"_size_"+str(choice + 1))
    
    def pooling_layer(self, choice, layer_name):
        if choice < 3:
            return MaxPooling2D(pool_size = (choice + 2, choice + 2), strides = (1, 1), padding = "same", name = layer_name+"_max_size_"+str(choice+2))
        else:
            return AveragePooling2D(pool_size = (choice - 1, choice - 1), strides = (1, 1), padding = "same", name = layer_name+"avg_size_"+str(choice-1))

    def conv2d_activ_layer(self, choice, layer_name):
        if choice == 0:
            return ReLU(name = layer_name+"_relu")
        elif choice == 1:
            return ReLU(max_value = 6.0, name = layer_name+"_relu")
        elif choice == 2:
            return PReLU(name = layer_name+"_prelu")

    def gate_activ_layer(self, choice, layer_name):
        if choice == 0:
            return Activation("sigmoid", name = layer_name+"_sigmoid")
        elif choice == 1:
            return ReLU(max_value = 6.0, name = layer_name+"_relu6")
        elif choice == 2:
            return Activation("tanh", name = layer_name+"_tanh")

    def func_model(self, nas_choice, feature_vec_len, nbhd_size, nbhd_type, flow_type, \
        optimizer = 'adagrad', loss = 'mse', metrics=[]):

        """
        short-term input
        including:
            1. nbhd: short_term_lstm_seq_len, (nbhd_size, nbhd_size, nbhd_type,)
            2. flow: short_term_lstm_seq_len, (nbhd_size, nbhd_size, flow_type,)
            3. lstm: (short_term_lstm_seq_len, feature_vec_len,)
        """
        nbhd_inputs = [Input(shape = (nbhd_size, nbhd_size, nbhd_type,), name = "nbhd_volume_input_time_{0}".format(ts+1)) for ts in range(self.short_term_lstm_seq_len)]
        flow_inputs = [Input(shape = (nbhd_size, nbhd_size, flow_type,), name = "flow_volume_input_time_{0}".format(ts+1)) for ts in range(self.short_term_lstm_seq_len)]
        lstm_inputs = Input(shape = (self.short_term_lstm_seq_len, feature_vec_len,), name = "lstm_input")

        """
        long-term input
        including:
            1. nbhd: att_lstm_num, long_term_lstm_seq_len, (nbhd_size, nbhd_size, nbhd_type,)
            2. flow: att_lstm_num, long_term_lstm_seq_len, (nbhd_size, nbhd_size, flow_type,)
            3. lstm: att_lstm_num, (long_term_lstm_seq_len, feature_vec_len,)
        """
        flatten_att_nbhd_inputs = [Input(shape = (nbhd_size, nbhd_size, nbhd_type,), name = "att_nbhd_volume_input_time_{0}_{1}".format(att+1, ts+1)) \
            for ts in range(self.long_term_lstm_seq_len) for att in range(self.att_lstm_num)]
        flatten_att_flow_inputs = [Input(shape = (nbhd_size, nbhd_size, flow_type,), name = "att_flow_volume_input_time_{0}_{1}".format(att+1, ts+1)) \
            for ts in range(self.long_term_lstm_seq_len) for att in range(self.att_lstm_num)]

        att_nbhd_inputs = []
        att_flow_inputs = []
        for att in range(self.att_lstm_num):
            att_nbhd_inputs.append(flatten_att_nbhd_inputs[att*self.long_term_lstm_seq_len:(att+1)*self.long_term_lstm_seq_len])
            att_flow_inputs.append(flatten_att_flow_inputs[att*self.long_term_lstm_seq_len:(att+1)*self.long_term_lstm_seq_len])
        att_lstm_inputs = [Input(shape = (self.long_term_lstm_seq_len, feature_vec_len,), name = "att_lstm_input_{0}".format(att+1)) for att in range(self.att_lstm_num)]

        #1st level gate
        level = 0
        nbhd_convs = [self.conv2d_layer(nas_choice[0][level][ts][0], "nbhd_conv_time{0}_{1}".format(level, ts+1))(nbhd_inputs[ts])for ts in range(self.short_term_lstm_seq_len)]
        nbhd_convs = [self.pooling_layer(nas_choice[1][level][ts][0], "nbhd_pooling_time{0}_{1}".format(level, ts+1))(nbhd_convs[ts]) \
            if nas_choice[1][level][ts][0] < 6 else nbhd_convs[ts] for ts in range(self.short_term_lstm_seq_len)]
        nbhd_convs = [self.conv2d_activ_layer(nas_choice[2][level][ts][0], "nbhd_activ_time{0}_{1}".format(level, ts+1))(nbhd_convs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        flow_convs = [self.conv2d_layer(nas_choice[0][level][ts][1], "flow_convs_time{0}_{1}".format(level, ts+1))(flow_inputs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        flow_convs = [self.pooling_layer(nas_choice[1][level][ts][1], "flow_pooling_time{0}_{1}".format(level, ts+1))(flow_convs[ts]) \
            if nas_choice[1][level][ts][0] < 6 else flow_convs[ts] for ts in range(self.short_term_lstm_seq_len)]
        flow_convs = [self.conv2d_activ_layer(nas_choice[2][level][ts][1], "flow_activ_time{0}_{1}".format(level, ts+1))(flow_convs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        flow_gates = [self.gate_activ_layer(nas_choice[3][level][ts], "gate_activ_time{0}_{1}".format(level, ts+1))(flow_convs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(self.short_term_lstm_seq_len)]


        #2nd level gate
        level = 1
        nbhd_convs = [self.conv2d_layer(nas_choice[0][level][ts][0], "nbhd_conv_time{0}_{1}".format(level, ts+1))(nbhd_convs[ts])for ts in range(self.short_term_lstm_seq_len)]
        nbhd_convs = [self.pooling_layer(nas_choice[1][level][ts][0], "nbhd_pooling_time{0}_{1}".format(level, ts+1))(nbhd_convs[ts]) \
            if nas_choice[1][level][ts][0] < 6 else nbhd_convs[ts] for ts in range(self.short_term_lstm_seq_len)]
        nbhd_convs = [self.conv2d_activ_layer(nas_choice[2][level][ts][0], "nbhd_activ_time{0}_{1}".format(level, ts+1))(nbhd_convs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        flow_convs = [self.conv2d_layer(nas_choice[0][level][ts][1], "flow_convs_time{0}_{1}".format(level, ts+1))(flow_inputs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        flow_convs = [self.pooling_layer(nas_choice[1][level][ts][1], "flow_pooling_time{0}_{1}".format(level, ts+1))(flow_convs[ts]) \
            if nas_choice[1][level][ts][0] < 6 else flow_convs[ts] for ts in range(self.short_term_lstm_seq_len)]
        flow_convs = [self.conv2d_activ_layer(nas_choice[2][level][ts][1], "flow_activ_time{0}_{1}".format(level, ts+1))(flow_convs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        flow_gates = [self.gate_activ_layer(nas_choice[3][level][ts], "gate_activ_time{0}_{1}".format(level, ts+1))(flow_convs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(self.short_term_lstm_seq_len)]

        #3rd level gate
        level = 2
        nbhd_convs = [self.conv2d_layer(nas_choice[0][level][ts][0], "nbhd_conv_time{0}_{1}".format(level, ts+1))(nbhd_convs[ts])for ts in range(self.short_term_lstm_seq_len)]
        nbhd_convs = [self.pooling_layer(nas_choice[1][level][ts][0], "nbhd_pooling_time{0}_{1}".format(level, ts+1))(nbhd_convs[ts]) \
            if nas_choice[1][level][ts][0] < 6 else nbhd_convs[ts] for ts in range(self.short_term_lstm_seq_len)]
        nbhd_convs = [self.conv2d_activ_layer(nas_choice[2][level][ts][0], "nbhd_activ_time{0}_{1}".format(level, ts+1))(nbhd_convs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        flow_convs = [self.conv2d_layer(nas_choice[0][level][ts][1], "flow_convs_time{0}_{1}".format(level, ts+1))(flow_inputs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        flow_convs = [self.pooling_layer(nas_choice[1][level][ts][1], "flow_pooling_time{0}_{1}".format(level, ts+1))(flow_convs[ts]) \
            if nas_choice[1][level][ts][0] < 6 else flow_convs[ts] for ts in range(self.short_term_lstm_seq_len)]
        flow_convs = [self.conv2d_activ_layer(nas_choice[2][level][ts][1], "flow_activ_time{0}_{1}".format(level, ts+1))(flow_convs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        flow_gates = [self.gate_activ_layer(nas_choice[3][level][ts], "gate_activ_time{0}_{1}".format(level, ts+1))(flow_convs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(self.short_term_lstm_seq_len)]


        #dense part
        nbhd_vecs = [Flatten(name = "nbhd_flatten_time_{0}".format(ts+1))(nbhd_convs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        nbhd_vecs = [Dense(units = self.cnn_flat_size, name = "nbhd_dense_time_{0}".format(ts+1))(nbhd_vecs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        nbhd_vecs = [Activation("relu", name = "nbhd_dense_activation_time_{0}".format(ts+1))(nbhd_vecs[ts]) for ts in range(self.short_term_lstm_seq_len)]

        #feature concatenate
        nbhd_vec = Concatenate(axis=-1)(nbhd_vecs)
        nbhd_vec = Reshape(target_shape = (self.short_term_lstm_seq_len, self.cnn_flat_size))(nbhd_vec)
        lstm_input = Concatenate(axis=-1)([lstm_inputs, nbhd_vec])

        #lstm
        lstm = LSTM(units=self.lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(lstm_input)

        #attention part
        level = 0
        att_nbhd_convs = [[self.conv2d_layer(nas_choice[4][level][att][ts][0], "att_nbhd_convs_time{0}_{1}_{2}".format(level, att+1,ts+1))(att_nbhd_inputs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.pooling_layer(nas_choice[5][level][att][ts][0], "att_nbhd_pooling_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_nbhd_convs[att][ts]) \
            if nas_choice[5][level][att][ts][0] < 6 else att_nbhd_convs[att][ts] for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.conv2d_activ_layer(nas_choice[6][level][att][ts][0], "att_nbhd_activ_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_nbhd_convs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.conv2d_layer(nas_choice[4][level][att][ts][1], "att_flow_convs_time{0}_{1}_{2}".format(level, att+1,ts+1))(att_flow_inputs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.pooling_layer(nas_choice[5][level][att][ts][1], "att_flow_pooling_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_flow_convs[att][ts]) 
            if nas_choice[5][level][att][ts][1] < 6 else att_flow_convs[att][ts] for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.conv2d_activ_layer(nas_choice[6][level][att][ts][1], "att_flow_activ_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_gates = [[self.gate_activ_layer(nas_choice[7][level][att][ts], "att_gate_activ_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]

        level = 1
        att_nbhd_convs = [[self.conv2d_layer(nas_choice[4][level][att][ts][0], "att_nbhd_convs_time{0}_{1}_{2}".format(level, att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.pooling_layer(nas_choice[5][level][att][ts][0], "att_nbhd_pooling_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_nbhd_convs[att][ts]) \
            if nas_choice[5][level][att][ts][0] < 6 else att_nbhd_convs[att][ts] for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.conv2d_activ_layer(nas_choice[6][level][att][ts][0], "att_nbhd_activ_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_nbhd_convs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.conv2d_layer(nas_choice[4][level][att][ts][1], "att_flow_convs_time{0}_{1}_{2}".format(level, att+1,ts+1))(att_flow_inputs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.pooling_layer(nas_choice[5][level][att][ts][1], "att_flow_pooling_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_flow_convs[att][ts]) 
            if nas_choice[5][level][att][ts][1] < 6 else att_flow_convs[att][ts] for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.conv2d_activ_layer(nas_choice[6][level][att][ts][1], "att_flow_activ_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_gates = [[self.gate_activ_layer(nas_choice[7][level][att][ts], "att_gate_activ_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]

        level = 2
        att_nbhd_convs = [[self.conv2d_layer(nas_choice[4][level][att][ts][0], "att_nbhd_convs_time{0}_{1}_{2}".format(level, att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.pooling_layer(nas_choice[5][level][att][ts][0], "att_nbhd_pooling_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_nbhd_convs[att][ts]) \
            if nas_choice[5][level][att][ts][0] < 6 else att_nbhd_convs[att][ts] for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.conv2d_activ_layer(nas_choice[6][level][att][ts][0], "att_nbhd_activ_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_nbhd_convs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.conv2d_layer(nas_choice[4][level][att][ts][1], "att_flow_convs_time{0}_{1}_{2}".format(level, att+1,ts+1))(att_flow_inputs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.pooling_layer(nas_choice[5][level][att][ts][1], "att_flow_pooling_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_flow_convs[att][ts]) 
            if nas_choice[5][level][att][ts][1] < 6 else att_flow_convs[att][ts] for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.conv2d_activ_layer(nas_choice[6][level][att][ts][1], "att_flow_activ_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_gates = [[self.gate_activ_layer(nas_choice[7][level][att][ts], "att_gate_activ_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]

        att_nbhd_vecs = [[Flatten(name = "att_nbhd_flatten_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_vecs = [[Dense(units = self.cnn_flat_size, name = "att_nbhd_dense_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_vecs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_vecs = [[Activation("relu", name = "att_nbhd_dense_activation_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_vecs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]

        att_nbhd_vec = [Concatenate(axis=-1)(att_nbhd_vecs[att]) for att in range(self.att_lstm_num)]
        att_nbhd_vec = [Reshape(target_shape = (self.long_term_lstm_seq_len, self.cnn_flat_size))(att_nbhd_vec[att]) for att in range(self.att_lstm_num)]
        att_lstm_input = [Concatenate(axis=-1)([att_lstm_inputs[att], att_nbhd_vec[att]]) for att in range(self.att_lstm_num)]

        att_lstms = [LSTM(units=self.lstm_out_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, name="att_lstm_{0}".format(att + 1))(att_lstm_input[att]) for att in range(self.att_lstm_num)]

        #compare
        att_low_level=[attention.Attention(method='cba')([att_lstms[att], lstm]) for att in range(self.att_lstm_num)]
        att_low_level=Concatenate(axis=-1)(att_low_level)
        att_low_level=Reshape(target_shape=(self.att_lstm_num, self.lstm_out_size))(att_low_level)


        att_high_level = LSTM(units=self.lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(att_low_level)

        lstm_all = Concatenate(axis=-1)([att_high_level, lstm])
        # lstm_all = Dropout(rate = .3)(lstm_all)
        lstm_all = Dense(units = self.output_shape_num)(lstm_all)
        pred_volume = Activation('tanh')(lstm_all)

        # input variable comparison
        # att_nbhd, att_flow, att_lstm, short_nbhd, short_flow, short_lstm
        # flatten_att_nbhd_inputs + flatten_att_flow_inputs + att_lstm_inputs + nbhd_inputs + flow_inputs + [lstm_inputs,]

        model = Model(inputs = flatten_att_nbhd_inputs + flatten_att_flow_inputs + att_lstm_inputs + \
            nbhd_inputs + flow_inputs + [lstm_inputs, ], outputs = pred_volume)
        model.compile(optimizer = optimizer, loss = loss, metrics=metrics)
        return model

class Region_Functional_Model:
    def __init__(self, config):

        self.short_term_lstm_seq_len = config["dataset"]["region"]["short_term_lstm_seq_len"]
        self.long_term_lstm_seq_len = config["dataset"]["region"]["long_term_lstm_seq_len"]
        self.att_lstm_num = config["dataset"]["region"]["att_lstm_num"]
        self.cnn_flat_size = config["model"]["region"]["cnn_flat_size"]
        self.lstm_out_size = config["model"]["region"]["lstm_out_size"]
        self.output_shape_num = config["model"]["region"]["output_shape_num"]

    def conv2d_layer(self, choice, layer_name):
        return Conv2D(filters = 64, kernel_size = (choice + 1, choice + 1), padding="same", name = layer_name+"_size_"+str(choice + 1))
    
    def pooling_layer(self, choice, layer_name):
        if choice < 3:
            return MaxPooling2D(pool_size = (choice + 2, choice + 2), strides = (1, 1), padding = "same", name = layer_name+"_max_size_"+str(choice+2))
        else:
            return AveragePooling2D(pool_size = (choice - 1, choice - 1), strides = (1, 1), padding = "same", name = layer_name+"avg_size_"+str(choice-1))

    def conv2d_activ_layer(self, choice, layer_name):
        if choice == 0:
            return ReLU(name = layer_name+"_relu")
        elif choice == 1:
            return ReLU(max_value = 6.0, name = layer_name+"_relu")
        elif choice == 2:
            return PReLU(name = layer_name+"_prelu")

    def gate_activ_layer(self, choice, layer_name):
        if choice == 0:
            return Activation("sigmoid", name = layer_name+"_sigmoid")
        elif choice == 1:
            return ReLU(max_value = 6.0, name = layer_name+"_relu6")
        elif choice == 2:
            return Activation("tanh", name = layer_name+"_tanh")

    def func_model(self, nas_choice, feature_vec_len, nbhd_size, nbhd_type, flow_type, \
        optimizer = 'adagrad', loss = 'mse', metrics=[]):

        """
        short-term input
        including:
            1. nbhd: short_term_lstm_seq_len, (nbhd_size, nbhd_size, nbhd_type,)
            2. flow: short_term_lstm_seq_len, (nbhd_size, nbhd_size, flow_type,)
            3. lstm: (short_term_lstm_seq_len, feature_vec_len,)
        """
        nbhd_inputs = [Input(shape = (nbhd_size, nbhd_size, nbhd_type,), name = "nbhd_volume_input_time_{0}".format(ts+1)) for ts in range(self.short_term_lstm_seq_len)]
        flow_inputs = [Input(shape = (nbhd_size, nbhd_size, flow_type,), name = "flow_volume_input_time_{0}".format(ts+1)) for ts in range(self.short_term_lstm_seq_len)]
        lstm_inputs = Input(shape = (self.short_term_lstm_seq_len, feature_vec_len,), name = "lstm_input")

        """
        long-term input
        including:
            1. nbhd: att_lstm_num, long_term_lstm_seq_len, (nbhd_size, nbhd_size, nbhd_type,)
            2. flow: att_lstm_num, long_term_lstm_seq_len, (nbhd_size, nbhd_size, flow_type,)
            3. lstm: att_lstm_num, (long_term_lstm_seq_len, feature_vec_len,)
        """
        flatten_att_nbhd_inputs = [Input(shape = (nbhd_size, nbhd_size, nbhd_type,), name = "att_nbhd_volume_input_time_{0}_{1}".format(att+1, ts+1)) \
            for ts in range(self.long_term_lstm_seq_len) for att in range(self.att_lstm_num)]
        flatten_att_flow_inputs = [Input(shape = (nbhd_size, nbhd_size, flow_type,), name = "att_flow_volume_input_time_{0}_{1}".format(att+1, ts+1)) \
            for ts in range(self.long_term_lstm_seq_len) for att in range(self.att_lstm_num)]

        att_nbhd_inputs = []
        att_flow_inputs = []
        for att in range(self.att_lstm_num):
            att_nbhd_inputs.append(flatten_att_nbhd_inputs[att*self.long_term_lstm_seq_len:(att+1)*self.long_term_lstm_seq_len])
            att_flow_inputs.append(flatten_att_flow_inputs[att*self.long_term_lstm_seq_len:(att+1)*self.long_term_lstm_seq_len])
        att_lstm_inputs = [Input(shape = (self.long_term_lstm_seq_len, feature_vec_len,), name = "att_lstm_input_{0}".format(att+1)) for att in range(self.att_lstm_num)]

        #1st level gate
        level = 0
        nbhd_convs = [self.conv2d_layer(nas_choice[0][level][ts][0], "nbhd_conv_time{0}_{1}".format(level, ts+1))(nbhd_inputs[ts])for ts in range(self.short_term_lstm_seq_len)]
        nbhd_convs = [self.pooling_layer(nas_choice[1][level][ts][0], "nbhd_pooling_time{0}_{1}".format(level, ts+1))(nbhd_convs[ts]) \
            if nas_choice[1][level][ts][0] < 6 else nbhd_convs[ts] for ts in range(self.short_term_lstm_seq_len)]
        nbhd_convs = [self.conv2d_activ_layer(nas_choice[2][level][ts][0], "nbhd_activ_time{0}_{1}".format(level, ts+1))(nbhd_convs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        flow_convs = [self.conv2d_layer(nas_choice[0][level][ts][1], "flow_convs_time{0}_{1}".format(level, ts+1))(flow_inputs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        flow_convs = [self.pooling_layer(nas_choice[1][level][ts][1], "flow_pooling_time{0}_{1}".format(level, ts+1))(flow_convs[ts]) \
            if nas_choice[1][level][ts][0] < 6 else flow_convs[ts] for ts in range(self.short_term_lstm_seq_len)]
        flow_convs = [self.conv2d_activ_layer(nas_choice[2][level][ts][1], "flow_activ_time{0}_{1}".format(level, ts+1))(flow_convs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        flow_gates = [self.gate_activ_layer(nas_choice[3][level][ts], "gate_activ_time{0}_{1}".format(level, ts+1))(flow_convs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(self.short_term_lstm_seq_len)]


        #2nd level gate
        level = 1
        nbhd_convs = [self.conv2d_layer(nas_choice[0][level][ts][0], "nbhd_conv_time{0}_{1}".format(level, ts+1))(nbhd_convs[ts])for ts in range(self.short_term_lstm_seq_len)]
        nbhd_convs = [self.pooling_layer(nas_choice[1][level][ts][0], "nbhd_pooling_time{0}_{1}".format(level, ts+1))(nbhd_convs[ts]) \
            if nas_choice[1][level][ts][0] < 6 else nbhd_convs[ts] for ts in range(self.short_term_lstm_seq_len)]
        nbhd_convs = [self.conv2d_activ_layer(nas_choice[2][level][ts][0], "nbhd_activ_time{0}_{1}".format(level, ts+1))(nbhd_convs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        flow_convs = [self.conv2d_layer(nas_choice[0][level][ts][1], "flow_convs_time{0}_{1}".format(level, ts+1))(flow_inputs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        flow_convs = [self.pooling_layer(nas_choice[1][level][ts][1], "flow_pooling_time{0}_{1}".format(level, ts+1))(flow_convs[ts]) \
            if nas_choice[1][level][ts][0] < 6 else flow_convs[ts] for ts in range(self.short_term_lstm_seq_len)]
        flow_convs = [self.conv2d_activ_layer(nas_choice[2][level][ts][1], "flow_activ_time{0}_{1}".format(level, ts+1))(flow_convs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        flow_gates = [self.gate_activ_layer(nas_choice[3][level][ts], "gate_activ_time{0}_{1}".format(level, ts+1))(flow_convs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(self.short_term_lstm_seq_len)]

        #3rd level gate
        level = 2
        nbhd_convs = [self.conv2d_layer(nas_choice[0][level][ts][0], "nbhd_conv_time{0}_{1}".format(level, ts+1))(nbhd_convs[ts])for ts in range(self.short_term_lstm_seq_len)]
        nbhd_convs = [self.pooling_layer(nas_choice[1][level][ts][0], "nbhd_pooling_time{0}_{1}".format(level, ts+1))(nbhd_convs[ts]) \
            if nas_choice[1][level][ts][0] < 6 else nbhd_convs[ts] for ts in range(self.short_term_lstm_seq_len)]
        nbhd_convs = [self.conv2d_activ_layer(nas_choice[2][level][ts][0], "nbhd_activ_time{0}_{1}".format(level, ts+1))(nbhd_convs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        flow_convs = [self.conv2d_layer(nas_choice[0][level][ts][1], "flow_convs_time{0}_{1}".format(level, ts+1))(flow_inputs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        flow_convs = [self.pooling_layer(nas_choice[1][level][ts][1], "flow_pooling_time{0}_{1}".format(level, ts+1))(flow_convs[ts]) \
            if nas_choice[1][level][ts][0] < 6 else flow_convs[ts] for ts in range(self.short_term_lstm_seq_len)]
        flow_convs = [self.conv2d_activ_layer(nas_choice[2][level][ts][1], "flow_activ_time{0}_{1}".format(level, ts+1))(flow_convs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        flow_gates = [self.gate_activ_layer(nas_choice[3][level][ts], "gate_activ_time{0}_{1}".format(level, ts+1))(flow_convs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(self.short_term_lstm_seq_len)]


        #dense part
        nbhd_vecs = [Flatten(name = "nbhd_flatten_time_{0}".format(ts+1))(nbhd_convs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        nbhd_vecs = [Dense(units = self.cnn_flat_size, name = "nbhd_dense_time_{0}".format(ts+1))(nbhd_vecs[ts]) for ts in range(self.short_term_lstm_seq_len)]
        nbhd_vecs = [Activation("relu", name = "nbhd_dense_activation_time_{0}".format(ts+1))(nbhd_vecs[ts]) for ts in range(self.short_term_lstm_seq_len)]

        #feature concatenate
        nbhd_vec = Concatenate(axis=-1)(nbhd_vecs)
        nbhd_vec = Reshape(target_shape = (self.short_term_lstm_seq_len, self.cnn_flat_size))(nbhd_vec)
        lstm_input = Concatenate(axis=-1)([lstm_inputs, nbhd_vec])

        #lstm
        lstm = LSTM(units=self.lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(lstm_input)

        #attention part
        level = 0
        att_nbhd_convs = [[self.conv2d_layer(nas_choice[4][level][att][ts][0], "att_nbhd_convs_time{0}_{1}_{2}".format(level, att+1,ts+1))(att_nbhd_inputs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.pooling_layer(nas_choice[5][level][att][ts][0], "att_nbhd_pooling_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_nbhd_convs[att][ts]) \
            if nas_choice[5][level][att][ts][0] < 6 else att_nbhd_convs[att][ts] for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.conv2d_activ_layer(nas_choice[6][level][att][ts][0], "att_nbhd_activ_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_nbhd_convs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.conv2d_layer(nas_choice[4][level][att][ts][1], "att_flow_convs_time{0}_{1}_{2}".format(level, att+1,ts+1))(att_flow_inputs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.pooling_layer(nas_choice[5][level][att][ts][1], "att_flow_pooling_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_flow_convs[att][ts]) 
            if nas_choice[5][level][att][ts][1] < 6 else att_flow_convs[att][ts] for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.conv2d_activ_layer(nas_choice[6][level][att][ts][1], "att_flow_activ_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_gates = [[self.gate_activ_layer(nas_choice[7][level][att][ts], "att_gate_activ_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]

        level = 1
        att_nbhd_convs = [[self.conv2d_layer(nas_choice[4][level][att][ts][0], "att_nbhd_convs_time{0}_{1}_{2}".format(level, att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.pooling_layer(nas_choice[5][level][att][ts][0], "att_nbhd_pooling_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_nbhd_convs[att][ts]) \
            if nas_choice[5][level][att][ts][0] < 6 else att_nbhd_convs[att][ts] for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.conv2d_activ_layer(nas_choice[6][level][att][ts][0], "att_nbhd_activ_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_nbhd_convs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.conv2d_layer(nas_choice[4][level][att][ts][1], "att_flow_convs_time{0}_{1}_{2}".format(level, att+1,ts+1))(att_flow_inputs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.pooling_layer(nas_choice[5][level][att][ts][1], "att_flow_pooling_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_flow_convs[att][ts]) 
            if nas_choice[5][level][att][ts][1] < 6 else att_flow_convs[att][ts] for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.conv2d_activ_layer(nas_choice[6][level][att][ts][1], "att_flow_activ_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_gates = [[self.gate_activ_layer(nas_choice[7][level][att][ts], "att_gate_activ_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]

        level = 2
        att_nbhd_convs = [[self.conv2d_layer(nas_choice[4][level][att][ts][0], "att_nbhd_convs_time{0}_{1}_{2}".format(level, att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.pooling_layer(nas_choice[5][level][att][ts][0], "att_nbhd_pooling_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_nbhd_convs[att][ts]) \
            if nas_choice[5][level][att][ts][0] < 6 else att_nbhd_convs[att][ts] for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[self.conv2d_activ_layer(nas_choice[6][level][att][ts][0], "att_nbhd_activ_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_nbhd_convs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.conv2d_layer(nas_choice[4][level][att][ts][1], "att_flow_convs_time{0}_{1}_{2}".format(level, att+1,ts+1))(att_flow_inputs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.pooling_layer(nas_choice[5][level][att][ts][1], "att_flow_pooling_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_flow_convs[att][ts]) 
            if nas_choice[5][level][att][ts][1] < 6 else att_flow_convs[att][ts] for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.conv2d_activ_layer(nas_choice[6][level][att][ts][1], "att_flow_activ_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_gates = [[self.gate_activ_layer(nas_choice[7][level][att][ts], "att_gate_activ_time{0}_{1}_{2}".format(level, att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]

        att_nbhd_vecs = [[Flatten(name = "att_nbhd_flatten_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_vecs = [[Dense(units = self.cnn_flat_size, name = "att_nbhd_dense_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_vecs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_vecs = [[Activation("relu", name = "att_nbhd_dense_activation_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_vecs[att][ts]) for ts in range(self.long_term_lstm_seq_len)] for att in range(self.att_lstm_num)]

        att_nbhd_vec = [Concatenate(axis=-1)(att_nbhd_vecs[att]) for att in range(self.att_lstm_num)]
        att_nbhd_vec = [Reshape(target_shape = (self.long_term_lstm_seq_len, self.cnn_flat_size))(att_nbhd_vec[att]) for att in range(self.att_lstm_num)]
        att_lstm_input = [Concatenate(axis=-1)([att_lstm_inputs[att], att_nbhd_vec[att]]) for att in range(self.att_lstm_num)]

        att_lstms = [LSTM(units=self.lstm_out_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, name="att_lstm_{0}".format(att + 1))(att_lstm_input[att]) for att in range(self.att_lstm_num)]

        #compare
        att_low_level=[attention.Attention(method='cba')([att_lstms[att], lstm]) for att in range(self.att_lstm_num)]
        att_low_level=Concatenate(axis=-1)(att_low_level)
        att_low_level=Reshape(target_shape=(self.att_lstm_num, self.lstm_out_size))(att_low_level)

        att_high_level = LSTM(units=self.lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(att_low_level)

        lstm_all = Concatenate(axis=-1)([att_high_level, lstm])
        # lstm_all = Dropout(rate = .3)(lstm_all)
        lstm_all = Dense(units = self.output_shape_num)(lstm_all)
        pred_volume = Activation('tanh')(lstm_all)

        # input variable comparison
        # att_nbhd, att_flow, att_lstm, short_nbhd, short_flow, short_lstm
        # flatten_att_nbhd_inputs + flatten_att_flow_inputs + att_lstm_inputs + nbhd_inputs + flow_inputs + [lstm_inputs,]

        model = Model(inputs = flatten_att_nbhd_inputs + flatten_att_flow_inputs + att_lstm_inputs + \
            nbhd_inputs + flow_inputs + [lstm_inputs, ], outputs = pred_volume)
        model.compile(optimizer = optimizer, loss = loss, metrics=metrics)
        return model