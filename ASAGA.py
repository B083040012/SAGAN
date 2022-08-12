import random, math
import numpy as np
from tensorflow.python.keras import backend as K
from criterion import eval_rmse

class ASAGA_Searcher():
    def __init__(self, config, logger, model, val_data, val_label, dataset_type):
        self.config = config
        self.logger = logger
        self.model = model
        self.dataset_type = dataset_type
        self.val_data = val_data
        self.val_label = val_label
        
        """
        Initialize parameters
        """
        self.gate_level = self.config["model"]["gate_level"]
        self.conv_choice_num = self.config["model"]["conv_choice_num"]
        self.pooling_choice_num = self.config["model"]["pooling_choice_num"]
        self.conv_activ_choice_num = self.config["model"]["conv_activ_choice_num"]
        self.gate_activ_choice_num = self.config["model"]["gate_activ_choice_num"]

        self.short_term_lstm_seq_len = config["dataset"][dataset_type]["short_term_lstm_seq_len"]
        self.att_lstm_num = config["dataset"][dataset_type]["att_lstm_num"]
        self.long_term_lstm_seq_len = config["dataset"][dataset_type]["long_term_lstm_seq_len"]

        self.generation_num = config["searching"]["asaga"]["generation_num"]
        self.population_num = config["searching"]["asaga"]["population_num"]
        self.annealing_ratio = config["searching"]["asaga"]["annealing_ratio"]
        self.initial_tmp = config["searching"]["asaga"]["initial_tmp"]
        self.final_tmp = config["searching"]["asaga"]["final_tmp"]
        self.crossover_rate = config["searching"]["asaga"]["crossover_rate"]
        self.mutation_rate = config["searching"]["asaga"]["mutation_rate"]

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
                print("invalid target for region-level dataset")
                raise Exception
            else:
                self.label_max = config["dataset"]["region"]["volume_test_max"]
        self.threshold = config["dataset"][dataset_type]["threshold"]

        # denormalized
        self.val_label *= self.label_max

        # if label are all below threshold, end the program
        valid_label = np.sum(self.val_label > self.threshold)
        if valid_label == 0:
            self.logger.info("[Architecture Searching] invalid label !!")
            raise Exception

    def search_architecture(self, val_data, val_label):
        """
        Initialization
            1. initialize population by randomly producing nas choice
            2. calculate fitness for each population
        """
        parent_population = []
        for p in range(self.population_num):
            """
            Initialize the nas choice for station-level / region-level datset:
            including:
                1. short_conv_choice: level * short_lstm_seq_len * 2 (nbhd / flow)
                2. short_pooling_choice: level * short_lstm_seq_len * 2 (nbhd / flow)
                3. short_conv_activ_choice: level * short_lstm_seq_len * 2 (nbhd / flow)
                4. short_gate_activ_choice: level * short_lstm_seq_len * 1
                5. long_conv_choice: level * long_lstm_num * long_lstm_seq_len * 2 (nbhd / flow)
                6. long_pooling_choice: level * long_lstm_num * long_lstm_seq_len * 2 (nbhd / flow)
                7. long_conv_activ_choice: level * long_lstm_num * long_lstm_seq_len * 2 (nbhd / flow)
                8. long_gate_activ_choice: level * long_lstm_num * long_lstm_seq_len * 1
            """
            # short_conv_choice = list(np.random.randint(self.conv_choice_num, size = (self.gate_level, self.short_term_lstm_seq_len, 2), dtype = int))
            # short_pooling_choice = list(np.random.randint(self.pooling_choice_num, size = (self.gate_level, self.short_term_lstm_seq_len, 2), dtype = int))
            # short_conv_activ_choice = list(np.random.randint(self.conv_activ_choice_num, size = (self.gate_level, self.short_term_lstm_seq_len, 2), dtype = int))
            # short_gate_activ_choice = list(np.random.randint(self.gate_activ_choice_num, size = (self.gate_level, self.short_term_lstm_seq_len), dtype = int))

            # # long term choice
            # long_conv_choice = list(np.random.randint(self.conv_choice_num, size = (self.gate_level, self.att_lstm_num, self.long_term_lstm_seq_len, 2), dtype = int))
            # long_pooling_choice = list(np.random.randint(self.pooling_choice_num, size = (self.gate_level, self.att_lstm_num, self.long_term_lstm_seq_len, 2), dtype = int))
            # long_conv_activ_choice = list(np.random.randint(self.conv_activ_choice_num, size = (self.gate_level, self.att_lstm_num, self.long_term_lstm_seq_len, 2), dtype = int))
            # long_gate_activ_choice = list(np.random.randint(self.gate_activ_choice_num, size = (self.gate_level, self.att_lstm_num, self.long_term_lstm_seq_len), dtype = int))
            # short term choice
            short_conv_choice = np.random.randint(self.conv_choice_num, size = (self.gate_level, self.short_term_lstm_seq_len, 2), dtype = int)
            short_pooling_choice = np.random.randint(self.pooling_choice_num, size = (self.gate_level, self.short_term_lstm_seq_len, 2), dtype = int)
            short_conv_activ_choice = np.random.randint(self.conv_activ_choice_num, size = (self.gate_level, self.short_term_lstm_seq_len, 2), dtype = int)
            short_gate_activ_choice = np.random.randint(self.gate_activ_choice_num, size = (self.gate_level, self.short_term_lstm_seq_len), dtype = int)

            # long term choice
            long_conv_choice = np.random.randint(self.conv_choice_num, size = (self.gate_level, self.att_lstm_num, self.long_term_lstm_seq_len, 2), dtype = int)
            long_pooling_choice = np.random.randint(self.pooling_choice_num, size = (self.gate_level, self.att_lstm_num, self.long_term_lstm_seq_len, 2), dtype = int)
            long_conv_activ_choice = np.random.randint(self.conv_activ_choice_num, size = (self.gate_level, self.att_lstm_num, self.long_term_lstm_seq_len, 2), dtype = int)
            long_gate_activ_choice = np.random.randint(self.gate_activ_choice_num, size = (self.gate_level, self.att_lstm_num, self.long_term_lstm_seq_len), dtype = int)
            architecture = [short_conv_choice, short_pooling_choice, short_conv_activ_choice, short_gate_activ_choice, \
                long_conv_choice, long_pooling_choice, long_conv_activ_choice, long_gate_activ_choice]
            parent_population.append(architecture)
        
        # evaluate each architecture
        # parent_population=np.array(parent_population)
        self.logger.info("[Architecture Searching] shape of parent_population")
        for sub_list in parent_population[0]:
            self.logger.info("shape = {0}".format(sub_list.shape))
        self.logger.info("[Architecture Searching] evaluating parent population, wait for a sec...")
        parent_fitness=self.evaluate_architecture(parent_population)
        self.logger.info("[Architecture Searching] fitness of each initialized parent: ")
        for index, fitness in enumerate(parent_fitness):
            self.logger.info("[Architecture {0}]: fitness = {1}".format(index, fitness))
        tmp_best_loss=min(parent_fitness)
        # determine the scaler for SA acceptence probability
        tmp_worst_loss = max(parent_fitness)
        self.scaler = 1
        while (tmp_worst_loss - tmp_best_loss) * self.scaler < 1:
            self.scaler *= 10
        self.logger.info("[Architecture Searching] scaler = {0}".format(self.scaler))
        tmp_best_index=parent_fitness.index(tmp_best_loss)
        self.logger.info("[Architecture Searching] population initialize complete,  tmp_best_loss: %.5f" %(tmp_best_loss))

        """
        Generation Start
        1. loop (n/2) times:
            (a) generate two offsprings from two randomly chosen parent by crossover
            (b) using SA to select the parent of offspring
            (c) overwrite the old architecture with selected architecture
        2. lower temperature T
        """
        self.curr_tmp=self.initial_tmp
        gen = 0
        global_best_loss=tmp_best_loss
        global_best_architecture=parent_population[tmp_best_index]

        self.logger.info("--------------[Generation Start]--------------")
        while gen < self.generation_num and self.curr_tmp >= self.final_tmp:
            if self.curr_tmp<=self.final_tmp:
                break
            # randomly choose parent and avoid choose repeatly
            all_index_list = [pop_index for pop_index in range(0, self.population_num)]
            random.shuffle(all_index_list)
            for loop in range(int(self.population_num/2)):
                index_list = [all_index_list.pop(), all_index_list.pop()]
                parent_list=[parent_population[index] for index in index_list]
                parent_subfitness=[parent_fitness[index] for index in index_list]
                # print(parent_list[1][])
                offspring_list=self.crossover(parent_list)
                # print(parent_list[1])
                self.mutation(offspring_list)
                offspring_subfitness=self.evaluate_architecture(offspring_list)
                new_fitness=self.selection(parent_subfitness, offspring_subfitness, parent_population, offspring_list, index_list)
                for i in range(len(new_fitness)):
                    parent_fitness[index_list[i]]=new_fitness[i]
            # tmp_best_index=np.argmin(parent_fitness)
            # tmp_best_loss=parent_fitness[tmp_best_index]
            tmp_best_loss=min(parent_fitness)
            tmp_best_index=parent_fitness.index(tmp_best_loss)
            tmp_best_architecture=parent_population[tmp_best_index]
            if global_best_loss>tmp_best_loss:
                global_best_loss=tmp_best_loss
                global_best_architecture=tmp_best_architecture
                self.logger.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                self.logger.info("[Best Loss] gen:%d, gloabl_best_loss: %.5f" %(gen, global_best_loss))
                self.logger.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            self.logger.info("[Generation %3d] temperature: %.5f, tmp_best: %.5f, gloabl_best_loss: %.5f" %(gen, self.curr_tmp, tmp_best_loss, global_best_loss))
            self.curr_tmp=self.curr_tmp*self.annealing_ratio
            gen += 1
        self.logger.info("--------------[Generation End]--------------")

        return global_best_architecture, global_best_loss

    def crossover(self, parent_list):
        """
        Crossover
            1. crossover on single point
            2. notice that each architecture has multiple sublist-->
                crossover in each sublist (current) vs. crossover on the whole list
        """
        offspring_list = list()
        offspring_list.append([])
        offspring_list.append([])
        for sub_index in range(0, len(parent_list[0])):
            prob = np.random.uniform(0,1)
            cross_point=np.random.randint(low=0, high=len(parent_list[0][sub_index].flatten()))
            shape = parent_list[0][sub_index].shape
            tmp_sublist = [parent_list[0][sub_index].flatten(), parent_list[1][sub_index].flatten()]
            if prob > self.crossover_rate:
                tmp_sublist[0][:cross_point]=parent_list[1][sub_index].flatten()[:cross_point]
                tmp_sublist[1][cross_point:]=parent_list[0][sub_index].flatten()[cross_point:]

            offspring_list[0].append(tmp_sublist[0].reshape(shape))
            offspring_list[1].append(tmp_sublist[1].reshape(shape))
        return offspring_list

    def mutation(self, offspring_list):
        """
        Mutation (Swap Mutation)
            for each sublist, if the prob is lower than mutation_rate:
            1. randomly choose two swap point
            2. choose again if both value are same
            3. swap the value of two point
        """
        for offspring in offspring_list:
            for sub_list in offspring:
                prob = np.random.uniform(0, 1)
                if prob > self.mutation_rate:
                    continue
                shape = sub_list.shape
                flat_sub_list = sub_list.flatten()
                swap_p1, swap_p2 = np.random.randint(low=0, high=len(flat_sub_list)), np.random.randint(low=0, high=len(flat_sub_list))
                cnt = 0
                while (flat_sub_list[swap_p1] == flat_sub_list[swap_p2]) and (cnt < 5):
                    swap_p1, swap_p2 = np.random.randint(low=0, high=len(flat_sub_list)), np.random.randint(low=0, high=len(flat_sub_list))
                    cnt += 1
                flat_sub_list[swap_p1], flat_sub_list[swap_p2] = flat_sub_list[swap_p2], flat_sub_list[swap_p1]
                sub_list = flat_sub_list.reshape(shape)

    def selection(self, parent_subfitness, offspring_subfitness, parent_population, offspring_list, index_list):
        """
        Selection
            1. select the better offspring or may reserve the bad offspring depending on SA
            2. replace current population directly
            3. return new fitness value (* may have better solution)
            * maybe have to check the architecture is valid or not--> for now, there is no need to check this problem
        """
        new_fitness=parent_subfitness
        for i in range(len(parent_subfitness)):
            prob=np.random.uniform(0,1)
            accept_prob = 0
            if parent_subfitness[i] < offspring_subfitness[i]:
                fitness_diff = -(offspring_subfitness[i]-parent_subfitness[i])
                # too small, ignore the difference
                if fitness_diff * self.scaler > -0.1:
                    continue
                # two criterion for SA acceptance probability
                accept_prob = math.exp( (fitness_diff * self.scaler) / self.curr_tmp)
                # accept_prob = 1 / (1 + math.exp( fitness_diff * self.scaler / self.curr_tmp))
                self.logger.info("temp: {0}, offspring: {1}, parent: {2}, accept_prob: {3}, prob: {4}".format(self.curr_tmp, \
                    offspring_subfitness[i], parent_subfitness[i], accept_prob, prob))
            if parent_subfitness[i] >= offspring_subfitness[i]:
                parent_population[index_list[i]] = offspring_list[i]
                new_fitness[i] = offspring_subfitness[i]
            elif prob < accept_prob:
                parent_population[index_list[i]] = offspring_list[i]
                new_fitness[i] = offspring_subfitness[i]
        return new_fitness

    def evaluate_architecture(self, architecture_list):
        """
        Evaluate architecture in population,
        return the loss value of each architecture
        """
        architecture_loss=[]
        for index, architecture in enumerate(architecture_list):
            self.model.set_choice(architecture)
            y_pred = self.model.predict(self.val_data)
            # denormalized
            y_pred=y_pred * self.label_max
            loss_rmse = eval_rmse(self.val_label, y_pred, self.threshold)
            architecture_loss.append(loss_rmse)
            K.clear_session()
        return architecture_loss