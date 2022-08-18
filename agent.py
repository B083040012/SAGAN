import logging, yaml, argparse, os, sys
import numpy as np
from time import strftime, gmtime
from file_loader import File_Loader
from supernet_training import Supernet_Training
from search import Search
from architecture_retrain import Architecture_Retrain
from evaluate_architecture import Evaluate_Architecture

parser = argparse.ArgumentParser(description = "SAGAN project")
parser.add_argument('--dataset_type', type = str, default = 'station', help = 'select dataset with different preprocessing')
parser.add_argument('--project_name', type = str, default = strftime("%m_%d_test", gmtime()), help = 'dir name for saving log file and model weight')
parser.add_argument('--start_phase', type = int, default = 1, help = '1: supernet training, 2: searching, 3: retrain, 4: evaluate')
parser.add_argument('--end_phase', type = int, default = 4, help = '1: supernet training, 2: searching, 3: retrain, 4: evaluate')
parser.add_argument('--running_times', type = int, default = 1, help = 'times for repeating start to end phase')
parser.add_argument('--debug_mode', type = str, default = "False", help = 'smaller dataset, epochs, etc for quick check of program bug')

args = parser.parse_args()

class Agent():
    def __init__(self, args):

        """
        Agent Constructor
            1. load command arguments (args)
            2. load yml parameters (config)
            3. create directory named by the args
            4. create log file
        """
        self.args = args
        
        # create project directory
        self.project_dir = "./project_file/" + self.args.project_name
        if not os.path.exists(self.project_dir):
            # if the project is new
            os.makedirs(self.project_dir)
            # load yaml from root
            with open('parameters.yml', 'r') as stream:
                config = yaml.load(stream, Loader = yaml.FullLoader)
                
            # store config to the project folder
            with open(self.project_dir + '/parameters.yml', 'w') as writer:
                yaml.dump(config, writer, default_flow_style = False)
            self.config = config["debug"] if self.args.debug_mode == "True" else config["experiment"]
        else:
            # load yaml from project folder if the project run before
            with open(self.project_dir + '/parameters.yml', 'r') as stream:
                config = yaml.load(stream, Loader = yaml.FullLoader)
            self.config = config["debug"] if self.args.debug_mode == "True" else config["experiment"]

        # create agent log file
        formatter = logging.Formatter('%(asctime)s %(message)s')
        file_handler = logging.FileHandler(self.project_dir+"\\agent.log")
        stream_handler = logging.StreamHandler()
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger = logging.getLogger('agent')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        
        self.logger = logging.getLogger('agent')
        self.logger.info("[Agent] Starting SAGAN procedure...")
        self.logger.info("[Agent] Dataset type: {0}, no external feature".format(self.args.dataset_type))
    
    def process(self):

        """
        Display args and parameters
        """
        self.logger.info("[Agent] args for project {0}: {1}".format(self.args.project_name, self.args))
        self.logger.info("[Agent] parameters for projects {0}: {1}".format(self.args.project_name, self.config))

        """
        Loop for test
            1. loading operation object and data
            2. run multiple times for single phase
                2.1 create test directory
                2.2 running operation process (set logger, load model and fit / evaluate)
        """
        # for args in itertools.product(range(self.args.start_phase, self.args.end_phase + 1), range(self.args.running_times)):
            # test_time, curr_phase = args[0], args[1]

        phase_list = ['Supernet_Training', 'Search', 'Architecture_Retrain', 'Evaluate_Architecture']
        for curr_phase in range(self.args.start_phase, self.args.end_phase + 1):
            # call operation object depends on curr_phase and load data first
            operation = phase_list[curr_phase - 1]
            phase_builder = getattr(sys.modules[__name__], operation)
            my_curr_phase = phase_builder(self.args.dataset_type, self.config)
            self.logger.info("[Agent] loading data for operation {0}...".format(operation))
            my_curr_phase.load_data(self.logger)
            self.logger.info("[Agent] loading data for operation {0} complete".format(operation))
            for test_time in range(self.args.running_times):
                # create test dir storing log and saved model
                test_dir = self.project_dir + "/test_" + str(test_time) + "/"
                if not os.path.exists(test_dir):
                    os.makedirs(test_dir + self.config["file"]["log_path"])
                    os.makedirs(test_dir + self.config["file"]["model_path"])
                self.logger.info("==========================================================")
                self.logger.info("[Agent] {0} phase on test {1}".format(operation, test_time))
                my_curr_phase.process(test_dir)
            self.logger.info("[Agent] end of all {0} tests on {1} phase".format(self.args.running_times, operation))
            self.logger.info("==========================================================")
            del my_curr_phase
        self.logger.info("[Agent] end of {0} tests from {1} to {2} phase\n\n".format(self.args.running_times, phase_list[self.args.start_phase - 1], phase_list[self.args.end_phase - 1]))

if __name__ == "__main__":
    my_agent = Agent(args)
    my_agent.process()