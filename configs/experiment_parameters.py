import argparse


class OnlineTDExperiment(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('--dataset_path', type=str, default="data-prep/processed/processed_data.hdf5",
                            help='path to the hd5f dataset file')
        self.add_argument('--sensor', nargs='+', type=str, default=["PIT101"],
                            help='the sensor to train for, e.g: PIT101')
        self.add_argument('--gamma', nargs='+', type=float, default=[0.99],
                            help='the value of gamma')
        self.add_argument('--epochs',  nargs='+', type=int, default=[5],
                            help='number of epochs')
        self.add_argument('--lr_offline',  nargs='+', type=float, default=[0.001],
                            help='learning rate')
        self.add_argument('--lr_online',  nargs='+', type=float, default=[0.001],
                            help='learning rate')
        self.add_argument('--l2lambda',  nargs='+', type=float, default=[1e-3],
                            help='l2lambda')
        self.add_argument('--load_mem', type=int, default=0,
                            help='whether we should load the entire dataset into the memory')
        self.add_argument('--nthreads', type=int, default=4,
                            help='number of threads for torch')
        self.add_argument('--shuffle',  nargs='+', type=int, default=[1],
                            help='whether data should be shuffled in the offline phase')
        self.add_argument('--batch_size',  nargs='+', type=int, default=[512],
                            help='batch size for offline phase')
        self.add_argument('--training_files',  nargs='+', type=str, default=["0-3"],
                            help='file ranges used for training')
        self.add_argument('--val_files',  nargs='+', type=str, default=["4"],
                            help='file ranges used for validation')
        self.add_argument('--testing_files',  nargs='+', type=str, default=["5"],
                            help='file ranges used for testing')
        self.add_argument('--hidden_layers',  nargs='+', type=int, default=2,
                            help='number of hidden layers')
        self.add_argument('--hidden_size',  nargs='+', type=int, default=512,
                            help='size of hidden layers')
        self.add_argument('--replay_steps',  nargs='+', type=int, default=0,
                            help='Number of replay iterations per time step of training')
        self.add_argument('--replay_buffer_size',  nargs='+', type=int, default=10,
                            help='Number of replay iterations per time step of training')
        self.add_argument('--device', type=str, default="cuda",
                            help='device')
        self.add_argument('--name', help='Name of experiment', default="online_td_1")
        self.add_argument('--saved_model_name', help='Name of the experiment which contains the saved model', default="online_td_1")
        self.add_argument('--seed', nargs='+', type=int, default=[0], help='Value of the seed')
        self.add_argument('--run', type=int,
                          help='Run number to index the right experiment. This parameter must be specified', default=0)

class LinearDExperiment(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('--dataset_path', type=str, default="data-logs/processed_data.hdf5",
                            help='path to the hd5f dataset file')
        self.add_argument('--sensor', nargs='+', type=str, default=["PIT101"],
                            help='the sensor to train for, e.g: PIT101')
        self.add_argument('--epochs',  nargs='+', type=int, default=[5],
                            help='number of epochs')
        self.add_argument('--lr_offline',  nargs='+', type=float, default=[0.001],
                            help='learning rate')
        self.add_argument('--lr_online',  nargs='+', type=float, default=[0.001],
                            help='learning rate')
        self.add_argument('--l2lambda',  nargs='+', type=float, default=[1e-3],
                            help='l2lambda')
        self.add_argument('--load_mem', type=int, default=0,
                            help='whether we should load the entire dataset into the memory')
        self.add_argument('--nthreads', type=int, default=4,
                            help='number of threads for torch')
        self.add_argument('--shuffle',  nargs='+', type=int, default=[1],
                            help='whether data should be shuffled in the offline phase')
        self.add_argument('--batch_size',  nargs='+', type=int, default=[512],
                            help='batch size for offline phase')
        self.add_argument('--training_files',  nargs='+', type=str, default=["0-3"],
                            help='file ranges used for training')
        self.add_argument('--val_files',  nargs='+', type=str, default=["4"],
                            help='file ranges used for validation')
        self.add_argument('--testing_files',  nargs='+', type=str, default=["5"],
                            help='file ranges used for testing')
        self.add_argument('--hidden_layers',  nargs='+', type=int, default=2,
                            help='number of hidden layers')
        self.add_argument('--hidden_size',  nargs='+', type=int, default=512,
                            help='size of hidden layers')
        self.add_argument('--device', type=str, default="cuda",
                            help='device')
        self.add_argument('--prediction_horizon',  nargs='+', type=int, default=100,
                            help='the horizon of prediction (the n in n-step methods)')
        self.add_argument('--lookback_window_size',  nargs='+', type=int, default=100,
                            help='the number of steps of past states provided as context to the model at every step')
        self.add_argument('--name', help='Name of experiment', default="online_td_1")
        self.add_argument('--saved_model_name', help='Name of the experiment which contains the saved model', default="online_td_1")
        self.add_argument('--seed', nargs='+', type=int, default=[0], help='Value of the seed')
        self.add_argument('--run', type=int,
                          help='Run number to index the right experiment. This parameter must be specified', default=0)
