import h5py
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import configs.experiment_parameters as reg_parser

from copy import deepcopy
from models import FFN
from dataloader import get_sequential_dataloader, get_batch_dataloader
from experiment import ExperimentManager, Metric
from utils import trainers, utils
from datetime import timedelta
from timeit import default_timer as timer

COMPUTE_CANADA_USERNAME = "hshah1"

p = reg_parser.OnlineTDExperiment()
all_args = vars(p.parse_known_args()[0])
args = utils.get_run(all_args, p.parse_known_args()[0].run)
exp = ExperimentManager(args["name"], args, COMPUTE_CANADA_USERNAME)

# create database tables
train_error_logger = Metric("train_error", {"run": 0, "epoch": 0, "error": 0.0}, ("run", "epoch"), exp)
val_error_logger = Metric("val_error", {"run": 0, "epoch": 0, "error": 0.0}, ("run", "epoch"), exp)
test_error_logger = Metric("test_error", {"run": 0, "epoch": 0, "test_idx": 0, "step": 0, "error": 0.0},
                           ("run", "test_idx", "epoch", "step"), exp)
nmse_logger = Metric("nmse_table", {"run": 0, "epoch": 0, "train_nmse": 0.0, "val_nmse": 0.0}, ("run", "epoch"), exp)
test_nmse_logger = Metric("test_nmse_table", {"run": 0, "epoch": 0, "test_idx": 0, "nmse": 0.0}, ("run", "epoch", "test_idx"), exp)

test_predictions_logger = Metric("test_predictions",
                                 {"run": 0, "epoch": 0, "test_idx": 0, "step": 0, "prediction": 0.0, "returns": 0.0,
                                  "cummulant": 0.0}, ("run", "test_idx","epoch", "step"), exp)
summary_logger = Metric("summary_table", {"run": 0, "epoch": 0, "train_error": 0.0, "val_error": 0.0, "test_error": 0.0,
                                          "train_nmse": 0.0, "val_nmse": 0.0, "test_nmse": 0.0, "time_taken": 0.0},
                        ("run", "epoch"), exp)

start = timer()
utils.set_seed(args['seed'])
torch.set_num_threads(args['nthreads'])

# load data and compute returns
tag = f"ZW_{args['sensor'].upper()}.DATA"
data = h5py.File(f"{args['dataset_path']}", 'r')
if args['load_mem']:
    data = utils.load_entire_hdf5(data)

cummulant_idx = np.where(data['labels'][:] == tag.encode("ascii"))[0][0]
cummulants = data['features'][:, cummulant_idx]
returns = utils.compute_returns(cummulants, gamma=args['gamma'])
# load splits
val_data = utils.get_split(data, args['val_files'], cummulants, returns)
test_data = utils.get_split(data, args['testing_files'], cummulants, returns)

val_dataloaders = get_sequential_dataloader(val_data,
                                            num_workers=args['nthreads'])

test_dataloaders = get_sequential_dataloader(test_data,
                                             num_workers=args['nthreads'])

val_replay_data = test_replay_data = None
if args['replay_buffer_size'] and args['replay_steps']:
    print("using replay buffer")
    val_replay_data = utils.get_replay_split(data, args['training_files'], cummulants, returns, args['val_files'], args['replay_buffer_size']) 
    test_replay_data = utils.get_replay_split(data, args['val_files'], cummulants, returns, args['testing_files'], args['replay_buffer_size']) 

# setup models and optim
model = FFN(val_data[0][0].shape[0], args['hidden_layers'], args['hidden_size']).to(args['device'])
evaluation_model = deepcopy(model)
optimizer = optim.Adam(model.parameters(), lr=args['lr_offline'], weight_decay=args['l2lambda'])
evaluation_optim = optim.Adam(evaluation_model.parameters(), lr=args['lr_online'], weight_decay=args['l2lambda'])
criterion = nn.MSELoss()

# load saved model
path_to_model = f"saved_models/{args['saved_model_name']}/{args['sensor']}/lr_{args['lr_offline']}.pt"
print(f"Loading the model from {path_to_model}")
checkpoint = torch.load(path_to_model, map_location=torch.device(args['device']))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.train()

eval_error_avg = test_error_avg = 0.0
eval_nmse_score = test_nmse_score = 0.0

# OnlineTD algorithm

# evaluation on validation set
print("Evaluating on Validation set...")
evaluation_model.load_state_dict(model.state_dict())
evaluation_optim.load_state_dict(optimizer.state_dict())
for param_group in evaluation_optim.param_groups:
    param_group['lr'] = args['lr_online']
evaluation_model.train()
eval_preds, _, eval_error_avg = trainers.train_epoch_with_replay(evaluation_model, val_dataloaders[0],
                                                                 evaluation_optim, criterion, val_replay_data, 
                                                                 args, log_every=-1)
eval_nmse_score = utils.compute_scores(returns[val_data.starting_idx + 1: val_data.ending_idx],
                                       eval_preds[:-1],
                                       val_data.return_variance)

print("Val NMSE: ", eval_nmse_score)
nmse_logger.add_data([exp.run, -1, -1.0, eval_nmse_score])
val_error_logger.add_data([exp.run, -1, eval_error_avg])
val_error_logger.commit_to_database()
nmse_logger.commit_to_database()

# evaluation on test set
test_error_avg = 0
test_nmse_scores = []
for test_idx, test_dataloader in enumerate(test_dataloaders):
    print("Evaluating on test set #", test_idx, " ranges: ", args['testing_files'].split(",")[test_idx])
    evaluation_model.load_state_dict(model.state_dict())
    evaluation_optim.load_state_dict(optimizer.state_dict())
    for param_group in evaluation_optim.param_groups:
        param_group['lr'] = args['lr_online']
    evaluation_model.train()
    test_preds, test_errors, test_error_avg = trainers.train_epoch_with_replay(evaluation_model, test_dataloader,
                                                                               evaluation_optim,
                                                                               criterion, test_replay_data, args, log_every=1)
    test_nmse_score = utils.compute_scores(returns[test_dataloader.dataset.starting_idx + 1:test_dataloader.dataset.ending_idx],
                                           test_preds[:-1],
                                           test_dataloader.dataset.return_variance)
    print("Test NMSE: ", test_nmse_score)
    test_nmse_logger.add_data([exp.run, -1, test_idx, test_nmse_score])
    test_nmse_scores.append(test_nmse_score)
    for step in range(len(test_preds)):
        test_error_logger.add_data([exp.run, -1, test_idx, step, test_errors[step]])
        test_predictions_logger.add_data([exp.run, -1, test_idx, step, test_preds[step].item(),
                                          float(returns[test_dataloader.dataset.starting_idx + step + 1]),
                                          float(cummulants[test_dataloader.dataset.starting_idx + step])])
    test_error_logger.commit_to_database()
    test_predictions_logger.commit_to_database()
    test_nmse_logger.commit_to_database()

summary_logger.add_data([exp.run, 0, -1, eval_error_avg, test_error_avg, -1, eval_nmse_score,
                         float(np.mean(test_nmse_scores)), timedelta(seconds=timer() - start).seconds / 60])

val_error_logger.commit_to_database()
nmse_logger.commit_to_database()
test_nmse_logger.commit_to_database()
test_error_logger.commit_to_database()
test_predictions_logger.commit_to_database()
summary_logger.commit_to_database()
print("Total time taken: ", timedelta(seconds=timer() - start).seconds / 60, " minutes")
