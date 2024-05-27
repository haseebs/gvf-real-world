import h5py
import numpy as np

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
test_error_logger = Metric("test_error", {"run": 0, "test_idx": 0, "step": 0, "error": 0.0},
                           ("run", "test_idx", "step"), exp)
nmse_logger = Metric("nmse_table", {"run": 0, "epoch": 0, "train_nmse": 0.0, "val_nmse": 0.0}, ("run", "epoch"), exp)
test_nmse_logger = Metric("test_nmse_table", {"run": 0, "test_idx": 0, "nmse": 0.0}, ("run", "test_idx"), exp)

test_predictions_logger = Metric("test_predictions",
                                 {"run": 0, "test_idx": 0, "step": 0, "prediction": 0.0, "returns": 0.0,
                                  "cummulant": 0.0}, ("run", "test_idx", "step"), exp)
summary_logger = Metric("summary_table", {"run": 0, "epoch": 0, "train_error": 0.0, "val_error": 0.0, "test_error": 0.0,
                                          "train_nmse": 0.0, "val_nmse": 0.0, "test_nmse": 0.0, "time_taken": 0.0},
                        ("run", "epoch"), exp)

start = timer()
utils.set_seed(args['seed'])

# load data and compute returns
tag = f"ZW_{args['sensor'].upper()}.DATA"
data = h5py.File(f"{args['dataset_path']}", 'r')
cummulant_idx = np.where(data['labels'][:] == tag.encode("ascii"))[0][0]
cummulants = data['features'][:, cummulant_idx]
returns = utils.compute_returns(cummulants, gamma=args['gamma'])
# load splits
train_data = utils.get_split(data, args['training_files'], cummulants, returns)
val_data = utils.get_split(data, args['val_files'], cummulants, returns)
test_data = utils.get_split(data, args['testing_files'], cummulants, returns)

# prepare various dataloaders from these splits
train_dataloaders = get_batch_dataloader(train_data,
                                         batch_size=args['batch_size'],
                                         num_workers=args['nthreads'],
                                         shuffle=args['shuffle'])
train_dataloaders_non_shuffled = get_batch_dataloader(train_data,
                                                      batch_size=args['batch_size'],
                                                      num_workers=args['nthreads'],
                                                      shuffle=False,
                                                      drop_last=False)
val_dataloaders = get_sequential_dataloader(val_data,
                                            num_workers=args['nthreads'])

test_dataloaders = get_sequential_dataloader(test_data,
                                             num_workers=args['nthreads'])

# setup models and optim
model = FFN(train_data[0][0].shape[0], args['hidden_layers'], args['hidden_size']).to(args['device'])
evaluation_model = deepcopy(model)
optimizer = optim.Adam(model.parameters(), lr=args['lr_offline'], weight_decay=args['l2lambda'])
evaluation_optim = optim.Adam(evaluation_model.parameters(), lr=args['lr_online'], weight_decay=args['l2lambda'])
criterion = nn.MSELoss()
train_error_avg = eval_error_avg = test_error_avg = 0.0
train_nmse_score = eval_nmse_score = test_nmse_score = 0.0

# OnlineTD algorithm
for epoch in range(args['epochs']):
    # train offline on training set
    print("Epoch: ", epoch, " : Training on Training set")
    _, _, train_error_avg = trainers.train_epoch(model, train_dataloaders[0], optimizer, criterion, args,
                                                 log_every=-1)
    train_error_logger.add_data([exp.run, epoch, train_error_avg])

    if epoch % 500 == 0:
        # evaluation on training set
        print("Evaluating on Training set...")
        evaluation_model.load_state_dict(model.state_dict())
        evaluation_optim.load_state_dict(optimizer.state_dict())
        evaluation_model.train()
        train_preds, _, _ = trainers.train_epoch(evaluation_model, train_dataloaders_non_shuffled[0],
                                                 evaluation_optim, criterion, args, frozen=True,
                                                 log_every=-1)
        train_nmse_score = utils.compute_scores(returns[train_data.starting_idx + 1: train_data.ending_idx],
                                                list(np.concatenate(train_preds).flat)[:-1],
                                                train_data.return_variance)

        # evaluation on validation set
        print("Evaluating on Validation set...")
        evaluation_model.load_state_dict(model.state_dict())
        evaluation_optim.load_state_dict(optimizer.state_dict())
        for param_group in evaluation_optim.param_groups:
            param_group['lr'] = args['lr_online']
        evaluation_model.train()
        eval_preds, _, eval_error_avg = trainers.train_epoch(evaluation_model, val_dataloaders[0],
                                                             evaluation_optim, criterion, args, log_every=-1)
        eval_nmse_score = utils.compute_scores(returns[val_data.starting_idx + 1: val_data.ending_idx],
                                               eval_preds[:-1],
                                               val_data.return_variance)

        print("Epoch: ", epoch, " Train NMSE: ", train_nmse_score, "Val NMSE: ", eval_nmse_score)
        nmse_logger.add_data([exp.run, epoch, train_nmse_score, eval_nmse_score])
        val_error_logger.add_data([exp.run, epoch, eval_error_avg])
        train_error_logger.commit_to_database()
        val_error_logger.commit_to_database()
        nmse_logger.commit_to_database()

# evaluation on test set
test_nmse_scores = []
for test_idx, test_dataloader in enumerate(test_dataloaders):
    print("Evaluating on test set #", test_idx, " ranges: ", args['testing_files'].split(",")[test_idx])
    evaluation_model.load_state_dict(model.state_dict())
    evaluation_optim.load_state_dict(optimizer.state_dict())
    for param_group in evaluation_optim.param_groups:
        param_group['lr'] = args['lr_online']
    evaluation_model.train()
    test_preds, test_errors, test_error_avg = trainers.train_epoch(evaluation_model, test_dataloader,
                                                                   evaluation_optim,
                                                                   criterion, args, log_every=1)
    test_nmse_score = utils.compute_scores(returns[test_dataloader.dataset.starting_idx + 1:test_dataloader.dataset.ending_idx],
                                           test_preds[:-1],
                                           test_dataloader.dataset.return_variance)
    print("Test NMSE: ", test_nmse_score)
    test_nmse_logger.add_data([exp.run, test_idx, test_nmse_score])
    test_nmse_scores.append(test_nmse_score)
    for step in range(len(test_preds)):
        test_error_logger.add_data([exp.run, test_idx, step, test_errors[step]])
        test_predictions_logger.add_data([exp.run, test_idx, step, test_preds[step].item(),
                                          float(returns[test_dataloader.dataset.starting_idx + step + 1]),
                                          float(cummulants[test_dataloader.dataset.starting_idx + step])])
    test_nmse_logger.commit_to_database()
    test_error_logger.commit_to_database()
    test_predictions_logger.commit_to_database()

summary_logger.add_data([exp.run, 0, train_error_avg, eval_error_avg, test_error_avg, train_nmse_score, eval_nmse_score,
                         float(np.mean(test_nmse_scores)), timedelta(seconds=timer() - start).seconds / 60])

train_error_logger.commit_to_database()
val_error_logger.commit_to_database()
nmse_logger.commit_to_database()
test_nmse_logger.commit_to_database()
test_error_logger.commit_to_database()
test_predictions_logger.commit_to_database()
summary_logger.commit_to_database()
print("Total time taken: ", timedelta(seconds=timer() - start).seconds / 60, " minutes")
