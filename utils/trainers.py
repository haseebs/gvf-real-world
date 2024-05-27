import torch
from tqdm import tqdm


def train_epoch(model, dataloader, optimizer, criterion, args, frozen=False, log_every=1000):
    """
    OnlineTD algorithm
    to be deleted, the replay version can replace this
    """
    model.train()
    predictions = []
    errors = []
    error_avg = -1
    # train offline on training set
    for step, (state, next_state, returns, cummulants) in tqdm(enumerate(dataloader), total=len(dataloader)):
        pred = model(state.to(args['device'])).squeeze()
        target = None
        with torch.no_grad():
            next_pred = model(next_state.to(args['device'])).squeeze()
            target = cummulants.to(args['device']) + args['gamma'] * next_pred
            predictions.append(next_pred.detach().to('cpu').numpy())
        loss = criterion(pred, target)
        if not frozen:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if error_avg == -1:
            error_avg = loss.detach().item()
        else:
            error_avg = error_avg * 0.995 + loss.detach().item() * 0.005

        if step % log_every == log_every - 1:
            errors.append(loss.detach().item())
    return predictions, errors, error_avg


def train_epoch_using_return(model, dataloader, optimizer, criterion, args, frozen=False, log_every=1000):
    model.train()
    predictions = []
    errors = []
    error_avg = -1
    # train offline on training set
    for step, (state, next_state, returns, cummulants) in tqdm(enumerate(dataloader), total=len(dataloader)):
        pred = model(state.to(args['device'])).squeeze()
        target = returns.to(args['device']).float()
        #if step == 50000:
        #    from IPython import embed; embed()
        #    exit()
        with torch.no_grad():
            next_pred = model(next_state.to(args['device'])).squeeze()
            predictions.append(next_pred.detach().to('cpu').numpy())
        loss = criterion(pred, target)
        if not frozen:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if error_avg == -1:
            error_avg = loss.detach().item()
        else:
            error_avg = error_avg * 0.995 + loss.detach().item() * 0.005

        if step % log_every == log_every - 1:
            errors.append(loss.detach().item())
    return predictions, errors, error_avg


def train_epoch_with_replay(model, dataloader, optimizer, criterion, replay_buffer, args, frozen=False, log_every=1000):
    """
    TDReplay algorithm from the paper
    """
    model.train()
    predictions = []
    errors = []
    error_avg = -1
    for step, (state, next_state, returns, cummulants) in tqdm(enumerate(dataloader), total=len(dataloader)):
        pred = model(state.to(args['device'])).squeeze()
        target = None
        with torch.no_grad():
            next_pred = model(next_state.to(args['device'])).squeeze()
            target = cummulants.to(args['device']) + args['gamma'] * next_pred
            predictions.append(next_pred.detach().to('cpu').numpy())
        loss = criterion(pred, target)
        if not frozen:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if error_avg == -1:
            error_avg = loss.detach().item()
        else:
            error_avg = error_avg * 0.995 + loss.detach().item() * 0.005

        if step % log_every == log_every - 1:
            errors.append(loss.detach().item())

        for replay_step in range(args['replay_steps']):
            state, next_state, returns, _, cummulants = replay_buffer.sample(args['batch_size'])
            pred = model(state.to(args['device'])).squeeze()
            target = None
            with torch.no_grad():
                next_pred = model(next_state.to(args['device'])).squeeze()
                target = cummulants.to(args['device']) + args['gamma'] * next_pred
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if args['replay_buffer_size'] > 0:
            replay_buffer.increment_buffer(1, dataloader.dataset.starting_idx + step)
    return predictions, errors, error_avg


def train_epoch_nstep_offline(model, dataloader, optimizer, criterion, args, frozen=False, log_every=1000):
    model.train()
    predictions = []
    errors = []
    error_avg = -1
    # train offline on training set
    for step, (state, _, returns, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        pred = model(state.to(args['device'])).squeeze()
        target = returns.to(args['device']).float()
        predictions.append(pred.detach().to('cpu').numpy()) #note that this is not next_pred but pred here
        loss = criterion(pred, target)
        if not frozen:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if error_avg == -1:
            error_avg = loss.detach().item()
        else:
            error_avg = error_avg * 0.995 + loss.detach().item() * 0.005

        if step % log_every == log_every - 1:
            errors.append(loss.detach().item())
    return predictions, errors, error_avg


def train_epoch_nstep_online(model, dataloader, optimizer, criterion, args, frozen=False, log_every=1000, nstep_n=100):
    """
    online n-step algorithm from the paper
    """
    # dont use this function with batch_size != 1 lol
    model.train()
    predictions = []
    errors = []
    past_states = []
    error_avg = -1
    # train offline on training set
    for step, (state, _, _, cummulants) in tqdm(enumerate(dataloader), total=len(dataloader)):
        past_states.append(state)
        pred_to_store = model(state.unsqueeze(0).to(args['device'])).squeeze().detach().to('cpu').numpy()
        predictions.append(pred_to_store)
        if step >= nstep_n:
            old_state = past_states[step - nstep_n]
            pred = model(old_state.unsqueeze(0).to(args['device'])).squeeze()
            target = cummulants.to(args['device'])
            loss = criterion(pred, target)
            if not frozen:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if error_avg == -1:
                error_avg = loss.detach().item()
            else:
                error_avg = error_avg * 0.995 + loss.detach().item() * 0.005

            if step % log_every == log_every - 1:
                errors.append(loss.detach().item())
            past_states[step-nstep_n] = None # free the memory
    return predictions, errors, error_avg
