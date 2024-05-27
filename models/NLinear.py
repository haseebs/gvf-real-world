import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NLinear(nn.Module):
    """
    Normalization-Linear
    sequence_length: lookback window size + 1
    prediction_idx: index for the feature that we are predicting
    independent: whether each variable is treated independently, like they do in time-series work
    """
    def __init__(self, embedding_dim, sequence_length, prediction_idx, independent=False):
        super(NLinear, self).__init__()
        self.seq_len = sequence_length
        self.embedding_dim = embedding_dim
        self.prediction_idx = prediction_idx
        self.independent = independent

        if independent:
            self.Linear = nn.Linear(self.seq_len, 1)
        else:
            self.Linear = nn.Linear(self.seq_len * self.embedding_dim, 1)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        if self.independent:
            x = x[:,:,self.prediction_idx]
            seq_last = x[:,-1:].detach()
        else:
            seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        x = x.flatten(1) # merge everything but the batch dimension
        x = self.Linear(x)
        if self.independent:
            x = x + seq_last
        else:
            x = x + seq_last[:,:,self.prediction_idx] # add the value back for only the pred idx since 1 output
        return x.squeeze() # [Batch] 


#class NLinear(nn.Module):
#    """
#    Normalization-Linear
#    sequence_length: lookback window size + 1
#    individual: if False: a single linear layer will be shared for each dim of embedding
#    """
#    def __init__(self, embedding_dim, sequence_length, individual=True):
#        super(NLinear, self).__init__()
#        self.seq_len = sequence_length
#        self.embedding_dim = embedding_dim
#        self.individual = individual
#
#        if self.individual:
#            self.Linear = nn.ModuleList()
#            for i in range(self.embedding_dim):
#                self.Linear.append(nn.Linear(self.seq_len,1))
#        else:
#            self.Linear = nn.Linear(self.seq_len, 1)
#
#    def forward(self, x):
#        # x: [Batch, Input length, Channel]
#        seq_last = x[:,-1:,:].detach()
#        x = x - seq_last
#        if self.individual:
#            output = torch.zeros([x.size(0),1,x.size(2)],dtype=x.dtype).to(x.device)
#            for i in range(self.embedding_dim):
#                output[:,:,i] = self.Linear[i](x[:,:,i])
#            x = output
#        else:
#            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
#        x = x + seq_last
#        return x # [Batch, Output length, Channel]
