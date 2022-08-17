import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import os
import copy
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, label_size, batch_size, use_gpu):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.device = "cuda" if use_gpu else "cpu"

        # self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers = 1, batch_first = True)
        self.lstm = nn.GRU(embedding_dim, hidden_dim, num_layers = 1, bidirectional=True, batch_first = True)
        self.hidden2label = nn.Linear(hidden_dim*2, label_size)
        self.num_layers = 1
        self.hidden = self.init_hidden()
        

    def init_hidden(self):
        if self.use_gpu:
            # h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            # c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            return torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim, device=self.device)
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

    # def init_hidden(self):
    #     if self.gpu:
    #         if isinstance(self.bigru, nn.LSTM):
    #             h0 = torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim, device=self.device)
    #             c0 = torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim, device=self.device)
    #             return h0, c0
    #         return torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim, device=self.device)
    #     else:
    #         return torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim, device=self.device)

    def forward(self, x):
        gru_out, self.hidden = self.lstm(x, self.hidden)
        # print('liner = ',lstm_out[:, -1,:].size())
        # res = lstm_out.size()
        # print('lstm_out shape = ', lstm_out.size())
        # lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        # print('lstm_out shape = ', lstm_out.size())
        # y  = self.hidden2label(lstm_out)
        # y  = self.hidden2label(lstm_out[:, -1,:])
        gru_out = torch.transpose(gru_out, 1, 2)
        # pooling
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        # linear
        y = self.hidden2label(gru_out)
        return y


