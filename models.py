import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing,GatedGraphConv,GCNConv,TopKPooling
# from torch_geometric.utils import degree, remove_self_loops, add_self_loops, softmax,scatter_
from torch_geometric.utils import degree, remove_self_loops, add_self_loops, softmax
#from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.glob import GlobalAttention
import sys

class GGNN(torch.nn.Module):
    def __init__(self,vocablen,embedding_dim,num_layers,device):
        super(GGNN, self).__init__()
        self.device=device
        #self.num_layers=num_layers
        self.embed=nn.Embedding(vocablen,embedding_dim)
        self.edge_embed=nn.Embedding(20,embedding_dim)
        #self.gmn=nn.ModuleList([GMNlayer(embedding_dim,embedding_dim) for i in range(num_layers)])
        self.ggnnlayer=GatedGraphConv(embedding_dim,num_layers) 
        self.mlp_gate=nn.Sequential(nn.Linear(embedding_dim,1),nn.Sigmoid())
        self.pool=GlobalAttention(gate_nn=self.mlp_gate)
        self.hidden2label = nn.Linear(embedding_dim, 2)

    def forward(self, data):
        x, edge_index, edge_attr = data
        # print(x.shape)
        x = self.embed(x)
        x = x.squeeze(1)
        print(x.size())
        if type(edge_attr)==type(None):
            edge_weight=None
        else:
            edge_weight=self.edge_embed(edge_attr)
            edge_weight=edge_weight.squeeze(1)
        # edge_attr=torch.div(edge_attr, edge_attr.sum())
        # x = self.ggnnlayer(x, edge_index,edge_attr)  # 增加边的权重
        x = self.ggnnlayer(x, edge_index, edge_attr)  # 增加边的权重
        batch=torch.zeros(x.size(0),dtype=torch.long).to(self.device)
        hg=self.pool(x,batch=batch)
        y  = self.hidden2label(hg)
        return y

class GGNN2(torch.nn.Module):
    def __init__(self,vocablen,embedding_dim,num_layers,device):
        super(GGNN2, self).__init__()
        self.device=device
        #self.num_layers=num_layers
        self.embed=nn.Embedding(vocablen,embedding_dim)
        self.edge_embed=nn.Embedding(20,embedding_dim)
        #self.gmn=nn.ModuleList([GMNlayer(embedding_dim,embedding_dim) for i in range(num_layers)])
        self.ggnnlayer=GatedGraphConv(embedding_dim,num_layers) 
        self.mlp_gate=nn.Sequential(nn.Linear(embedding_dim,1),nn.Sigmoid())
        self.pool=GlobalAttention(gate_nn=self.mlp_gate)
        self.hidden2label = nn.Linear(embedding_dim, 2)

    def forward(self, data):
        x, edge_index, edge_attr = data

        # x = self.embed(x)
        # x = x.squeeze(1)
        # x=self.id_vector
        if type(edge_attr)==type(None):
            edge_weight=None
        else:
            edge_weight=self.edge_embed(edge_attr)
            edge_weight=edge_weight.squeeze(1)
        # edge_attr=torch.div(edge_attr, edge_attr.sum())
        x = self.ggnnlayer(x, edge_index)  # 增加边的权重
        batch=torch.zeros(x.size(0),dtype=torch.long).to(self.device)
        hg=self.pool(x,batch=batch)
        y  = self.hidden2label(hg)
        return y

class GCN(torch.nn.Module):
    def __init__(self,vocablen,embedding_dim,num_class,device):
        super(GCN, self).__init__()
        self.device=device
        #self.num_layers=num_layers
        self.embed=nn.Embedding(vocablen,embedding_dim)
        self.edge_embed=nn.Embedding(20,embedding_dim)
        self.gcnlayer1=GCNConv(embedding_dim, 64)
        self.gcnlayer2=GCNConv(64, 32)
        # self.gcnlayer3=GCNConv(32, 16)
        # self.gcnlayer4=GCNConv(16, 8)

        self.mlp_gate=nn.Sequential(nn.Linear(32,1),nn.Sigmoid())
        self.pool=GlobalAttention(gate_nn=self.mlp_gate)
        self.hidden2label = nn.Linear(32, 2)

    def forward(self, data):
        x, edge_index, edge_attr = data
        x = self.embed(x)
        x = x.squeeze(1)
        if type(edge_attr)==type(None):
            edge_weight=None
        else:
            edge_weight=self.edge_embed(edge_attr)
            edge_weight=edge_weight.squeeze(1)
        x = self.gcnlayer1(x, edge_index)
        x = F.relu(x)
        x = self.gcnlayer2(x, edge_index)
        batch=torch.zeros(x.size(0),dtype=torch.long).to(self.device)
        hg=self.pool(x,batch=batch)
        y  = self.hidden2label(hg)
        return y
