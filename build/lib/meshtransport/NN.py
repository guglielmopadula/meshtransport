from torch import nn
import torch
import torch_sparse
from .utils import compute_list



class ContinousConvolution(nn.Module):

    def __init__(self,inp_positions,out_positions):
        super(ContinousConvolution, self).__init__()
        self.alpha=nn.Parameter(torch.ones((1,len(inp_positions))),requires_grad=True)
        self.alpha=torch.nn.init.xavier_uniform_(self.alpha)
        self.sigma=nn.Parameter(torch.ones(len(inp_positions)),requires_grad=True)
        self.inp_positions=inp_positions
        self.out_positions=out_positions
        self.list=torch.tensor(compute_list(inp_positions,out_positions))

    def forward(self, x):
        y=torch.zeros(len(self.list[0]),3)
        y=y+self.inp_positions[self.list[1]]
        y=y-self.out_positions[self.list[0]]
        y=-torch.linalg.norm(y,dim=1)**2
        y=y/(self.sigma[self.list[1]]**2)
        y=torch.exp(y)
        y=torch_sparse.spmm(self.list,y,len(self.out_positions),len(self.inp_positions),(self.alpha*x).reshape(-1,self.inp_positions.shape[0],1)).reshape(-1,self.out_positions.shape[0]) 
        return y
    


