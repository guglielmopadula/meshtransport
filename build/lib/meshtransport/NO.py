from torch import nn
import torch
import torch_sparse
from .utils import compute_list












class KNeighBallChanger(nn.Module):

    def __init__(self):
        super(KNeighBallChanger, self).__init__()

    def forward(self, x, inp_positions, out_positions,radius=0.015):
        self.list=torch.tensor(compute_list(inp_positions,out_positions,r=radius))
        y=torch.ones(len(self.list[0]))
        num=torch_sparse.spmm(torch.tensor(self.list),y,len(out_positions),len(inp_positions),x.reshape(-1,inp_positions.shape[0],1)).reshape(-1,out_positions.shape[0])
        dem=torch_sparse.spmm(torch.tensor(self.list),y,len(out_positions),len(inp_positions),torch.ones_like(x.reshape(-1,inp_positions.shape[0],1))).reshape(-1,out_positions.shape[0])
        dem=torch.where(dem>0,dem,1.)
        return num/dem







class KIWDBallChanger(nn.Module):

    def __init__(self):
        super(KIWDBallChanger, self).__init__()

    def forward(self, x,inp_positions,out_positions,radius=0.015):
        self.list=torch.tensor(compute_list(inp_positions,out_positions,r=radius))
        y=torch.ones(len(self.list[0]),3)
        y=y+inp_positions[self.list[1]]
        y=y-out_positions[self.list[0]]
        y=1/torch.linalg.norm(y,dim=1)
        num=torch_sparse.spmm(self.list,y,len(out_positions),len(inp_positions),x.reshape(-1,inp_positions.shape[0],1)).reshape(-1,out_positions.shape[0])
        dem=torch_sparse.spmm(self.list,y,len(out_positions),len(inp_positions),torch.ones_like(x.reshape(-1,inp_positions.shape[0],1))).reshape(-1,out_positions.shape[0])
        dem=torch.where(dem>0,dem,1.)
        return num/dem

