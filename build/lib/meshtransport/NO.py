from torch import nn
import torch
import torch_sparse
from .utils import compute_list
from .utils import topyg
from torch_geometric.nn.conv import NNConv
from torch_geometric.utils import to_dense_batch











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


class KernelChanger(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(KernelChanger, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.nn=nn.Sequential(nn.Linear(6,100),nn.ReLU(),nn.Linear(100,100),nn.ReLU(),nn.Linear(100,100),nn.ReLU(),nn.Linear(100,in_channels*out_channels))
        self.nn_conv=NNConv(in_channels,out_channels,self.nn,aggr="mean")
    
    def forward(self,x,inp_positions,out_positions,radius=0.015):
        num_in_pos=inp_positions.shape[0]
        num_out_pos=out_positions.shape[0]
        mylist=torch.tensor(compute_list(inp_positions,out_positions,r=radius))
        mylist_values=torch.cat([out_positions[mylist[0]],inp_positions[mylist[1]]],dim=1)
        mylist[0]=mylist[0]+num_in_pos
        x=torch.cat((x.reshape(x.shape[0],num_in_pos,self.in_channels),torch.zeros(x.shape[0],num_out_pos,self.in_channels)),dim=1)
        batch=topyg(x,mylist,mylist_values)
        x=self.nn_conv(batch.x,batch.edge_index,batch.edge_attr)
        x=x.reshape(-1,num_in_pos+num_out_pos,self.out_channels)
        return x[:,num_in_pos:]
    
