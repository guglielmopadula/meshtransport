import numpy as np
from scipy.stats import special_ortho_group
import scipy.spatial.distance
from tqdm import trange

def generate_uniform_box_points(v_max,v_min,N):
    points=np.zeros((N,N,N,3))
    for i in trange(N):
        for j in range(N):
            for k in range(N):
                points[i,j,k]=np.array([v_min[0]+(v_max[0]-v_min[0])*i/(N-1),v_min[1]+(v_max[1]-v_min[1])*j/(N-1),v_min[2]+(v_max[2]-v_min[2])*k/(N-1)])

    return points.reshape(-1,3)


def find_minimuum_bounding_box(points,Ntimes):
    dist_best=np.inf
    v_m_best=np.zeros(3)
    v_M_best=np.zeros(3)
    R_best=np.zeros((3,3))
    for _ in trange(Ntimes):
        R=special_ortho_group.rvs(3)
        points_rotated=(R@points.T).T
        v_min=np.min(points_rotated,axis=0)
        v_max=np.max(points_rotated,axis=0)
        grid_points=generate_uniform_box_points(v_max,v_min,100)
        dist=scipy.spatial.distance.directed_hausdorff(grid_points,points_rotated)[0]
        if dist<dist_best:
            dist_best=dist
            v_M_best=v_max
            v_m_best=v_min
            R_best=R
    tmp_max=R_best.T@v_M_best
    tmp_min=R_best.T@v_m_best
    return np.minimum(tmp_max,tmp_min),np.maximum(tmp_max,tmp_min)

def compute_list(inp_positions,out_positions,r=0.015):
    tot=0
    tree=scipy.spatial.cKDTree(inp_positions)
    for i in range(len(out_positions)):
        l=tree.query_ball_point(out_positions[i],r=r)
        tot=tot+len(l)
    list=np.zeros((2,tot),dtype=np.int64)
    acc=0
    for i in range(len(out_positions)):
        l=tree.query_ball_point(out_positions[i],r=r)
        for j in range(len(l)):
            list[0,acc]=i
            list[1,acc]=l[j]
            acc=acc+1
    return list
