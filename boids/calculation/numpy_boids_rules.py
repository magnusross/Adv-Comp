import numpy as np 
import matplotlib.pyplot as plt
import numba 

def rule_com(i, pos_all, vel_all, factor=0.01):
    
    com = np.mean(np.delete(pos_all, i, axis=0), axis=0)

    return (com  - pos_all[i]) * factor 

def rule_avoid(i, pos_all, vel_all, radius=1):
    pos_other = np.delete(pos_all, i, axis=0)
    rel_pos = pos_other - pos_all[i]
    in_region = np.sqrt(rel_pos[:, 0]**2 + rel_pos[:, 1]**2) < 10
    
    return np.sum(rel_pos * in_region.reshape(-1, 1), axis=0)

def rule_match(i, pos_all, vel_all, factor=0.01):
   
    v_mean =  np.mean(np.delete(vel_all, i, axis=0), axis=0) 

    return (v_mean - vel_all[i]) * factor
    
