from ast import literal_eval
import pickle
import os 
import numpy as np
import pandas as pd
import multiprocessing as mp
import sys
import random as rd
import torch as t

from pymatgen.core import lattice

from pymatgen.core.structure import Structure

#Parallel 
def parallel_Featurize(featureFunc, s_list, n_jobs):
    pool = mp.Pool(processes=n_jobs)
    Feature_list = pool.starmap_async(func=featureFunc, iterable=s_list).get()
    pool.close()
    pool.join()
    return Feature_list

def parallel_single(Func,s_list,n_jobs):
    pool = mp.Pool(processes=n_jobs)
    Feature_list = pool.map_async(func=Func, iterable=s_list).get()
    pool.close()
    pool.join()
    return Feature_list
    
#x: AR, MB, Go, Mu, MV
def loadDictionary(df, str):
    labels = df.columns
    if str in labels and 'element' in labels:
        elements = df['element'].values
        x = df[str].values
    else:
        print('ERROR input csv')
        exit(0)
    x_dict = {}
    nElements = len(elements)
    for i in range(nElements):
        x_dict[elements[i]] = x[i]
    return x_dict

def loadmodel(name:str):
    with open(os.path.join(os.getcwd(), 'formation_energy', 'methods', name + '.pickle'), 'rb') as f:
        model = pickle.load(f)

    return model

def get_value_electrons(full_e_structure):
    outside=full_e_structure[-1]
    inside=full_e_structure[-2]
    total_number=outside[-1]+inside[-1]
    return total_number

def dump_error_structure(s):
    string=s.composition.reduced_formula
    filename=string+'.cif'
    s.to(filename=filename)

def site_property(s,p_key,condition):
    '''
    s: pymatgen structure
    p_key: site_properties is a dictionary
    '''
    try:
        L=s.site_properties[p_key]
    except KeyError:
        print('this structure has no key, use all sites')
        return None
    else:
        indexs=[]
        for i in range(len(L)):
            if L[i]==condition:
                indexs.append(i)
            else:
                continue
        return indexs

def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return t.mean(t.abs(target - prediction))

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor, dim=None):
        """tensor is taken as a sample to calculate the mean and std"""
        if dim is None:
            self.mean = t.mean(tensor)
            self.std = t.std(tensor)
        else:
            self.mean = t.mean(tensor,dim)
            self.std = t.std(tensor,dim)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


if __name__=='__main__':
    df=pd.read_csv('testdata\\test.csv')
    S=df['structures'][1]
    S=Structure.from_dict(literal_eval(S))
    s3=S[3]
    print(S.index(s3))