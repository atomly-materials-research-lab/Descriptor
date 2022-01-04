###########################################################################################################
#   This file contains method to featurize feature of local environment,                                  #
#   each structure can be described as a graph, and the descriptors contains information                  #
#   of local environment are generate by graph convlution: Aggragate features of each node(atom) and its  #
#   neighbors, then use an operation 'readout' to aggragate all the nodes to a single value. This method  #
#   is also used in classic Grapy Neural Network: Neural Network for Grapy(NN4G)                          #
###########################################################################################################
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Structure
from pymatgen.core import periodic_table as pt 
from ast import literal_eval

import numpy as np
import scipy.stats as st

import os
import pandas as pd

from .utils import loadDictionary,parallel_Featurize,dump_error_structure,site_property
from tqdm import tqdm

oliynyk_dir=os.path.join(os.getcwd(),'Descriptor','oliynyk.csv')

class StructureNeighbors:
    '''
    site_index_list: specify the index of sites to find nn, if it is None, all the site are calculate
    '''
    def __init__(self,site_index_lst=None,element=None):
        self.if_error=False
        self.specify_element=element
        self.site_index=site_index_lst
        self.elements=[]
        self.elements_coords=[]
        self.neighbors_lst=[]
        self.neighbors_coords=[]
    
    def is_empty(self):
        if len(self.elements)==0 or len(self.neighbors_lst)==0:
            return False
        elif self.if_error==True:
            return False
        else:
            return True
    
    def get_all_neighbors_by_cutoff(self,s,cutoff=None):
        if cutoff is None:
            SUM=0
            for Site in s:
                try:
                    SUM+=Site.species.elements[0].atomic_radius
                except TypeError:
                    SUM+=1
                finally:
                    pass
            avg_atom_radius=SUM/len(s)
            radius=3.6*avg_atom_radius
        else:
            radius=cutoff
        if self.site_index is None:
            if self.specify_element is None:
                self.elements=[site.species.elements[0].symbol for site in s]
                self.elements_coords=[site.coords for site in s]
                neighbor_lst=s.get_all_neighbors(r=radius,include_index=True)
                #sorted by distance
                neighbor_lst=[sorted(nbrs, key=lambda x: x[1]) for nbrs in neighbor_lst]
                #Just get the nearest 12 neighbor
                #check if any error
                for n in neighbor_lst:
                    #Just get the nearest 12 neighbor
                    if len(n)==0:
                        self.if_error=True
                        self.neighbors_lst.append([])
                        self.neighbors_coords.append([])
                    elif len(n)>12:
                        self.neighbors_lst.append([site.species.elements[0].symbol for site in n[:12]])
                        self.neighbors_coords.append([site.coords for site in n[:12]])
                    else:
                        self.neighbors_lst.append([site.species.elements[0].symbol for site in n])
                        self.neighbors_coords.append([site.coords for site in n])
            else:
                site_lst=[]
                for site in s:
                    if site.species.elements[0].symbol==self.specify_element:
                        site_lst.append(site)
                if len(site_lst)==0:
                    self.if_error=True
                    pass
                else:
                    self.elements=[site.species.elements[0].symbol for site in site_lst]
                    self.elements_coords=[site.coords for site in site_lst]
                    neighbor_lst=s.get_all_neighbors(r=radius,sites=site_lst,include_index=True)
                    #sorted by distance
                    neighbor_lst=[sorted(nbrs, key=lambda x: x[1]) for nbrs in neighbor_lst]
                    #check if any error
                    for n in neighbor_lst:
                        if len(n)==0:
                            self.if_error=True
                            self.neighbors_lst.append([])
                            self.neighbors_coords.append([])
                        elif len(n)>12:
                            self.neighbors_lst.append([site.species.elements[0].symbol for site in n[:12]])
                            self.neighbors_coords.append([site.coords for site in n[:12]])
                        else:
                            self.neighbors_lst.append([site.species.elements[0].symbol for site in n])
                            self.neighbors_coords.append([site.coords for site in n])
        else:
            if self.specify_element is None:
                site_lst=[]
                for index in self.site_index:
                    site_lst.append(s[index])
                self.elements=[site.species.elements[0].symbol for site in site_lst]
                self.elements_coords=[site.coords for site in site_lst]
                neighbor_lst=s.get_all_neighbors(r=radius,sites=site_lst,include_index=True)
                #sorted by distance
                neighbor_lst=[sorted(nbrs, key=lambda x: x[1]) for nbrs in neighbor_lst]
                #check if any error
                for n in neighbor_lst:
                    if len(n)==0:
                        self.if_error=True
                        self.neighbors_lst.append([])
                        self.neighbors_coords.append([])
                    elif len(n)>12:
                        self.neighbors_lst.append([site.species.elements[0].symbol for site in n[:12]])
                        self.neighbors_coords.append([site.coords for site in n[:12]])
                    else:
                        self.neighbors_lst.append([site.species.elements[0].symbol for site in n])
                        self.neighbors_coords.append([site.coords for site in n])
            else:
                site_lst=[]
                for index in self.site_index:
                    if s[index].species.elements[0].symbol==self.specify_element:
                        site_lst.append(s[index])
                    else:
                        continue
                if len(site_lst)==0:
                    self.if_error=True
                    pass
                else:
                    self.elements=[site.species.elements[0].symbol for site in site_lst]
                    self.elements_coords=[site.coords for site in site_lst]
                    neighbor_lst=s.get_all_neighbors(r=radius,sites=site_lst,include_index=True)
                    #sorted by distance
                    neighbor_lst=[sorted(nbrs, key=lambda x: x[1]) for nbrs in neighbor_lst]
                    #check if any error
                    for n in neighbor_lst:
                        if len(n)==0:
                            self.if_error=True
                            self.neighbors_lst.append([])
                            self.neighbors_coords.append([])
                        elif len(n)>12:
                            self.neighbors_lst.append([site.species.elements[0].symbol for site in n[:12]])
                            self.neighbors_coords.append([site.coords for site in n[:12]])
                        else:
                            self.neighbors_lst.append([site.species.elements[0].symbol for site in n])
                            self.neighbors_coords.append([site.coords for site in n])

    
    def get_site_neighbors(self,s,i):
        nn=CrystalNN(distance_cutoffs=None,x_diff_weight=0.0,porous_adjustment=False)
        try:
            nn_list_per_atom=nn.get_nn(s,i)
        except ValueError:
            print('Check this structure...')
            print(s.species)
            print('Check this site')
            print(s[i].species)
            print('Ignore and dump this structure')
            dump_error_structure(s)
            self.if_error=True
        else:
            self.elements.append(s[i].species.elements[0].symbol)
            self.elements_coords.append(s[i].coords)
            self.neighbors_lst.append([site.species.elements[0].symbol for site in nn_list_per_atom])
            self.neighbors_coords.append([site.coords for site in nn_list_per_atom])
    
    def get_all_neighbors(self,s):
        '''
        S: a pymatgen structure
        '''
        if self.site_index is None:
            #No marked site
            natoms=len(s)
            if self.specify_element is None:
                #No specify element
                for i in range(natoms):
                    self.get_site_neighbors(s,i)
            else:
                for i in range(natoms):
                    #only find neighbors of speicify element
                    if s[i].species.elements[0].symbol==self.specify_element:
                        self.get_site_neighbors(s,i)
                    else:
                        pass
        else:
            natoms=len(s)
            if self.specify_element is None:
                #No Specify element
                for i in self.site_index:
                    if i>=natoms:
                        print('Out of range, ignore this!')
                        continue
                    else:
                        self.get_site_neighbors(s,i)
            else:
                #only find specify element
                for i in self.site_index:
                    if i>=natoms:
                        print('Out of range, ignore this!')
                        continue
                    else:
                        if s[i].species.elements[0].symbol==self.specify_element:
                            self.get_site_neighbors(s,i)
                        else:
                            continue
    
    def __call__(self,s):
        self.get_all_neighbors(s)

class nn_batch:
    def __init__(self,p_key=None,condition=None,element=None,neighbors_method='Voronoi',cutoff=None):
        self.cutoff=cutoff
        self.neighbors_method=neighbors_method
        self.specify_element=element
        self.p_key=p_key
        self.condition=condition
        self.structnn_lst=[]
    
    def site_property(self,s):
        '''
        s: pymatgen structure
        p_key: site_properties is a dictionary
        '''
        indexs=site_property(s,self.p_key,self.condition)
        return indexs

    def find_nn(self,s,site_index_list=None):
        nn=StructureNeighbors(site_index_lst=site_index_list,element=self.specify_element)
        if self.neighbors_method=='Voronoi':
            nn.get_all_neighbors(s)
        elif self.neighbors_method=='cutoff':
            nn.get_all_neighbors_by_cutoff(s,self.cutoff)
        return nn

    def find_nn_in_dataframe(self,df,n_jobs,local_lst=None,global_lst=None): 
        try:
            structures=df['structures']
        except KeyError:
            print('The DataFrame must have a columns with structure dictionary')
        else:
            s_lst=[]
            for s in structures:
                S=Structure.from_dict(literal_eval(s))
                if self.p_key is None:
                    s_lst.append([S])
                else:
                    if self.condition is None:
                        print('Find no condition,ignore the key!')
                        s_lst.append([S])
                    else:
                        indexs=self.site_property(S)
                        s_lst.append([S,indexs])
            nn_lst=parallel_Featurize(self.find_nn,s_lst,n_jobs)
            self.structnn_lst=[[s,local_lst,global_lst] for s in nn_lst]

#base Featurize
class basefeaturizer:
    def __init__(self,local_key='avg',global_key='avg',structnn_lst=None,p_key=None,condition=None,element=None,neighbors_methods='Voronoi',cutoff=None):
        self.cutoff=cutoff
        self.neighbors_method=neighbors_methods
        self.features_type=['_'] #rewrite this attribute
        self.feature_type='_' #rewrite this attribute
        self.p_key=p_key
        self.condition=condition
        self.specify_element=element
        self.oliynyk_csv_df=pd.read_csv(oliynyk_dir)
        self.local_keyName=local_key
        self.global_keyName=global_key
        self.structnn_lst=structnn_lst
    
    def featurize_local(self,nn):
        '''
        rewrite this function
        '''
        local_feature_lst=[]
        return local_feature_lst
    
    def featurize_global(self,nn):
        local_feature_lst=self.featurize_local(nn)
        if local_feature_lst is None:
            return np.nan
        else:
            local_feature_lst=np.array(local_feature_lst)
            if self.global_keyName=='avg':
                return np.mean(local_feature_lst)
            elif self.global_keyName=='std':
                return np.std(local_feature_lst)
            elif self.global_keyName=='ra':
                return max(local_feature_lst)-min(local_feature_lst)
            elif self.global_keyName=='sk':
                return st.skew(local_feature_lst)
            elif self.global_keyName=='ku':
                return st.kurtosis(local_feature_lst)
            elif self.global_keyName=='sum':
                return np.sum(local_feature_lst)
            else:
                print('error key!!')
                return np.nan
    
    def featurize(self,s):
        '''
        S: a pymatgen Structure object
        '''
        if self.p_key is None:
            site_index_lst=None
        else:
            if self.condition is None:
                site_index_lst=None
            else:
                site_index_lst=site_property(s,self.p_key,self.condition)
        nn=StructureNeighbors(site_index_lst,self.specify_element)
        if self.neighbors_method=='Voronoi':
            nn.get_all_neighbors(s)
        else:
            nn.get_all_neighbors_by_cutoff(s,self.cutoff)
        if nn.is_empty()==False:
            return np.nan
            
        feature=self.featurize_global(nn)
        return feature
    
    def featurize_dataframe(self,df,n_jobs):
        '''
        df: a pandas DataFrame containing structure information
        its column name should be 'structures'
        n_jobs: Finding Neighbors needs long time, using multi cpu cores to acclerate
        '''

        S_lst=[[Structure.from_dict(literal_eval(s))] for s in df['structures']]
        print(S_lst[0])
        result_lst=parallel_Featurize(self.featurize,S_lst,n_jobs)
        cols_name="_".join([self.feature_type,self.local_keyName,self.global_keyName])
        df[cols_name]=result_lst
        return df

    def featurize_structnn(self,nn,local_lst=None,global_lst=None):
        # key list
        feature_lst = self.features_type
        if local_lst is None:
            local_lst = ['avg','std', 'ra','sk', 'ku', 'sum'] #统一顺序
        else:
            pass
        if global_lst is None:
            global_lst = ['avg', 'std', 'ra','sk', 'ku', 'sum']
        else:
            pass
        output_dict={}

        #check if empty
        if nn.is_empty()==False:
            for f in feature_lst:
                for l in local_lst:
                    for g in global_lst:
                        output_dict['_'.join([f,l,g])]=np.nan
        else:
            #Featurize
            for i in range(len(feature_lst)):
                for j in range(len(global_lst)):
                    for k in range(len(local_lst)):
                        self.feature_type=feature_lst[i]
                        self.global_keyName=global_lst[j]
                        self.local_keyName=local_lst[k]
                        output_dict['_'.join([feature_lst[i],local_lst[k],global_lst[j]])] = self.featurize_global(nn)
        return output_dict
    
    def featurize_all(self,s,local_lst=None,global_lst=None):
        '''
        S: a pymatgen structure object
        This function will return a dictionary containing all kinds of electronegative
        '''
        #get nn
        if self.p_key is None:
            site_index_lst=None
        else:
            if self.condition is None:
                site_index_lst=None
            else:
                site_index_lst=site_property(s,self.p_key,self.condition)
        nn=StructureNeighbors(site_index_lst,self.specify_element)
        if self.neighbors_method=='Voronoi':
            nn.get_all_neighbors(s)
        else:
            nn.get_all_neighbors_by_cutoff(s,self.cutoff)
        D=self.featurize_structnn(nn,local_lst,global_lst)
        return D
    
    def featurize_all_dataframe(self,n_jobs,df,local_lst=None,global_lst=None):
        '''
        df: a pandas dataframe containing structures,the name of the column containing structures should
        be 'structures'
        '''
        s_list=[[Structure.from_dict(literal_eval(s)),local_lst,global_lst] for s in df['structures']]
        D_list=parallel_Featurize(self.featurize_all,s_list,n_jobs)
        total_dict={}
        for item in D_list:
            for key,value in item.items():
                total_dict.setdefault(key,[]).append(value)
        df1=pd.DataFrame(total_dict)
        return pd.concat([df,df1],axis=1)

    def featurize_neighbor_lst(self):
        '''
        Because finding the neighbors is time wasting, When featurize more than one features in this file, use this function
        '''
        #check nnlist
        if self.structnn_lst is None:
            print('finding the neighbors first')
            exit(0)
        else:
            dict_lst=[self.featurize_structnn(s[0],s[1],s[2]) for s in tqdm(self.structnn_lst)]
            total_dict={}
            for d in dict_lst:
                for key,value in d.items():
                    total_dict.setdefault(key,[]).append(value)
            return pd.DataFrame(total_dict)

#Featurize Electronegative

class ElectronegativeFeatures(basefeaturizer):
    '''
    en has choices below:
    'AR_EN': Allred-Rockow Electronegative, 
    'MB_EN': Martynov-Batsanov Electronegative, 
    'Go_EN': Gordy Electronegative, 
    'Mu_EN': Mulliken Electronegative, 
    'Pa_EN': Pauling Electronegative
    'TO_EN': Tantardini and Oganov Electronegative
    '''
    def __init__(self,en='Pa_EN',local_key='avg',global_key='avg',structnn_lst=None,p_key=None,condition=None,element=None,neighbors_methods='Voronoi',cutoff=None):
        basefeaturizer.__init__(self,local_key,global_key,structnn_lst,p_key,condition,element,neighbors_methods,cutoff)
        self.feature_type=en
        self.features_type=['AR_EN','MB_EN','Go_EN','Mu_EN','Pa_EN','TO_EN']
        self.property_dict=None
    
    def load_property(self):
        #EN_dict
        if self.feature_type=='AR_EN':
            self.property_dict=loadDictionary(self.oliynyk_csv_df,'Allred-Rockow_electronegativity')
        elif self.feature_type=='MB_EN':
            self.property_dict=loadDictionary(self.oliynyk_csv_df,'MB_electonegativity')
        elif self.feature_type=='Pa_EN':
            self.property_dict=loadDictionary(self.oliynyk_csv_df,'Pauling_Electronegativity')
        elif self.feature_type=='Go_EN':
            self.property_dict=loadDictionary(self.oliynyk_csv_df,'Gordy_electonegativity')
        elif self.feature_type=='Mu_EN':
            self.property_dict=loadDictionary(self.oliynyk_csv_df,'Mulliken_EN')
        elif self.feature_type=='TO_EN':
            self.property_dict=loadDictionary(self.oliynyk_csv_df,'TO_Electronegativity')
        else:
            print('{} is invalid key'.format(self.feature_type))
            exit(0)

    def gen_site_feature(self,nn):
        #elements and its neighbors
        elements=nn.elements
        nn_lst=nn.neighbors_lst

        l=len(elements)
        self.load_property()
        nbrs_property=[]
        for i in range(l):
            try:
                x_self_EN=self.property_dict[elements[i]]
            except KeyError:
                print('This Element {} has no specific properties!!!'.format(elements[i]))   
                return None
            else:
                if x_self_EN != x_self_EN:
                    print('This element {} has no EN data,returns NaN'.format(elements[i]))
                    return None
                nn_EN=[]
                for e in nn_lst[i]:
                    try:
                        x_nn_EN=self.property_dict[e]
                    except KeyError:
                        print('This Element {} has no specific properties!!!'.format(e))
                        return None
                    else:
                        if x_nn_EN != x_nn_EN:
                            print('This neighbor {} has no EN data,returns NaN'.format(e))
                            return None
                        nn_EN.append(x_nn_EN)
                nbrs_property.append(np.abs(np.array(nn_EN)-x_self_EN))
        return nbrs_property
    
    def featurize_local(self,nn):
        #gen feature for each site
        nbrs_property=self.gen_site_feature(nn)
        if nbrs_property is None:
            return None
        #aggregate
        local_feature_lst=[]
        for delta_EN in nbrs_property:
            #delta_EN=np.abs(nn_property-element_property[i])
            if self.local_keyName=='avg':
                avg=np.mean(delta_EN)
                local_feature_lst.append(avg)
            elif self.local_keyName=='std':
                std=np.std(delta_EN)
                local_feature_lst.append(std)
            elif self.local_keyName=='sk':
                sk=st.skew(delta_EN)
                local_feature_lst.append(sk)
            elif self.local_keyName=='ku':
                ku=st.kurtosis(delta_EN)
                local_feature_lst.append(ku)
            elif self.local_keyName=='ra':
                ra=max(delta_EN)-min(delta_EN)
                local_feature_lst.append(ra)
            elif self.local_keyName=='sum':
                SUM=np.sum(delta_EN)
                local_feature_lst.append(SUM)
            else:
                print('Error local key!')
                return None
        return local_feature_lst
  
    def featurize_labels(self):
        '''
        return available features
        '''
        all_features = ['Allred-Rockow', 'Martynov-Batsanov', 'Gordy', 'Mulliken', 'Pauling']  # enforce order
        return [x for x in all_features]


#Chemical Bonds 
class BondSphericalCoordinatesFeatures(basefeaturizer):
    '''
    Bond features describe the length and the angle of a bond
    Bond_features has choices below:
    Radius: length of bond
    theta: angle between bond and xoy plane, return the value of cosine
    Phi: angle between bond and yoz plane,return the value of cosine
    '''
    def __init__(self,Bond_feature='Radius',global_keyName='avg',local_keyName='avg',structnn_lst=None,p_key=None,condition=None,element=None,neighbors_methods='Voronoi',cutoff=None):
        basefeaturizer.__init__(self,global_keyName,local_keyName,structnn_lst,p_key,condition,element,neighbors_methods,cutoff)
        self.features_type=['R','T','P']
        self.feature_type=Bond_feature
    
    def transform_sphere(self,vec):
        r=np.linalg.norm(vec)
        theta=abs(vec[2])/r
        phi=abs(vec[0])/r
        return np.array([r,theta,phi])
    
    def Range_vector(self,vector):
        '''
        vector is a 2D array like data
        '''
        V=np.array(vector)
        if len(V.shape)==1:
            return np.max(V)-np.min(V)
        else:
            L=V.shape[1]
            return np.array([max(V[:,i])-min(V[:,i]) for i in range(L)])
    
    def gen_site_feature(self,nn,all=False):
        '''
        element_coords: attribution of class Structure_NN
        nn_coords: attribution of class Structure_NN
        '''
        element_coords=nn.elements_coords
        nn_coords=nn.neighbors_coords
        #Featurize
        nAtoms=len(element_coords)
        nbr_property=[]
        for i in range(nAtoms):
            spheres_per_atom=[]
            if len(nn_coords[i])==0:
                print("This site has no neighbors")
                nbr_property.append(np.array([]))
            else:
                for site_coord in nn_coords[i]:
                    deltaR=site_coord-element_coords[i]
                    if deltaR is None:
                        print('Error,dump this structure for debug')
                        return None
                    else:
                        sphere=self.transform_sphere(deltaR)
                        spheres_per_atom.append(sphere)
                spheres_per_atom=np.array(spheres_per_atom)
                if all==True:
                    nbr_property.append(spheres_per_atom)
                else:
                    if self.feature_type=='Radius':
                        nbr_property.append(spheres_per_atom[:,0])
                    elif self.feature_type=='Theta':
                        nbr_property.append(spheres_per_atom[:,1])
                    elif self.feature_type=='Phi':
                        nbr_property.append(spheres_per_atom[:,2])
        return nbr_property
    
    def featurize_local(self,nn,all=False):
        nbr_property=self.gen_site_feature(nn,all)
        if nbr_property is None:
            return None
        local_feature_lst=[]
        for spheres_per_atom in nbr_property:
            if self.local_keyName=='avg':
                local_feature_lst.append(np.mean(spheres_per_atom,axis=0))
            elif self.local_keyName=='std':
                local_feature_lst.append(np.std(spheres_per_atom,axis=0))
            elif self.local_keyName=='sk':
                local_feature_lst.append(st.skew(spheres_per_atom,axis=0))
            elif self.local_keyName=='ku':
                local_feature_lst.append(st.kurtosis(spheres_per_atom,axis=0))
            elif self.local_keyName=='sum':
                local_feature_lst.append(np.sum(spheres_per_atom,axis=0))  
            elif self.local_keyName=='ra':
                local_feature_lst.append(self.Range_vector(spheres_per_atom))
            else:
                print('Error Key! ')
                return None
        local_feature_lst=np.array(local_feature_lst)
        return local_feature_lst
              

    def featurize_structnn(self,nn,local_lst=None,global_lst=None):
        #all the keyname
        if local_lst is None:
            local_lst = ['avg','std', 'ra','sk', 'ku', 'sum'] #统一顺序
        else:
            pass
        if global_lst is None:
            global_lst = ['avg', 'std', 'ra','sk', 'ku', 'sum']
        else:
            pass
        rtp=self.features_type

        D={}
        #check error
        if nn.is_empty() == False:
            for l in local_lst:
                for g in global_lst:
                    for r in rtp:
                        D['_'.join([r,l,g])]=np.nan
            return D
        
        #Featurize
        for l in local_lst:
            for g in global_lst:
                self.local_keyName=l
                local_feature_lst=self.featurize_local(nn,all=True)
                if local_feature_lst is None:
                    return np.nan
                else:
                    local_feature_lst=np.array(local_feature_lst)
                    if g=='avg':
                        Bond_vec=np.mean(local_feature_lst,axis=0)
                        for i in range(3):
                            D['_'.join([rtp[i],l,g])]=Bond_vec[i]
                    elif g=='std':
                        Bond_vec=np.std(local_feature_lst,axis=0)
                        for i in range(3):
                            D['_'.join([rtp[i],l,g])]=Bond_vec[i]
                    elif g=='sk':
                        Bond_vec=st.skew(local_feature_lst,axis=0)
                        for i in range(3):
                            D['_'.join([rtp[i],l,g])]=Bond_vec[i]
                    elif g=='ku':
                        Bond_vec=st.kurtosis(local_feature_lst,axis=0)
                        for i in range(3):
                            D['_'.join([rtp[i],l,g])]=Bond_vec[i]
                    elif g=='ra':
                        Bond_vec=self.Range_vector(local_feature_lst)
                        for i in range(3):
                            D['_'.join([rtp[i],l,g])]=Bond_vec[i]
                    elif g=='sum':
                        Bond_vec=np.sum(local_feature_lst,axis=0)
                        for i in range(3):
                            D['_'.join([rtp[i],l,g])]=Bond_vec[i]
        return D


#Coordination Number  
class CoordinationNumeber(basefeaturizer):
    '''
    This feature describe the number neighbors of each atom in a structure
    '''
    def __init__(self,global_keyName='avg',structnn_lst=None,p_key=None,condition=None,element=None,neighbors_methods='Voronoi',cutoff=None):
        basefeaturizer.__init__(self,local_key='avg',global_key=global_keyName,structnn_lst=structnn_lst,p_key=p_key,condition=condition,element=element,neighbors_methods=neighbors_methods,cutoff=cutoff)
        self.features_type=['CN']
        self.feature_type='CN'
    
    def featurize_local(self,nn):
        '''
        return a list
        '''
        elements=nn.elements
        nn_lst=nn.neighbors_lst
        nAtoms=len(elements)
        local_feature_list=[]
        for i in range(nAtoms):
            local_feature_list.append(len(nn_lst[i]))
        return local_feature_list

    def featurize_structnn(self,nn,local_lst=None,global_lst=None):
        #all the keyname
        if global_lst is None:
            global_lst = ['avg', 'std', 'ra','sk', 'ku', 'sum']
        else:
            pass

        D={}
        #check if empty
        if nn.is_empty()==False:
            for g in global_lst:
                 D['_'.join(['CN',g])]=np.nan
        else:
            for g in global_lst:
                self.global_keyName=g
                D['_'.join(['CN',g])]=self.featurize_global(nn)
        return D




if __name__=='__main__':
    from datetime import datetime
    import sys
    testset=pd.read_csv('testdata\\HSE_semi.csv')
    #testset=testset.head(5000)
    N=nn_batch(neighbors_method='cutoff',cutoff=8)
    N.find_nn_in_dataframe(testset,n_jobs=8,local_lst=['avg'])
    method=ElectronegativeFeatures(structnn_lst=N.structnn_lst)
    start_time=datetime.now()
    D=method.featurize_neighbor_lst()
    end_time = datetime.now()
    print(f'TOTAL TIME TO Featurize = {(end_time - start_time).seconds} seconds')
    print(D)
    # S=Structure.from_file('testdata\\KRb2RhF6.cif')
    # E=ElectronegativeFeatures()
    # Nbr=StructureNeighbors()
    # Nbr.get_all_neighbors_by_cutoff(S,cutoff=8)
    # nbr_property=E.gen_site_feature(Nbr)
    # nbr_fea_EN = []
    # for nbr in nbr_property:
    #     if len(nbr)<12:
    #         nbr_fea_EN.append(nbr.tolist()+[0]*(12-len(nbr)))
    #     else:
    #         nbr_fea_EN.append(nbr[:12])
    # nbr_fea_EN=np.array(nbr_fea_EN)
    # #nbr_fea_EN=nbr_fea_EN[...,np.newaxis]
    # print(nbr_fea_EN)
    # all_nbrs=S.get_all_neighbors(r=8,include_index=True)
    # all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    # nbr_fea_idx, nbr_fea = [], []
    # for nbr in all_nbrs:
    #     if len(nbr)<12:
    #         nbr_fea_idx.append(list(map(lambda x:x[2],nbr))+[0]*(12-len(nbr)))
    #         nbr_fea.append(list(map(lambda x:x[1],nbr))+[8+1]*(12-len(nbr)))
    #     else:
    #         nbr_fea_idx.append(list(map(lambda x: x[2],
    #                                         nbr[:12])))
    #         nbr_fea.append(list(map(lambda x: x[1],
    #                                     nbr[:12])))
    # nbr_fea_idx,nbr_fea=np.array(nbr_fea_idx), np.array(nbr_fea)
    # f=np.arange(0,8.2,0.2)
    # nbr_fea=np.exp(-(nbr_fea[...,np.newaxis]-f)**2/0.2**2)
    # print(nbr_fea*nbr_fea_EN)