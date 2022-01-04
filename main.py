import LocalStructure as ls
import pandas as pd 
import argparse

from datetime import datetime
from tqdm import tqdm


def featurizeall(df,n_jobs,p_key=None,condition=None,element=None,neigbors_method='Voronoi',cutoff=None):
    N=ls.nn_batch(p_key,condition,element,neigbors_method,cutoff)
    start_time=datetime.now()
    N.find_nn_in_dataframe(df,n_jobs)
    end_time = datetime.now()
    print(f'TOTAL TIME TO Find nn = {(end_time - start_time).seconds} seconds')
    DE=ls.ElectronegativeFeatures(structnn_lst=N.structnn_lst)
    BSC=ls.BondSphericalCoordinatesFeatures(structnn_lst=N.structnn_lst)
    CN=ls.CoordinationNumeber(structnn_lst=N.structnn_lst)
    start_time=datetime.now()
    for D in tqdm([DE,BSC,CN]):
        d_feature=D.featurize_neighbor_lst()
        df=pd.concat([df,d_feature],axis=1)
    end_time = datetime.now()
    print(f'TOTAL TIME TO Featurize = {(end_time - start_time).seconds} seconds')
    return df

def main():
    parse=argparse.ArgumentParser()
    parse.add_argument('-e',help='Specify the element to find neighbors',type=str, default=None)
    parse.add_argument('-k',help='Specify the site properties key name marked in the pymatgen Structure object',type=str,default=None)
    parse.add_argument('-c',help='if Specify the -k, then you should specify a condition that sites in structure match',default=None)
    parse.add_argument('--type',help='Specify the type of condition, int, float, str or bool',type=str,choices=['int','float','str'],default='int')
    parse.add_argument('-t',help='Specify the cpu cores you used',type=int, default=1)
    parse.add_argument('-o',help='output filename',type=str,default='result.csv')
    parse.add_argument('-n',help='Specify the method to find neighbors, Voronoi or cutoff can be choice',type=str,choices=['Voronoi','cutoff'],default='Voronoi')
    parse.add_argument('--cutoff',help='give the cutoff radius to find neighbors,ignored when using Voronoi',type=float,default=None)
    parse.add_argument('descriptor',help="Specify the descriptor, available choice:Electronegative, Bond, CoordinationNumber",choices=['Electronegative', 'Bond', 'CoordinationNumber', 'All'])
    parse.add_argument('FilePath',help="Specify the csv file path")
    #load_data
    args=parse.parse_args()
    file_path=args.FilePath
    try:
        data=pd.read_csv(file_path)
    except FileNotFoundError:
        print('Can not find your file')
    else:
        try:
            s=data['structures']
        except KeyError:
            print('This file has no structures')
        else:
            #check element
            n_jobs=args.t
            element=args.e
            p_key=args.k
            condition=args.c 
            condition_type=args.type
            outputfilename=args.o
            neighbors_method=args.n 
            cutoff=args.cutoff
            if condition_type=='int':
                condition=int(condition)
            elif condition_type=='float':
                condition=float(condition)
            else:
                condition=str(condition)
            print("No Specify element, featurize all for you")
            D=args.descriptor
            
            if D=='Electronegative':
                method=ls.ElectronegativeFeatures(p_key=p_key,condition=condition,element=element,neighbors_methods=neighbors_method,cutoff=cutoff)
                df=method.featurize_all_dataframe(data,n_jobs)
            elif D=='Bond':
                method=ls.BondSphericalCoordinatesFeatures(p_key=p_key,condition=condition,element=element,neighbors_methods=neighbors_method,cutoff=cutoff)
                df=method.featurize_all_dataframe(data,n_jobs)
            elif D=='CoordinationNumber':
                method=ls.CoordinationNumeber(p_key=p_key,condition=condition,element=element,neighbors_methods=neighbors_method,cutoff=cutoff)
                df=method.featurize_all_dataframe(data,n_jobs)
            elif D=='All':
                df=featurizeall(data,n_jobs,p_key=p_key,condition=condition,element=element,neigbors_method=neighbors_method,cutoff=cutoff)
            df.to_csv(outputfilename)
            
if __name__=='__main__':
    main()