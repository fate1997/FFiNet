import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.sparse import coo_matrix
import os

from torch_geometric.data import Data
import torch

from rdkit import Chem
from rdkit.Chem import AllChem

from .atom_features import get_available_features_generators, atom_features_union

import networkx as nx


class DataGenerating:
    def __init__(self,
                 folder: str = './data/',
                 raw_dataset_name: str = None,
                 feature_dict_name: str = None,
                 dataset_name: str = None):
        self.folder = folder
        self.raw_dataset_name = raw_dataset_name
        self.feature_dict_name = feature_dict_name
        self.dataset_name = dataset_name
        self.features_dict_exist = False

        # load data file
        data_path = os.path.join(self.folder, 'raw_files', self.raw_dataset_name)

        self.csv = self.raw_dataset_name.split('.')[-1].lower() == 'csv'
        if self.csv:
            df = pd.read_csv(data_path)
            # find smiles column and collect smiles
            columns = df.columns.to_list()
            columns_lower = [column.lower() for column in columns]
            assert 'smiles' in columns_lower
            df.columns = columns_lower
            self.smiles_list = []
            for smiles in df['smiles']:
                self.smiles_list.append(smiles)
            self.mols = len(self.smiles_list) * [None]
            for i, smiles in enumerate(self.smiles_list):
                self.mols[i] = Chem.MolFromSmiles(self.smiles_list[i])
        else: # for SDF file
            self.mols = Chem.SDMolSupplier(data_path)

    def features_generating(self):
        
        features_dict = {}
        for i, mol in enumerate(tqdm(self.mols, desc='Data Processing')):
            
            features_dict[i] = {}
            # atom position
            if self.csv:
                
                num_atoms = mol.GetNumAtoms()
                mol_add_hs = Chem.AddHs(mol)
                
                # generate conformer by EDKDG method
                AllChem.EmbedMolecule(mol_add_hs, randomSeed=0xf00d)
                try:
                    conf = mol_add_hs.GetConformers()[0]
                except IndexError:
                    AllChem.EmbedMultipleConfs(mol_add_hs, 50, pruneRmsThresh=0.5)
                    try:
                        conf = mol_add_hs.GetConformers()[0]
                    except IndexError:
                        print(f'{Chem.MolToSmiles(mol)}\'s conformer can not be generated')
                        conf = None
                
                # Throw the molecules with no conformer generated
                if conf != None:
                    features_dict[i]['pos'] = conf.GetPositions()[:num_atoms, :]
                else:
                    features_dict.pop(i)
                    continue
            # Throw the invalid molecules in SDF file
            else:
                if mol == None:
                    print(f'Number of {i} can not generate mol object.')
                    features_dict.pop(i)
                    continue
            
                pos = mol.GetConformer().GetPositions()
                features_dict[i]['pos'] = pos

            # edge index (1-hop index)
            adj = Chem.GetAdjacencyMatrix(mol)
            coo_adj = coo_matrix(adj)
            features_dict[i]['edge_index'] = [coo_adj.row, coo_adj.col]

            # atom features (z for DimeNet)
            x = []
            z = []
            for atom in mol.GetAtoms():
                atom_generator_name_list = get_available_features_generators()
                x.append(atom_features_union(mol, atom, atom_generator_name_list))
                z.append(atom.GetAtomicNum())
            features_dict[i]['x'] = x
            features_dict[i]['z'] = z

        # save files as npy
        if not os.path.exists(os.path.join(self.folder, 'feature_files')):
            os.makedirs(os.path.join(self.folder, 'feature_files'))
        feature_save_path = os.path.join(self.folder, 'feature_files', self.feature_dict_name)
        np.save(feature_save_path, features_dict)

        self.features_dict_exist = True

    def dataset_creating(self, target_name, dtype=torch.float32):

        # the function feature_generating must be first applied to get the feature dict 
        if self.features_dict_exist:
            features_dict = np.load(os.path.join(self.folder, 'feature_files', self.feature_dict_name), allow_pickle=True).item()
            data_list = []
            for i, mol in enumerate(tqdm(self.mols, desc='Dataset creating')):
                if i not in features_dict.keys():
                     print(f'Number of {i} do not have features.')
                     continue
                features = features_dict[i]

                # load label of dataset 
                if type(target_name) is list: # for multi-label tasks like QM9
                    y = [float(mol.GetProp(t)) for t in target_name]
                elif type(target_name) is pd.core.frame.DataFrame:
                    y = target_name.iloc[i, :]
                elif type(target_name) is pd.core.series.Series:
                    y = target_name.iloc[i]
                elif type(target_name) is dict:
                    y = target_name[mol.GetProp('_Name')]
                
                # generate data object
                data = Data(
                    z = torch.tensor(features['z'], dtype=torch.long),
                    x=torch.tensor(features['x'], dtype=dtype),
                    edge_index=torch.tensor(features['edge_index'], dtype=torch.long),
                    pos=torch.tensor(features['pos'], dtype=dtype),
                    y=torch.tensor(y, dtype=dtype))
                
                adj = Chem.GetAdjacencyMatrix(mol)
                G = nx.from_numpy_matrix(adj)
                data.triple_index = subgraph_index(G, 2) # 2-hop index
                data.quadra_index = subgraph_index(G, 3) # 3-hop index

                if self.csv:
                    data.smiles = Chem.MolToSmiles(mol)
                else:
                    data.smiles = mol.GetProp('_Name')
                data_list.append(data)

            torch.save(data_list, self.folder + self.dataset_name)
        else:
            raise FileNotFoundError(f"There are no features dictionary in the {self.folder}")


# Finding all paths/walks of given length in a networkx graph
# code from https://www.py4u.net/discuss/162645
def subgraph_index(G, n):

    allpaths = []
    for node in G:
        paths = findPaths(G, node , n)
        allpaths.extend(paths)
    allpaths = torch.tensor(allpaths, dtype=torch.long).T
    return allpaths

def findPaths(G,u,n,excludeSet = None):
    if excludeSet == None:
        excludeSet = set([u])
    else:
        excludeSet.add(u)
    if n==0:
        return [[u]]
    paths = [[u]+path for neighbor in G.neighbors(u) if neighbor not in excludeSet for path in findPaths(G,neighbor,n-1,excludeSet)]
    excludeSet.remove(u)
    return paths