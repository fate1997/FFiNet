import os
from tqdm import tqdm

from rdkit import Chem
from rdkit import Chem

import torch
from torch_geometric.data import Data
import networkx as nx

from scipy.spatial import distance
from scipy.sparse import coo_matrix
import numpy as np

from .atom_features import get_available_features_generators, atom_features_union
from .data_generating import subgraph_index

from openbabel import pybel


def combine_ligand_pocket(ligand, pocket, y, cutoff):   
        """
        Combine the ligand graph and pocket graphs to one graph
        :ligand: An RDKit Mol
        :pocket: An RDKit Mol
        :cutoff: Minimum Distance for Connection
        return: An PyG Data Object
        """

        # atom features of ligand and pocket
        x_ligand = []
        for atom in ligand.GetAtoms():
            atom_generator_name_list = get_available_features_generators()
            atom_generator_name_list.pop(-1) # since the hydrogen bond generation is very time-consuming
            features = atom_features_union(ligand, atom, atom_generator_name_list)
            features = np.insert(features, -1, 0)
            x_ligand.append(features)
        x_ligand = torch.tensor(x_ligand, dtype=torch.float32)

        x_pocket = []
        for atom in pocket.GetAtoms():
            atom_generator_name_list = get_available_features_generators()
            atom_generator_name_list.pop(-1)
            features = atom_features_union(pocket, atom, atom_generator_name_list)
            features = np.insert(features, -1, 1)
            x_pocket.append(features)
        x_pocket = torch.tensor(x_pocket, dtype=torch.float32)

        x = torch.cat([x_pocket, x_ligand], dim=0)


        # position
        conformer_pocket = pocket.GetConformer()
        pos_pocket = conformer_pocket.GetPositions()
        pos_pocket = torch.tensor(pos_pocket, dtype=torch.float32)

        conformer_ligand = ligand.GetConformer()
        pos_ligand = conformer_ligand.GetPositions()
        pos_ligand = torch.tensor(pos_ligand, dtype=torch.float32)

        pos = torch.cat([pos_pocket, pos_ligand], dim=0)

        
        # bonded edges
        pocket_adj = Chem.GetAdjacencyMatrix(pocket)
        pocket_G = nx.from_numpy_matrix(pocket_adj)

        ligand_adj = Chem.GetAdjacencyMatrix(ligand)
        ligand_G = nx.from_numpy_matrix(ligand_adj)
        ligand_G = nx.convert_node_labels_to_integers(ligand_G, first_label=sorted(pocket_G)[-1] + 1)

        edges_bonded = list(pocket_G.edges) + list(ligand_G.edges)

        # unbonded edges
        pocket_SGs = [pocket_G.subgraph(c) for c in nx.connected_components(pocket_G)]
        distance_graph = distance.cdist(pos, pos, 'euclidean')
        distance_graph[distance_graph >= cutoff] = 0
        SGs = pocket_SGs + [ligand_G]
        for SG in SGs:
            low_bond = sorted(SG)[0]
            high_bond = sorted(SG)[-1] + 1
            distance_graph[low_bond: high_bond, low_bond: high_bond] = 0
        coo = coo_matrix(distance_graph)
        edges_unbonded = [(i, j) for i, j in zip(coo.row, coo.col)]

        # edge_index and edge attributes
        edges = edges_bonded + edges_unbonded

        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_attr = [0 for _ in range(len(edges_bonded))] + [1 for _ in range(len(edges_unbonded))]
        edge_attr = torch.tensor(edge_attr, dtype=torch.bool)

        # data, triple_index and quadra_index 
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y)
        G = nx.from_edgelist(edges_bonded)
        data.triple_index = subgraph_index(G, 2)
        data.quadra_index = subgraph_index(G, 3)

        return data


class PDBBindPreprocess:
    def __init__(self, core_path, refined_path):
        self.core_path = core_path
        self.refined_path = refined_path

    # Transform the ligand file to the file that RDKit can read
    def mol2_to_pdb(self):
        # core set
        for name in tqdm(os.listdir(self.core_path), desc='ligand in core set'):
            if len(name) != 4:
                continue
            
            pybel_mol = next(pybel.readfile('mol2', '%s/%s/%s_ligand.mol2' % (self.core_path, name, name)))
            pybel_mol.write("pdb", '%s/%s/%s_ligand_pybel.pdb' % (self.core_path, name, name), overwrite=True)

        # refine set
        for name in tqdm(os.listdir(self.refined_path), desc='ligand in refined set'):
            if len(name) != 4 or (name in os.listdir(self.core_path)):
                continue
            pybel_mol = next(pybel.readfile('mol2', '%s/%s/%s_ligand.mol2' % (self.refined_path, name, name)))
            pybel_mol.write("pdb", '%s/%s/%s_ligand_pybel.pdb' % (self.refined_path, name, name), overwrite=True)

    # Removes all HETATM records in the protein file by pdb-tools
    def remove_hetatm(self):
        # core set 
        for name in tqdm(os.listdir(self.core_path), desc='protein in core set'):
            if len(name) != 4:
                continue
            os.system('pdb_delhetatm %s/%s/%s_pocket.pdb > %s/%s/%s_pocket_clean.pdb' % (self.core_path, name, name, self.core_path, name, name))

        # refine set
        for name in tqdm(os.listdir(self.refined_path), desc='protein in refined set'):
            if len(name) != 4 or (name in os.listdir(self.core_path)):
                continue
            os.system('pdb_delhetatm %s/%s/%s_pocket.pdb > %s/%s/%s_pocket_clean.pdb' % (self.refined_path, name, name, self.refined_path, name, name))
    

def pdbbind_dataset_generate(
            core_path='..\data_pipeline\data_files\PDBbind_dataset\core-set\\', 
            refined_path='..\data_pipeline\data_files\PDBbind_dataset\\refined-set\\', 
            pk_path='..\data_pipeline\data_files\PDBbind_dataset\\refined-set\index\INDEX_general_PL_data.2016', 
            save_path='..\data_pipeline\data_files\PDBbind_dataset\\',
            cutoff=5):

    # generate dict {code:pka}
    res = {}
    with open(pk_path) as f:
        for line in f:
            if '#' in line:
                continue
            cont = line.strip().split()
            if len(cont) < 5:
                continue
            code, pk = cont[0], cont[3]
            res[code] = [float(pk)]
    
    # core set
    data_list = []
    for name in tqdm(os.listdir(core_path)):
        if len(name) != 4:
            print(name)
            continue
        
        pocket = Chem.MolFromPDBFile('%s/%s/%s_pocket_clean.pdb' % (core_path, name, name))
        ligand = Chem.MolFromPDBFile('%s/%s/%s_ligand_pybel.pdb' % (core_path, name, name))

        if (not pocket) or (not ligand):
            continue

        data = combine_ligand_pocket(ligand, pocket, torch.tensor(res[name], dtype=torch.float32), cutoff)
        data.smiles = name
        data_list.append(data)
    torch.save(data_list, save_path + 'pdbbind_core_' + str(cutoff) + 'A.pt') 
    
    # refined set
    refined_list = []
    for name in tqdm(os.listdir(refined_path)):
        if len(name) != 4 or (name in os.listdir(core_path)):
            continue
        
        try:
            pocket = Chem.MolFromPDBFile('%s/%s/%s_pocket_clean.pdb' % (refined_path, name, name))
            ligand = Chem.MolFromPDBFile('%s/%s/%s_ligand_pybel.pdb' % (refined_path, name, name))
        except OSError:
            continue

        if (not pocket) or (not ligand):
            continue

        data = combine_ligand_pocket(ligand, pocket, torch.tensor(res[name], dtype=torch.float32), cutoff)
        data.smiles = name
        refined_list.append(data)

    torch.save(refined_list, save_path + 'pdbbind_refined_' + str(cutoff) + 'A.pt')
