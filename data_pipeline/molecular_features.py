# generate molecular features

from typing import Callable, List

import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


MoleculeFeaturesGenerator = Callable[[str], np.ndarray]

MOLECULE_FEATURES_GENERATOR_REGISTRY = {}


def register_molecule_features_generator(features_generator_name: str) \
        -> Callable[[MoleculeFeaturesGenerator], MoleculeFeaturesGenerator]:
    """
    Creates a decorator which registers a molecule feature generator in global dictionaries to enable access by nome.

    :param features_generator_name: The name to use to access the features generator
    :return: A decorator which will add a molecule features generator to the registry using the specified name
    """

    def decorator(features_generator: MoleculeFeaturesGenerator) -> MoleculeFeaturesGenerator:
        MOLECULE_FEATURES_GENERATOR_REGISTRY[features_generator_name] = features_generator
        return features_generator

    return decorator


def get_features_generator(features_generator_name: str) -> MoleculeFeaturesGenerator:
    """
    Gets a registered features generator by name.

    :param features_generator_name: The name of the features generator.
    :return: The desired features generator.
    """
    if features_generator_name not in MOLECULE_FEATURES_GENERATOR_REGISTRY:
        raise ValueError(f'Features generator "{features_generator_name}" could not be found. ')

    return MOLECULE_FEATURES_GENERATOR_REGISTRY[features_generator_name]


def get_available_features_generators() -> List[str]:
    """Returns a list of names of available features generators."""
    return list(MOLECULE_FEATURES_GENERATOR_REGISTRY.keys())


@register_molecule_features_generator('morgan_fingerprint')
def morgan_fingerprint_generator(mol: Chem.Mol,
                                 radius: int = 2,
                                 num_bits: int = 1024) -> np.ndarray:
    """
    Generate morgen fingerprint with RDKit
    :param mol: rdkit molecule
    :param radius: Morgen fingerprint radius
    :param num_bits: Number of bits in Morgan fingerprint
    :return: A 1D numpy array containing the binary Morgan fingerprint
    """
    morgan_fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, num_bits, useChirality=True)
    morgan_fingerprint_array = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(morgan_fingerprint, morgan_fingerprint_array)
    return morgan_fingerprint_array


@register_molecule_features_generator('RDKit_2d_descriptors')
def rdkit_2d_descriptors_generator(mol: Chem.Mol) -> np.ndarray:
    """
    Generate RDKit 2D descriptors
    :param mol: rdkit molecule
    :return: A 1D numpy array containing the RDKit 2D descriptors
    """
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    descriptors = calc.CalcDescriptors(mol)
    return np.array(descriptors)


def get_rdkit_2d_descriptors_name() -> List:
    # return the name of RDKit 2D descriptors by order
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    return calc.GetDescriptorNames()