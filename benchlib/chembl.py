import os
import csv
import cPickle as pickle
from rdkit import Chem
import gzip
from itertools import ifilter

def load_chembl():
    smiles_lookup = dict()
    with gzip.open(os.path.join(here, 'chembl_20.sdf.gz')) as inf:
        suppl = Chem.ForwardSDMolSupplier(inf)
        smiles_lookup = dict((mol.GetProp('chembl_id')[6:], Chem.MolToSmiles(mol)) for mol in ifilter(None, suppl))
    return smiles_lookup


here = os.path.dirname(os.path.realpath(__file__))
if os.path.isfile(os.path.join(here, 'chembl_20.smi.pickle')):
    with open(os.path.join(here, 'chembl_20.smi.pickle'), 'rb') as handle:
        smiles_lookup = pickle.load(handle)
else:
    smiles_lookup = load_chembl()
    with open(os.path.join(here, 'chembl_20.smi.pickle'), 'wb') as handle:
        pickle.dump(smiles_lookup, handle, protocol=pickle.HIGHEST_PROTOCOL)
