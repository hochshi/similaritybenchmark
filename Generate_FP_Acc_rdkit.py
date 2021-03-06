import benchlib.chembl as chembl
import benchlib.fingerprint_lib as flib # From benchmarking platform
from rdkit import Chem
import numpy as np
import scipy.sparse as sparse
import os
from sys import argv
import progressbar

def save_sparse_matrix(filename, x):
    row = x.row
    col = x.col
    data = x.data
    shape = x.shape
    np.savez(filename, row=row, col=col, data=data, shape=shape)

def load_sparse_matrix(filename):
    y = np.load(filename)
    z = sparse.coo_matrix((y['data'], (y['row'], y['col'])), shape=y['shape'])
    return z

def resize_acc(acc, newSize):
    print "This shouldn't happen - resizing the acc matrix"
    mat = acc.tocoo()
    return sparse.coo_matrix((mat.data, (mat.row, mat.col)), shape=(1, newSize+1e9)).tocsr()

all_smiles = chembl.smiles_lookup.values()
init_smile = all_smiles.pop()

def accfp(fpName, fpCalculator):
    print "Doing fingerprint %s" % fpName
    if os.path.isfile('%s.npz' % fpName):
        return
    init_fp = fpCalculator(Chem.MolFromSmiles(init_smile))
    bar = progressbar.ProgressBar(widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Percentage(), progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ])
    for smile in bar(all_smiles):
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            fp = fpCalculator(mol)
            init_fp += fp
    if fpName in ["ap", "tt"] or fpName.startswith("ecfc") \
            or fpName.startswith("fcfc") or fpName.startswith("nc_"):
        col = init_fp.GetNonzeroElements().keys()
        data = init_fp.GetNonzeroElements().values()
        row = [0] * len(col)
        acc = sparse.coo_matrix((data, (row, col)), shape=(1, max(col)*2)).tocsr()
    else:
        # max_col = 1024
        # if fpName.startswith("lecfp") or fpName.startswith("lfcfp") or fpName == "laval":
        #     max_col = 16384
        max_col = init_fp.GetNumBits()
        acc = np.zeros(max_col)
        for i in init_fp.GetOnBits():
            acc[i] += 1
    save_sparse_matrix(fpName, sparse.coo_matrix(acc))

if __name__ == "__main__":
    fpName = argv[1]
    try:
        fpCalculator = flib.fpdict[fpName]
        accfp(fpName, fpCalculator)
    except KeyError:
        print "Key %s was not found" % fpName
        quit(1)