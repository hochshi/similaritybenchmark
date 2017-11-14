import os
#import sys
#import cPickle as pickle
#import itertools
#from collections import defaultdict, Counter
import benchlib.chembl as chembl
import benchlib.fingerprint_lib as flib # From benchmarking platform
from rdkit import Chem #, DataStructs
#from rdkit.Chem import Descriptors
#from rdkit.Chem.Fraggle import FraggleSim
import numpy as np
import scipy.sparse as sp
from sys import argv

# Added for parallelism
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

local_location = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

def check_sparse(name):
    if name in ["ap", "tt"] or name.startswith("ecfc") \
            or name.startswith("fcfc") or name.startswith("nc_"):
        return True
    return False

def load_sparse_matrix(fpName):
    y = np.load(os.path.join(local_location, "%s.npz" % fpName))
    z = sp.coo_matrix((y['data'], (y['row'], y['col'])), shape=y['shape'], dtype=np.float64).tocsr()
    if check_sparse(fpName):
        return z
    return z.todense()

def accToWeights(acc):
    return np.log1p(acc.sum()/acc)

def sparseAccToWeights(acc):
    return acc.power(-1).multiply(np.sum(acc.data)).log1p()

def loadWeights(fpName):
    if check_sparse(fpName):
        return sparseAccToWeights(load_sparse_matrix(fpName))
    return accToWeights(load_sparse_matrix(fpName))

def molToFp(mol, fpName, fpCalculator, sparseShape):
    fp = fpCalculator(mol)
    return fpToNumpyArray(fpName, fp, sparseShape)

def fpToNumpyArray(fpName, fp, sparseShape):
    if check_sparse(fpName):
        col = fp.GetNonzeroElements().keys()
        data = fp.GetNonzeroElements().values()
        row = [0] * len(col)
        acc = sp.coo_matrix((data, (row, col)), shape=sparseShape).tocsr()
    else:
        max_col = 1024
        if fpName.startswith("lecfp") or fpName.startswith("lfcfp") or fpName == "laval":
            max_col = 16384
        acc = np.zeros(max_col)
        for i in fp.GetOnBits():
            acc[i] += 1
    return acc

def weightedTanimotoSimilarity(fp1, fp2, wv):
    #andFp = fp1*fp2
    #andSum = np.nansum(np.multiply(andFp, wv))
    return np.nansum(np.multiply(fp1*fp2, wv))/np.nansum(np.multiply(np.maximum(fp1, fp2), wv))

# Make sure input to this function are all CSR.
def sparseWeightedTanimotoSimilarity(fp1, fp2, wv):
    #andFp = fp1.multiply(fp2)
    #andSum = np.sum(andFp.multiply(wv).data)
    return np.sum(fp1.multiply(fp2).multiply(wv).data)/(np.sum(fp1.maximum(fp2).multiply(wv).data))

# Make sure input to this function are all CSR.
def sparseWeightedDiceSimilarity(fp1, fp2, wv):
    return np.sum(fp1.minimum(fp2).multiply(wv).data)/np.sum((fp1+fp2).multiply(wv).data)

def readBenchmark(fname):
    for line in open(fname):
        yield line.rstrip().split()

def evaluate_similarity_method(dataset, resultsdir):
    size = 4
    ref_corr = range(size, 0, -1)
    ref_corr_b = range(0, size)

    # Setup results dir
    if not os.path.isdir(resultsdir):
        os.mkdir(resultsdir)

    writer = Writer(resultsdir)
    for i, d in enumerate(get_rdkitmols(dataset)):
        for fpName, fpCalculator in flib.fpdict.iteritems():
            #print "Doing fingerprint %s" % fpName
            wv = loadWeights(fpName)
            ref_mol = d[0]
            # ref_fp = fpCalculator(ref_mol)
            # ref_fp = fpToNumpyArray(fpName, ref_fp)
            ref_fp = molToFp(ref_mol, fpName, fpCalculator, wv.shape)
            #ref_nbonds = ref_mol.GetNumBonds()
            tanimotos = []
            #adjusted_tanimotos = []
            for smol in d[1:]:
                #sfp = fpCalculator(smol)
                #sfp = fpToNumpyArray(fpName, sfp)
                sfp = molToFp(smol, fpName, fpCalculator, wv.shape)
                if fpName in ["ap", "tt"] or fpName.startswith("ecfc") or fpName.startswith("fcfc"):
                    tanimoto = sparseWeightedDiceSimilarity(ref_fp, sfp, wv)
                    #tanimoto = DataStructs.DiceSimilarity(ref_fp, sfp)
                elif fpName.startswith("nc_"):
                    tanimoto = sparseWeightedTanimotoSimilarity(ref_fp, sfp, wv)
                else:
                    tanimoto = weightedTanimotoSimilarity(ref_fp, sfp, wv)
                    #tanimoto = DataStructs.cDataStructs.TanimotoSimilarity(ref_fp, sfp)
                tanimotos.append(tanimoto)

            label = fpName
            writer.write_result(label, tanimotos, i==0)

class Writer(object):
    def __init__(self, resultsdir):
        self.files = {}
        self.resultsdir = resultsdir
    def write_result(self, fpname, fpresult, deletefile):
        ## Write out the results
        fname = os.path.join(self.resultsdir, fpname + ".txt")
        if deletefile:
            if os.path.isfile(fname):
                os.remove(fname)
            self.files[fname] = open(fname, "w")
        f = self.files[fname]
        f.write(" ".join(map(str, fpresult)))
        f.write("\n")

def write_results(fpname, resultsdir, fpresults):
    ## Write out the results
    fname = open(os.path.join(resultsdir, fpname + ".txt"), "w")
    for fpresult in fpresults:
        fname.write(" ".join(map(str, fpresult)))
        fname.write("\n")
    fname.close()

def get_rdkitmols(dataset):
    for d in dataset:
        tmp = []
        for smi in d:
            mol = Chem.MolFromSmiles(smi)
            if "." in smi:
                frags = list(Chem.GetMolFrags(mol, asMols=True))
                frags.sort(key=lambda x:x.GetNumHeavyAtoms(), reverse=True)
                mol = frags[0]
            tmp.append(mol)
        yield tmp

def run_iteration(benchmark,M):
    #print "\nITERATION %d\n" % M
    filename = os.path.join(benchmark, "dataset", "%d.txt" % M)
    dataset = list(readBenchmark(filename))

    d = []
    for data in dataset:
        d.append([chembl.smiles_lookup[x] for x in data])
    evaluate_similarity_method(d, os.path.join(benchmark, "similarities", str(M)))

if __name__ == "__main__":
    i = int(argv[1])
    print local_location
    for benchmark in ["SingleAssay", "MultiAssay"]:
        # Note that the following loop is completely parallelisable
        # e.g. you could run from 0->500 on one CPU and from 500->1000 on
        #      another to finish in half the time
        if not os.path.isdir(os.path.join(local_location, benchmark, "similarities")):
            os.mkdir(os.path.join(local_location, benchmark, "similarities"))
        #iters = range(1000);
        #Parallel(n_jobs=num_cores)(delayed(run_iteration)(benchmark,i) for i in iters)
        run_iteration(os.path.join(local_location, benchmark), i)
        #for M in range(1000):
        #    print "\nITERATION %d\n" % M
        #    filename = os.path.join(benchmark, "dataset", "%d.txt" % M)
        #    dataset = list(readBenchmark(filename))

        #    d = []
        #    for data in dataset:
        #        d.append([chembl.smiles_lookup[x] for x in data])
        #    evaluate_similarity_method(d, os.path.join(benchmark, "similarities", str(M)))
