import os
import sys
#import cPickle as pickle
#import itertools
#from collections import defaultdict, Counter
import benchlib.chembl as chembl
import benchlib.fingerprint_lib as flib # From benchmarking platform
from rdkit import Chem, DataStructs
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
    return np.load(os.path.join(local_location, "%s.npz" % fpName))

def accToWeights(acc):
    return np.log1p(acc.sum()/acc)

def createweightVector(weightData, isSparse):
    if (isSparse):
        weightVector = DataStructs.cDataStructs.MSVectorXdHelper.Create()
        DataStructs.cDataStructs.MSVectorXdHelper.Map(weightVector, np.double(weightData['col']), np.double(accToWeights(weightData['data'])), len(np.double(accToWeights(weightData['data']))))
    else:
        weightVector = DataStructs.cDataStructs.MVectorXdHelper.Create()
        DataStructs.cDataStructs.MVectorXdHelper.Map(weightVector, np.double(accToWeights(weightData['data'])), len(np.double(accToWeights(weightData['data']))))
    return weightVector

def loadWeights(fpName):
    weightVector = None
    try:
        weightVector = flib.weightVectors[fpName]
    except KeyError as e:
        weightData = load_sparse_matrix(fpName)
        weightVector = createweightVector(weightData, check_sparse(fpName))
        flib.weightVectors[fpName] = weightVector
    finally:
        return weightVector

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
            ref_fp = fpCalculator(ref_mol)
            tanimotos = []
            for smol in d[1:]:
                sfp = fpCalculator(smol)
                tanimoto = DataStructs.WeightedTanimotoSimilarity(ref_fp, sfp, wv)
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
    flib.weightVectors = {}
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
