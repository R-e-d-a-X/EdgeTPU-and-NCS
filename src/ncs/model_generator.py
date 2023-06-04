import tensorflow as tf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import conversion_tf as conv
from hummingbird.ml import convert
import time

def generate_model(bs, nf, nt, md, name, model):
    # variables 
    BATCH_SIZE = bs
    N_FEATURES = nf
    N_TREES = nt
    MAX_DEPTH = md
    
    # init random forest
    forest = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH)
    
    tf.config.run_functions_eagerly(True)

    X, y = make_classification(n_samples=1500, n_features=8,
                               n_informative=6, n_redundant=0,
                               random_state=0, shuffle=True,
                               n_classes=4)

    x_train, y_train = X, y

    forest.fit(x_train, y_train)

    # initialize model
    if model == 'GEMM':
        model = conv.GEMMDecisionTreeImplKeras(forest)
        op = convert(forest, 'torch', extra_config={"tree_implementation":"gemm"}).model._operators[0]
    elif model == 'TT':
        model = conv.TreeTraversalDecisionTreeImplKeras(forest)
        op = convert(forest, 'torch', extra_config={"tree_implementation":"tree_trav"}).model._operators[0]
    elif model == 'PTT':
        model = conv.PerfectTreeTraversalDecisionTreeImplKeras(forest)
        op = convert(forest, 'torch', extra_config={"tree_implementation":"perf_tree_trav"}).model._operators[0]
    elif model == 'Data':
        model = conv.DataTransferTest()

    x = (np.abs(X.min()) + np.abs(X.max())) * np.random.random_sample((BATCH_SIZE,8)) + X.min()

    start = time.perf_counter()
    test = model(x)
    inf_time = (time.perf_counter() - start) * 1000

    print(inf_time)
    model.save(f'../../saved_models/ncs/benchmarks/{name}')

    # get model prediction
    #print(test)

        
def main():
    bsizes = [1, 32, 64, 128, 256, 512, 1024] 
    mdepth =  [1,  4,  4,  8,   8,  16,  16]
    n_trees = [1, 16, 32, 64, 128, 256, 512]
    params = list(zip(mdepth, n_trees))
    types = ['GEMM', 'TT', 'PTT']

    #for (depth, ntrees) in params:
    #    for t in types:
    #        name = f"{t}_scaling_both_{ntrees}_{depth}"
    #        generate_model(1, 8, ntrees, depth, name, t)

    for bs in bsizes:
        for t in types:
            name = f"{t}_scaling_batch_{bs}"
            generate_model(bs, 8, 100, 8, name, t)

    #name = input("Modelname: ")
    #type = input("Modeltype <GEMM> or <TT> or <PTT>: ")
    #bsize = int(input("Batchsize: "))
    #nfeat = int(input("Number of Features: "))
    #mdepth = int(input("Max Depth: "))
    #n_trees = int(input("Number of Trees: "))
    
    


if __name__ == '__main__':
    main()