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

    X, y = make_classification(n_samples=1300, n_features=N_FEATURES,
                               n_informative=N_FEATURES, n_redundant=0,
                               random_state=0, shuffle=True,
                               n_classes=4)

    x_train, y_train = X[:1000], y[:1000]

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


    start = time.perf_counter()
    x = np.array([[1.,2.,3.,4.,5.,6.,7.,8.], [8.,8.,8.,8.,8.,8.,8.,8.]], dtype=np.float32)
    test = model(9 * np.random.random_sample((BATCH_SIZE, 8)) + 1)
    inf_time = (time.perf_counter() - start) * 1000
    inf_time = (time.perf_counter() - start) * 1000

    print(inf_time)
    model.save(f'../../saved_models/ncs/test/{name}')

    # get model prediction
    print(test)

        
def main():
    bsizes = [1, 64, 256, 512]
    nfeat  = [2, 8, 16]
    mdepth = [1, 4, 16]
    n_trees = [1, 32, 128]

    #for max_depth in mdepth:
    #    for n_tree in n_trees:
    #        generate_model(256, 16, n_tree, max_depth)

    name = input("Modelname: ")
    type = input("Modeltype <GEMM> or <TT> or <PTT>: ")
    bsize = int(input("Batchsize: "))
    nfeat = int(input("Number of Features: "))
    mdepth = int(input("Max Depth: "))
    n_trees = int(input("Number of Trees: "))
    
    generate_model(bsize, nfeat, n_trees, mdepth, name, type)


if __name__ == '__main__':
    main()