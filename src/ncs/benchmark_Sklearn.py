import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pyRAPL
import pandas as pd

pyRAPL.setup()

dataoutput= pyRAPL.outputs.DataFrameOutput()

n_trees = int(input("N_TREES: "))
max_depth = int(input("MAX_DEPTH: "))
batch_size = int(input("BATCH_SIZE: "))
n_iter = int(input("N_ITER: "))
test_type = int(input('Type: <0 for Latency> <1 for Energy> '))

def bench(N_TREES, MAX_DEPTH, BATCH_SIZE, N_ITER, ttype):
    forest = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH)

    X, y = make_classification(n_samples=1500, n_features=8,
                               n_informative=6, n_redundant=0,
                               random_state=0, shuffle=True,
                               n_classes=4)
    x_train, y_train = X, y
    forest.fit(x_train, y_train)

    if ttype == 0:
        inf_times = []
        for iteration in range(N_ITER+1):
            x = (np.abs(X.min()) + np.abs(X.max())) * np.random.random_sample((BATCH_SIZE,8)) + X.min()
            start = time.perf_counter()
            res = forest.predict(x)
            inf_time_s = time.perf_counter() - start
            inf_time_ms = inf_time_s * 1000
            if iteration > 0:
                inf_times.append(inf_time_ms)
            if iteration % 100 == 0:
                print(f"{(iteration+1) / (N_ITER+1) * 100 : .2f}% done")

        inf_times = np.array(inf_times)
        time_total = inf_times.sum(0)
        time_mean = inf_times.mean()
        time_min = inf_times.min()
        time_max = inf_times.max()
        throughput = 1000 * (1000 / time_total)

        print(f'------------------------Results-----------------------------')
        print(f'Count: {N_ITER} iterations')
        print(f'Duration: {time_total : .2f}ms')
        print(f'Latency:')
        print(f'\tMean: {time_mean : .2f}ms')
        print(f'\tMin: {time_min : .2f}ms')
        print(f'\tMax: {time_max : .2f}ms')
        print(f'Throughput: {throughput : .2f} FPS')
    elif ttype == 1:
        for iteration in range(N_ITER):
            x = (np.abs(X.min()) + np.abs(X.max())) * np.random.random_sample((BATCH_SIZE,8)) + X.min()
            res = infer(forest, x)
        
        data = pd.DataFrame(dataoutput.data)
        values = data['pkg'].to_numpy()

        print(f'------------------------Results-----------------------------')
        print(f'\tMean: {values.mean() / 1000 : .2f}mJ')
        print(f'\tMin: {values.min() / 1000 : .2f}mJ')
        print(f'\tMax: {values.max() / 1000 : .2f}mJ')

@pyRAPL.measureit(number=n_iter, output=dataoutput)
def infer(forest, x):
    return forest.predict(x)

def main():
    bench(n_trees, max_depth, batch_size, n_iter, test_type)

if __name__ == '__main__':
    main()


