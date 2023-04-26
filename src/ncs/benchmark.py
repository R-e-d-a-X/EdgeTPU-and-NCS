import tensorflow as tf
import time
import numpy as np
from openvino.runtime import Core
from hummingbird.ml import convert
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

model_xml = f"../../saved_models/ncs/test/tt-test/tt-test.xml"

ie = Core()
model = ie.read_model(model=model_xml)
caching_supported = 'EXPORT_IMPORT' in ie.get_property("MYRIAD", 'OPTIMIZATION_CAPABILITIES')

print(caching_supported)
ie.set_property({'CACHE_DIR': '/home/tobi/Bachelorarbeit/'})
ie.set_property('MYRIAD', {'PERFORMANCE_HINT' : 'THROUGHPUT'})
ie.set_property('MYRIAD', {"PERFORMANCE_HINT_NUM_REQUESTS": "4"})
ie.set_property('MYRIAD', {"MYRIAD_ENABLE_HW_ACCELERATION": "YES"})
ie.set_property('MYRIAD', {"MYRIAD_ENABLE_RECEIVING_TENSOR_TIME": "YES"})
ie.set_property('CPU', {'PERFORMANCE_HINT' : 'THROUGHPUT'})
ie.set_property('CPU', {"PERFORMANCE_HINT_NUM_REQUESTS": "4"})

config = {"NUM_STREAMS": "3"}

# Neural Compute Stick
compiled_model = ie.compile_model(model=model, device_name="MYRIAD", config=config)
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

print(compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS"))

x = [np.array([[1,2,3,4,5,6,7,8]], dtype=np.float32).tolist()]
inf_times = []
for iteration in range(1001):
    x = [9 * np.random.random_sample((1,8)) + 1]
    start = time.perf_counter()
    res = compiled_model(x)[output_layer]
    inf_time_s = time.perf_counter() - start
    inf_time_ms = inf_time_s * 1000
    if iteration > 0:
        inf_times.append(inf_time_ms)
    
inf_times = np.array(inf_times)
time_total = inf_times.sum(0)
time_mean = inf_times.mean()
time_min = inf_times.min()
time_max = inf_times.max()
throughput = 1000 * (1000 / time_total)

print(f'------------------------Results-----------------------------')
print(f'Count: {1000} iterations')
print(f'Duration: {time_total : .2f}ms')
print(f'Latency:')
print(f'\tMean: {time_mean : .2f}ms')
print(f'\tMin: {time_min : .2f}ms')
print(f'\tMax: {time_max : .2f}ms')
print(f'Throughput: {throughput : .2f} FPS')



