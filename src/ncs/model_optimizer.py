import tensorflow as tf
import time
import numpy as np
from openvino.runtime import Core
from hummingbird.ml import convert
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


#model_name = "second"
#model_path = Path('../../saved_models/ncs/test/partial')
#ir_model_name = "partial_ir"

# Construct the command for Model Optimizer
#mo_command = f"""mo --saved_model_dir "../../saved_models/ncs/test/first" --input_shape "[1,4]" --output_dir "../../saved_models/ncs/test/first" --model_name "first_ir"
#              """
#mo_command = " ".join(mo_command.split())
#print("Model Optimizer command to convert TensorFlow to OpenVINO:")
#print(mo_command)

forest = RandomForestClassifier(n_estimators=100, max_depth=10)

X, y = make_classification(n_samples=1300, n_features=8,
                               n_informative=8, n_redundant=0,
                               random_state=0, shuffle=True,
                               n_classes=4)

x_train, y_train = X[:1000], y[:1000]
x_test, y_test = X[1000:], y[1000:]
forest.fit(x_train, y_train)

hb_model = convert(forest, 'torch', extra_config={"tree_implementation":"tree_trav"})

x = np.array([[1,2,3,4,5,6,7,8]], dtype=np.float32)
start = time.perf_counter()
skl_result = forest.predict_proba(x)
time_skl = (time.perf_counter() - start) * 1000

model_types = {0 : "GEMM", 1 : "Tree Traversal", 2 : "Perfect Tree Traversal"}

# Multi Batch GEMM Model
model_xml = f"../../saved_models/ncs/test/ptt_test/ptt_test.xml"
MODEL_TYPE = 2

# GEMM Model
#model_xml = f"../../saved_models/ncs/test/bigger/bigger_model.xml"
#MODEL_TYPE = 0

# Load model
ie = Core()
model = ie.read_model(model=model_xml)
#print(ie.available_devices)

res = []

# Neural Compute Stick
compiled_model = ie.compile_model(model=model, device_name="MYRIAD")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

x = [np.array([[1,2,3,4,5,6,7,8]], dtype=np.float32).tolist()]

multi_x = [tf.reshape(np.arange(1, 16*8+1, dtype=np.float32).tolist(), (16, 8))]
start = time.perf_counter()
res = compiled_model(x)[output_layer]
inf_time_s = time.perf_counter() - start
inf_time_ms = inf_time_s * 1000
ncs = inf_time_ms
    

# NCS and CPU together
compiled_model = ie.compile_model(model=model, device_name="MULTI:CPU,MYRIAD")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

start = time.perf_counter()
res_multi = compiled_model(x)[output_layer]
inf_time = (time.perf_counter() - start) * 1000
multi = inf_time

print(f'\n\nModeltype: {model_types[MODEL_TYPE]}')

print(f'----------------Inference Results----------------')
print(f'NCS:\n{res}')
print(f'SKL:\n{skl_result}')

compiled_model = ie.compile_model(model=model, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
#x = np.array([[1,2,3,4]], dtype=np.float32)
x = [np.array([[1,2,3,4,5,6,7,8]], dtype=np.float32).tolist()]

start = time.perf_counter()
res = compiled_model(x)[output_layer]
inf_time_s = time.perf_counter() - start
inf_time_ms = inf_time_s * 1000
cpu = inf_time_ms
print(f'CPU:\n{res}')
print(f'MULTI:\n{res_multi}')


print(f'----------------Inference Times----------------')
print(f'NCS2: {ncs : .2f}ms\tCPU: {cpu : .2f}ms\tSKL: {time_skl : .2f}ms\tMULTI: {multi : .2f}ms')
