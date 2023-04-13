import os
import sys
from pathlib import Path
import time

import numpy as np
from openvino.runtime import Core

REDUCE_SUM = True

#model_name = "second"
#model_path = Path('../../saved_models/ncs/test/partial')
#ir_model_name = "partial_ir"

# Construct the command for Model Optimizer
#mo_command = f"""mo --saved_model_dir "../../saved_models/ncs/test/first" --input_shape "[1,4]" --output_dir "../../saved_models/ncs/test/first" --model_name "first_ir"
#              """
#mo_command = " ".join(mo_command.split())
#print("Model Optimizer command to convert TensorFlow to OpenVINO:")
#print(mo_command)

model_xml = f"../../saved_models/ncs/test/ncs_test/ncs_test.xml"

# Load model
ie = Core()
model = ie.read_model(model=model_xml)
#print(ie.available_devices)

# Neural Compute Stick
compiled_model = ie.compile_model(model=model, device_name="MYRIAD")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)


x = [np.array([[1,2,3,4]], dtype=np.float32).tolist()]
ncs = []
for _ in range(5):
    start = time.perf_counter()
    res = compiled_model(x)[output_layer]
    inf_time_s = time.perf_counter() - start
    inf_time_ms = inf_time_s * 1000
    ncs.append(inf_time_ms)

print(f'----------------Inference Results----------------')
print(f'NCS:\n{res}')

if not REDUCE_SUM:
    compiled_model = ie.compile_model(model=model, device_name="CPU")

    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    #x = [np.array([[[[1.],[2.],[3.],[4.]]]], dtype=np.float32).tolist()]
    x = [np.array([[1,2,3,4]], dtype=np.float32).tolist()]
    
    start = time.perf_counter()
    res = compiled_model(x)[output_layer]
    inf_time_s = time.perf_counter() - start
    inf_time_ms = inf_time_s * 1000

    cpu = inf_time_ms

    print(f'CPU:\n{res}')
else:
    print(f'CPU: NA')
print(f'----------------Inference Times----------------')
if not REDUCE_SUM:
    print(f'NCS2: {ncs : .2f}ms\t\tCPU: {cpu : .2f}ms')
else:
    #print(f'NCS2: {ncs : .2f}ms\t\tCPU: NA')
    print(ncs)