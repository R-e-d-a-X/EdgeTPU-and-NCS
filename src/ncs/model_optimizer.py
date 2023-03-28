import os
import sys
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from openvino.runtime import Core
from openvino.tools.mo import mo_tf

model_name = "first"
model_path = Path('../../saved_models/ncs/test/first')
ir_model_name = "first_ir"

# Get the path to the Model Optimizer script

# Construct the command for Model Optimizer
mo_command = f"""mo
                 --saved_model_dir "../../saved_models/ncs/test/first"
                 --input_shape "[1, 4]"
                 --output_dir "../../saved_models/ncs/test/first"
                 --model_name "first_ir"
                 """
mo_command = " ".join(mo_command.split())
print("Model Optimizer command to convert TensorFlow to OpenVINO:")
print(mo_command)

model_xml = f"../../saved_models/ncs/test/first/first_ir.xml"

# Load model
ie = Core()
model = ie.read_model(model=model_xml)

# Neural Compute Stick
# compiled_model = ie.compile_model(model=model, device_name="MYRIAD")
compiled_model = ie.compile_model(model=model, device_name="CPU")

del model

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

start = time.perf_counter()
res = compiled_model(np.array([[1,2,3,4]]))[output_layer]
inf_time_s = time.perf_counter() - start
inf_time_ms = inf_time_s * 1000

print(f'OpenVino:\t\tRes: {res}\tTime: {inf_time_ms}ms')