import time
import numpy as np
from openvino.runtime import Core

model_xml = f"../../benchmarks/ncs/"

def main():
    model_name = input("Modelname: ")
    model_xml = f"../../benchmarks/ncs/{model_name}.xml"
    n_iter = int(input("Iterations: "))
    device = int(input("Device <0 for NCS> <1 for CPU> "))

    device_map = {0 : "MYRIAD", 1 : "CPU"}

    ie = Core()
    model = ie.read_model(model=model_xml)

    ie.set_property({'CACHE_DIR': '/home/tobi/Bachelorarbeit/'})
    ie.set_property('MYRIAD', {'PERFORMANCE_HINT' : 'LATENCY'})
    ie.set_property('MYRIAD', {"MYRIAD_ENABLE_HW_ACCELERATION": "YES"})

    compiled_model = ie.compile_model(model=model, device_name=device_map[device])
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    inf_times = []
    for iteration in range(n_iter + 1):
        if iteration % 100 == 0:
            print(f"{(iteration+1) / (n_iter+1) * 100 : .2f}% done")
        if device == 0:
            x = [14 * np.random.random_sample((1,8)) - 7.135]
        else:
            x = [14 * np.random.random_sample((1,8)) - 7.135]
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

    inf_times.sort()

    time_med = 0
    if n_iter % 2 == 0:
        idx1 = int((n_iter / 2) - 1)
        idx2 = int(n_iter / 2)
        time_med = (inf_times[idx1] + inf_times[idx2]) / 2
    else:
        idx = n_iter // 2
        time_med = inf_times[idx]

    print(f'------------------------Results-----------------------------')
    print(f'Count: {n_iter} iterations')
    print(f'Duration: {time_total : .2f}ms')
    print(f'Latency:')
    print(f'\tMean: {time_mean - 1.57 : .2f}ms\t\t({time_mean : .2f}ms)')
    print(f'\tMin: {time_min: .2f}ms')
    print(f'\tMax: {time_max: .2f}ms')
    print(f'\tMedian: {time_med - 1.57 : .2f}ms')
    print(f'Throughput: {throughput : .2f} FPS')

if __name__ == '__main__':
    main()