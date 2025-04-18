import pickle
from datetime import datetime

from zigzag import api
from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)
from zigzag.visualization.results.print_mapping import print_mapping

model = "UNET"
accel = "tpu_like"
workload_path = "zigzag/inputs/workload/UNET_layer_by_layer_64x64.yaml"
accelerator_path = "zigzag/inputs/hardware/tpu_like.yaml"
mapping_path = "zigzag/inputs/mapping/tpu_like.yaml"
pickle_filename = f"outputs/{model}-{accel}-saved_list_of_cmes.pickle"


energy, latency, cmes = api.get_hardware_performance_zigzag(
    workload=workload_path,
    accelerator=accelerator_path,
    mapping=mapping_path,
    opt="latency", # "energy"
    dump_folder=f"outputs/{datetime.now()}",
    pickle_filename=pickle_filename,
)
print(f"Total network energy = {energy:.2e} pJ")
print(f"Total network latency = {latency:.2e} cycles")

with open(pickle_filename, "rb") as fp:
    cmes = pickle.load(fp)


bar_plot_cost_model_evaluations_breakdown(cmes[0:5], save_path=f"outputs/{model}-{accel}-plot_breakdown.png")
print_mapping(cmes[0])
