import pickle
from datetime import datetime

from zigzag import api
from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)
from zigzag.visualization.results.print_mapping import print_mapping
import argparse

temporal_mapping_search_engine = "loma"
def run_zigzag(model: str = "", accel: str = "", mapping: str = ""):
    workload_path = f"zigzag/inputs/workload/{model}.yaml"
    accelerator_path = f"zigzag/inputs/hardware/{accel}.yaml"
    mapping_path = f"zigzag/inputs/mapping/{mapping}.yaml"
    pickle_filename = f"outputs/{model}-{accel}-saved_list_of_cmes.pickle"
    energy, latency, cmes = api.get_hardware_performance_zigzag(
        workload=workload_path,
        accelerator=accelerator_path,
        mapping=mapping_path,
        temporal_mapping_search_engine=temporal_mapping_search_engine,
        opt="latency", # "energy"
        dump_folder=f"outputs/{datetime.now()}-{model}-{accel}-{mapping}",
        pickle_filename=pickle_filename,
    )
    print(f"Total network energy = {energy:.2e} pJ")
    print(f"Total network latency = {latency:.2e} cycles")

    with open(pickle_filename, "rb") as fp:
        cmes = pickle.load(fp)
    bar_plot_cost_model_evaluations_breakdown(cmes[0:5], save_path=f"outputs/{model}-{accel}-{mapping}-plot_breakdown.png")
    print_mapping(cmes[0])

if __name__ == "__main__":
    # add model and accel and mapping as parsed arguments of the python script
    # parser = argparse.ArgumentParser(description="Run ZigZag with specified model, accelerator, and mapping.")
    # parser.add_argument("--model", type=str, required=True, help="The model to be used.")
    # parser.add_argument("--accel", type=str, required=True, help="The accelerator to be used.")
    # parser.add_argument("--mapping", type=str, required=True, help="The mapping to be used.")
    
    # args = parser.parse_args()

    # run_zigzag(model=args.model, accel=args.accel, mapping=args.mapping)
    # run_zigzag(model="UNETConvGeMM512x512", accel="UNET_accel_hw", mapping= "UNET_accel_mapping")
    # run_zigzag(model="UNETGeMM512x512", accel="UNET_GeMM_accelerator", mapping= "UNET_GeMM_mapping")
    # run_zigzag(model="UNETConv512x512", accel="UNET_Conv_accelerator", mapping= "UNET_Conv_mapping")
    # run_zigzag(model="UNETConvGeMM512x512", accel="UNET_ConvGeMM_accelerator", mapping= "UNET_ConvGeMM_mapping")
    # run_zigzag(model="UNETConv512x512", accel="default_Gemmini_accelerator", mapping= "default_Gemmini_mapping1")

    # run_zigzag(model="UNETConv512x512", accel="default_Gemmini_accelerator", mapping= "default_Gemmini_mapping1")
    
    # run_zigzag(model="UNETConv512x512", accel="default_Gemmini_accelerator", mapping= "default_Gemmini_mapping1_mergeOY")
    
    # run_zigzag(model="UNETConv512x512", accel="default_Gemmini_accelerator", mapping= "default_Gemmini_mapping1_temporalDefined")
    # run_zigzag(model="UNETConv512x512", accel="default_Gemmini_accelerator", mapping= "default_Gemmini_mapping1_temporalDefined_sameOrder_justSplitted")

    # run_zigzag(model="UNETConv512x512", accel="default_Gemmini_accelerator", mapping= "default_Gemmini_mapping3")

    # ############################################
    run_zigzag(model="UNETStaticMatMul512x512", accel="default_Gemmini_accelerator_OS", mapping= "default_Gemmini_mapping_StaticMatMul")
