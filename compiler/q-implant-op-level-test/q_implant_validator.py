#!/usr/bin/env python3
import h5py as h5
import numpy as np
import argparse
import os.path
import json
import sys


def validate(h5_path, qparam_dir):
    with open(qparam_dir, "r") as qparams:
        json_load = json.load(qparams)
    with h5.File(h5_path, "r") as model:
        for tensor_name in model.keys():
            print(tensor_name)
            pass

def compare_quantization(tensor, tensor_name, expect_dir):
    global failed_cases
    with open(expect_dir + "/" + tensor_name + ".json", "r") as expect_file:
        json_load = json.load(expect_file)
    for key in json_load:
        if key == "weights":
            expected_weights = np.array(json_load["weights"])
            input_weights = tensor["weights"][()]
            abs_tolerance = 1
            # We use higher tolerance for int64 data (bias of int16-quantized model)
            if tensor["weights"].dtype == 'int64':
                abs_tolerance = 5

            if np.allclose(
                    input_weights, expected_weights, rtol=0, atol=abs_tolerance) == False:
                print("Quantized weights of " + tensor_name + " (" + str(input_weights) +
                      ") do not match with expected value (" + str(expected_weights) +
                      ").")
                failed_cases += 1

        if key == "scale":
            expected_scale = np.array(json_load["scale"])
            input_scale = tensor["scale"][:]
            if np.allclose(input_scale, expected_scale, rtol=1.e-5, atol=1.e-5) == False:
                print("Quantized scale of " + tensor_name + " (" + str(input_scale) +
                      ") do not match with expected value (" + str(expected_scale) + ").")
                failed_cases += 1

        if key == "zero_point":
            expected_zero_point = np.array(json_load["zero_point"])
            input_zero_point = tensor["zero_point"][:]
            if np.allclose(
                    input_zero_point, expected_zero_point, rtol=0, atol=1) == False:
                print("Quantized zero_point of " + tensor_name + " (" +
                      str(input_zero_point) + ") do not match with expected value (" +
                      str(expected_zero_point) + ").")
                failed_cases += 1