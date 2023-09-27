#!/usr/bin/env python3
import h5py as h5
import numpy as np
import argparse
import os.path
import json
import sys


def validate(h5_path, qparam_dir, qparam_json):
    valid = True
    with open(qparam_json, "r") as qparams:
        json_load = json.load(qparams)
    with h5.File(h5_path, "r") as model:
        for node_name in model.keys():
            # not quantized node exists (reshape, pad...)
            if not json_load.get(node_name):
                continue

            for tensor_name in json_load[node_name]:
                np_path = f"{qparam_dir}/{json_load[node_name][tensor_name]}"
                if tensor_name == "value":
                    expected_weights = np.load(np_path)
                    h5_weights = model[node_name]["weights"][:]
                    if np.allclose(h5_weights, expected_weights, rtol=1.e-5, atol=1.e-5) == False:
                        print("Implanted weights of " + node_name + "." + tensor_name + " (" + str(h5_weights) +
                            ") do not match with expected value (" + str(expected_weights) + ").")
                        valid = False

                if tensor_name == "scale":
                    expected_scale = np.load(np_path)
                    h5_scale = model[node_name]["scale"][:]
                    if np.allclose(h5_scale, expected_scale, rtol=1.e-5, atol=1.e-5) == False:
                        print("Implanted scale of " + node_name + "." + tensor_name + " (" + str(h5_scale) +
                            ") do not match with expected value (" + str(expected_scale) + ").")
                        valid = False

                if tensor_name == "zerop":
                    expected_zerop = np.load(np_path)
                    input_zerop = model[node_name]["zero_point"][:]
                    if np.allclose(
                            input_zerop, expected_zerop, rtol=0, atol=1) == False:
                        print("Implanted zero point of " + tensor_name + " (" +
                            str(input_zerop) + ") do not match with expected value (" +
                            str(expected_zerop) + ").")
                        valid = False

    return valid
