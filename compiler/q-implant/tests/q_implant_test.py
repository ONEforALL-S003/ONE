#!/usr/bin/env python3
import argparse
import subprocess
import os

from test_utils import TestCase
from test_utils import TestRunner
from test_utils import gen_random_tensor

class Conv2D_000_Q8(TestCase):
    def __init__(self):
        self.name = "Conv2D_000_Q8"

    def generate(self) -> dict:
        json_content = dict()

        # Generate ifm
        json_content['ifm'] = gen_random_tensor(
            "uint8",  # dtype_str
            (1),  # scale_shape
            (1),  # zerop_shape
            0)  # quantized_dimension

        # Generate ker
        json_content['ker'] = gen_random_tensor(
            "uint8",  #dtype_str
            (1),  # scale_shape
            (1),  # zerop_shape
            0,  # quantized_dimension
            (1, 1, 1, 2))  # value_shape (OHWI)

        # Generate bias
        json_content['bias'] = gen_random_tensor(
            "int32",  #dtype_str
            (1),  # scale_shape
            (1),  # zerop_shape
            0,  # quantized_dimension
            (1))  # value_shape

        # Generate ofm
        json_content['ofm'] = gen_random_tensor(
            "uint8",  # dtype_str
            (1),  # scale_shape
            (1),  # zerop_shape
            0)  # quantized_dimension

        return json_content
    
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--driver', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
driver = args.driver
model = args.model

input_circle = input_dir + '.circle'
output_circle = output_dir + '/Conv2D_000_Q8/output.circle'
qparam_dir = output_dir + '/Conv2D_000_Q8/qparam.json'

if not os.path.exists(input_circle):
    print('fail to load input circle')
    quit(255)

test_runner = TestRunner(output_dir)

test_runner.register(Conv2D_000_Q8())

test_runner.run()

if not os.path.exists(qparam_dir):
    print('qparam generate fail')
    quit(255)

subprocess.run(
    [
        driver, input_circle, qparam_dir, output_circle
    ],
    check=True)

if not os.path.exists(output_circle):
    print('output circle generate fail')
    quit(255)

quit(0)