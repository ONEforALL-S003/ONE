# Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess

# To check if there is a test_output directory
output_dir = "/one/compiler/q-implant/tests/test_output"

if os.path.exists(output_dir):
    # Remove directory
    os.system(f"rm -r {output_dir}")

# Generate test qparams
generate_file = "/one/compiler/q-implant/tests/gen_test_data.py"
gen_command = ["python3", generate_file, "--output_dir", output_dir]
try:
    subprocess.run(gen_command, check=True)
    print(f"test files are generated in {output_dir}")
except subprocess.CalledProcessError as e:
    print(f"Error while generating qparams : {e}")

    # test fail
    quit(1)

# Make tflite file using tensorflow recipe
recipe_dir = "/one/res/TensorFlowLiteRecipes/Conv2D_000/test.recipe"
chef_dir = "/one/build/compiler/tflchef/tools/console/tflchef"
tflite_path = f"{output_dir}/test.tflite"
tflite_command = f"cat {recipe_dir} | {chef_dir} > {tflite_path}"
try:
    subprocess.run(tflite_command, shell=True, check=True, executable="/bin/bash")
    print(f"test.tflite file are generated in {output_dir}")
except subprocess.CalledProcessError as e:
    print(f"Error while generating tflite file : {e}")
    quit(1)

# Execute tflite2circle
tflite2circle_file = "/one/build/compiler/tflite2circle/tflite2circle"
input_circle_path = f"{output_dir}/input.circle"
tflite2circle_command = f"{tflite2circle_file} {tflite_path} {input_circle_path}"
try:
    subprocess.run(tflite2circle_command, shell=True, check=True, executable="/bin/bash")
    print(f"test.tflite file is converted to input.circle")
except subprocess.CalledProcessError as e:
    print(f"Error while converting tflite file : {e}")
    quit(1)

# Execute q-implant
q_implant_file = "/one/build/compiler/q-implant/q-implant"
output_circle_path = f"{output_dir}/output.circle"
q_implant_command = f"{q_implant_file} {input_circle_path} {output_dir}/*/qparam.json {output_circle_path}"
try:
    subprocess.run(q_implant_command, shell=True, check=True, executable="/bin/bash")
    print(f"input.circle is quantized to output.circle by q-implant")
except subprocess.CalledProcessError as e:
    print(f"Error while quantizing circle file : {e}")
    quit(1)

# test OK
quit(0)