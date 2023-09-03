# It's just the sample Tensorflow Lite Quantization Parameter Exporter
# It uses Tensorflow Lite Interpreter to export Quantization Parameter
# But the tensorflow Lite Interpreter have problem to get whole tensor data
# So, DO NOT COMMIT/PUSH TO THE ORIGINAL SAMSUNG ONE REPOSITORY
# Just use it to check/develop Q-Implant
import os
import tensorflow as tf
import numpy as np
import json


class TFParamExporter:
    def __save_np(self, data):
        file_name = str(self.__np_idx) + ".npy"
        np.save(os.path.join(self.__dir_path, file_name), data)
        self.__np_idx += 1
        return file_name

    def __init__(self, json_path, model_path=None, model_content=None):
        self.__json_path = json_path
        idx = json_path.rfind(os.path.sep)
        if idx == -1:
            self.__dir_path = ""
        else:
            self.__dir_path = json_path[:idx + 1]
        self.__np_idx = 0
        self.__interpreter = tf.lite.Interpreter(model_path=model_path, model_content=model_content, experimental_preserve_all_tensors=True)
        self.__interpreter.allocate_tensors()

    def save(self):
        data = {}
        if not os.path.exists(self.__dir_path):
            os.makedirs(self.__dir_path)
        details = self.__interpreter.get_tensor_details()
        sig = self.__interpreter.get_signature_list()
        for i in range(len(details)):
            detail = details[i]
            idx = detail['index']
            # Tensorflow Lite Interpreter has problem to read Tensor Data, check issue below
            # https://github.com/tensorflow/tensorflow/issues/57971
            # When check Tensorflow Lite Interpreter Wrapper code,
            # The error occurred the raw tensor data is null even tensor size is not 0
            # Maybe there is problem to allocate tensor
            # (Because the raw tensor data is null, even we call allocate_tensor
            # https://github.com/tensorflow/tensorflow/blob/41b0df1efb2ad226dc178c366c68a8d78514e454/tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc#L656C8-L656C8
            # Maybe we need to use flatbuffers instead of using the tensorflow lite interpreter
            try:
                tensor = self.__interpreter.get_tensor(idx)
            except:
                continue
            param = detail['quantization_parameters']
            if param['scales'].size == 0 and param['zero_points'].size == 0:
                continue
            datum = {}
            datum['dtype'] = np.dtype(detail['dtype']).name
            datum['scale'] = self.__save_np(param['scales'])
            datum['zerop'] = self.__save_np(param['zero_points'])
            datum['value'] = self.__save_np(tensor)
            data[detail['name']] = datum
        with open(self.__json_path, 'w') as json_file:
            json.dump(data, json_file)
