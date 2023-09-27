from test_utils import TestCase
from test_utils import gen_random_tensor


class Mean_000_Q8(TestCase):
    def __init__(self):
        self.name = _name_

    def generate(self) -> dict:
        json_content = dict()

        # Generate ifm
        json_content['ifm'] = gen_random_tensor(
            "uint8",  # dtype_str
            (1),  # scale_shape
            (1),  # zerop_shape
            0)  # quantized_dimension

        # Generate reduction_indices
        # json_content['reduction_indices'] = gen_random_tensor(
        #     "uint8",  #dtype_str
        #     (1),  # scale_shape
        #     (1),  # zerop_shape
        #     0,  # quantized_dimension
        #     (1))  # value_shape (OHWI)

        # Generate ofm
        json_content['ofm'] = gen_random_tensor(
            "uint8",  # dtype_str
            (1),  # scale_shape
            (1),  # zerop_shape
            0)  # quantized_dimension

        return json_content


_name_ = 'Mean_000_Q8'

_model_ = Mean_000_Q8()
