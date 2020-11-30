import pytest
import torch
import numpy as np
import os
import shutil

from onnxruntime.training import _binary

# Helper functions

def equals(a, b):
    if isinstance(a, dict):
        for key in a:
            if key not in b or not equals(a[key], b[key]):
                return False
        return True
    else:
        are_equal = a == b
        return are_equal if isinstance(are_equal, bool) else are_equal.all()
    
    return False

def numpy_types(obj_value):
    if not isinstance(obj_value, dict):
        return type(obj_value).__module__ == np.__name__ or isinstance(obj_value, str)

    for _, value in obj_value.items():
        if not numpy_types(value):
            return False
    return True

def get_dict(separated_key):
    test_dict = {'int1':1, 'int2': 2, 'int_list': [1,2,3,5,6], 'dict1': {'np_array': np.arange(100), 'dict2': {'int3': 3, 'int4': 4}, 'str1': "onnxruntime"}, 'bool1': bool(True), 'int5': 5, 'float1': 2.345, 'np_array_float': np.array([1.234, 2.345, 3.456]), 'np_array_float_3_dim': np.array([[[1,2],[3,4]], [[5,6],[7,8]]])}
    key = ''
    expected_val = test_dict
    for single_key in separated_key:
        key += single_key + '/'
        expected_val = expected_val[single_key]
    return test_dict, {'key': key} if len(separated_key) > 0 else dict(), expected_val

# Test fixtures

@pytest.fixture(scope="function")
def binary_test_setup():
    binary_dir = os.path.abspath('binary_dir/')
    if not os.path.exists(binary_dir):
        os.makedirs(binary_dir, exist_ok = True)
    pytest.binary_path = os.path.join(binary_dir, 'binary.ortcp')
    yield 'binary_test_setup'
    shutil.rmtree(binary_dir)

@pytest.fixture(scope="function")
def binary_parameterized_setup(request, binary_test_setup):
    yield request.param

# Tests

@pytest.mark.parametrize("binary_parameterized_setup", [
    get_dict([]),
    get_dict(['int1']),
    get_dict(['dict1']),
    get_dict(['dict1', 'dict2']),
    get_dict(['dict1', 'dict2', 'int4']),
    get_dict(['dict1', 'str1']),
    get_dict(['bool1']),
    get_dict(['float1']),
    get_dict(['np_array_float'])], indirect=True)
def test_binary_saved_dict_matches_loaded(binary_parameterized_setup):
    to_save = binary_parameterized_setup[0]
    args = binary_parameterized_setup[1]
    expected = binary_parameterized_setup[2]
    _binary.save(to_save, pytest.binary_path)
    loaded = _binary.load(pytest.binary_path, **args)
    assert equals(loaded, expected)
    assert numpy_types(loaded)

@pytest.mark.parametrize("binary_parameterized_setup", [
    {'str_list': ['one', 'two']},
    {'int_set': {1, 2, 3, 4, 5}},
    {'str_set': {'one', 'two'}},
    {'np_str_array': np.array(['apples', 'banana'])},
    [1, 2, 3],
    2.352], indirect=True)
def test_binary_saving_non_supported_types_fails(binary_parameterized_setup):
    to_save = binary_parameterized_setup
    with pytest.raises(Exception):
        _binary.save(to_save, pytest.binary_path)

@pytest.mark.parametrize("binary_parameterized_setup", [
    ({'int64_tensor': torch.tensor(np.arange(100))}, 'int64_tensor', torch.int64, np.int64),
    ({'int32_tensor': torch.tensor(np.arange(100), dtype=torch.int32)}, 'int32_tensor', torch.int32, np.int32),
    ({'int16_tensor': torch.tensor(np.arange(100), dtype=torch.int16)}, 'int16_tensor', torch.int16, np.int16),
    ({'int8_tensor': torch.tensor(np.arange(100), dtype=torch.int8)}, 'int8_tensor', torch.int8, np.int8),
    ({'float64_tensor': torch.tensor(np.array([1.0, 2.0]))}, 'float64_tensor', torch.float64, np.float64),
    ({'float32_tensor': torch.tensor(np.array([1.0, 2.0]), dtype=torch.float32)}, 'float32_tensor', torch.float32, np.float32),
    ({'float16_tensor': torch.tensor(np.array([1.0, 2.0]), dtype=torch.float16)}, 'float16_tensor', torch.float16, np.float16)], indirect=True)
def test_binary_saving_tensor_datatype(binary_parameterized_setup):
    tensor_dict = binary_parameterized_setup[0]
    tensor_name = binary_parameterized_setup[1]
    tensor_dtype = binary_parameterized_setup[2]
    np_dtype = binary_parameterized_setup[3]

    _binary.save(tensor_dict, pytest.binary_path)

    loaded = _binary.load(pytest.binary_path)
    assert isinstance(loaded[tensor_name], np.ndarray)
    assert tensor_dict[tensor_name].dtype == tensor_dtype
    assert loaded[tensor_name].dtype == np_dtype
    assert (tensor_dict[tensor_name].numpy() == loaded[tensor_name]).all()

@pytest.mark.parametrize("binary_parameterized_setup", [
    ({'two_dim': torch.ones([2, 4], dtype=torch.float64)}, 'two_dim'),
    ({'three_dim': torch.ones([2, 4, 6], dtype=torch.float64)}, 'three_dim'),
    ({'four_dim': torch.ones([2, 4, 6, 8], dtype=torch.float64)}, 'four_dim')], indirect=True)
def test_binary_saving_multiple_dimension_tensors(binary_parameterized_setup):
    tensor_dict = binary_parameterized_setup[0]
    tensor_name = binary_parameterized_setup[1]

    _binary.save(tensor_dict, pytest.binary_path)

    loaded = _binary.load(pytest.binary_path)
    assert isinstance(loaded[tensor_name], np.ndarray)
    assert (tensor_dict[tensor_name].numpy() == loaded[tensor_name]).all()

@pytest.mark.parametrize("binary_parameterized_setup", [
    {},
    {'a': {}},
    {'a': {'b': {}}}], indirect=True)
def test_binary_saving_and_loading_empty_dictionaries_succeeds(binary_parameterized_setup):
    saved = binary_parameterized_setup
    _binary.save(saved, pytest.binary_path)

    loaded = _binary.load(pytest.binary_path)
    assert equals(saved, loaded)

def test_binary_load_file_that_does_not_exist_fails(binary_test_setup):
    with pytest.raises(Exception):
        _binary.load(pytest.binary_path)
