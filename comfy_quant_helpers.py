import torch
import json

def dict_to_tensor(data_dict):
    json_str = json.dumps(data_dict)
    byte_data = json_str.encode('utf-8')
    tensor_data = torch.tensor(list(byte_data), dtype=torch.uint8)
    return tensor_data

def tensor_to_dict(tensor_data):
    byte_data = bytes(tensor_data.tolist())
    json_str = byte_data.decode('utf-8')
    data_dict = json.loads(json_str)
    return data_dict
