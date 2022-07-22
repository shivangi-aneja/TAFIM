import os
import json
import torch


def write_json_data(dic, file_name, mode='a'):
    with open(file_name, mode) as output_file:
        json.dump(dic, output_file)
        output_file.write(os.linesep)


def read_json_data(file_name):
    with open(file_name) as f:
        article_list = [json.loads(line) for line in f]
        return article_list


def init_adversarial_noise(mode="zero"):
    if mode == "zero":
        return torch.zeros((3, 256, 256))
    elif mode == "rand":
        return torch.rand((3, 256, 256))
    else:
        print("Invalid Noise Type")
        exit()
