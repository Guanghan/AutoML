"""
@author: Guanghan Ning
@file: test_read_genotypes.py.py
@time: 12/4/20 8:01 下午
@file_desc: Read genotypes in json file, parse to python
"""
import json
def read_json_from_file(input_path):
    with open(input_path, "r") as read_file:
        python_data = json.load(read_file)
    return python_data

json_path = "/Users/ngh/Desktop/dev/AutoML/res/latency_14ms.json"
desc = read_json_from_file(json_path)
print(desc)

normal = desc['super_network']['normal']['genotype']
reduce = desc['super_network']['reduce']['genotype']
print(normal)

normal = [tuple(element) for element in normal]
reduce = [tuple(element) for element in reduce]
print(normal)


