import sys

import mat73



import scipy.io as io
import h5py
import os
import json
from glob import glob
from tqdm import tqdm
import numpy as np
import pickle
import argparse


def check_if_numpy_array(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data
def convert_dictionary(dict):
    for key in dict.keys():
        dict[key] = check_if_numpy_array(dict[key])
    return dict
parser = argparse.ArgumentParser(description='Specify task name for converting ZuCo v1.0 Mat file to Pickle')
parser.add_argument('-t', '--task_name', help='name of the task in /dataset/ZuCo, choose from {task1-SR,task2-NR,task3-TSR}', required=True)
parser.add_argument('-v', '--version', help='version of the dataset, choose from {v1,v2}', required=True)
parser.add_argument("-d","--directory", help="directory of the dataset, default: ./dataset/ZuCo", default="./dataset/ZuCo")
parser.add_argument("-o","--output", help="output directory of the converted pickle files, default: ./dataset/ZuCo/pickle", default="./dataset/ZuCo/pickle")
parser.add_argument("-m","--machine",required=True, help="machine to run the script, choose from {tehr09,tehr10,archie-west}")
args = vars(parser.parse_args())

if args["machine"] == "archie-west":
    sys.path.insert(1, "/users/wrb15144/EEG-To-Text/")
elif args["machine"] == "tehr09" or args["machine"] == "tehr10":
    sys.path.insert(1, "/home/wrb15144/zenon/EEG-To-Text/")

from util.Convertor import Convertor
from util.Reader import Reader
"""config"""
version = args["version"] # 'old'
# version = 'v2' # 'new'

task_name = args['task_name']
# task_name = 'task1-SR'
# task_name = 'task2-NR'
# task_name = 'task3-TSR'


print('##############################')
print(f'start processing ZuCo {task_name}...')



    # old version 
input_mat_files_dir = f'{args["directory"]}{task_name}/Matlab_files/'

output_dir = f'./dataset/ZuCo/{task_name}/pickle'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

"""load files"""
mat_files = glob(os.path.join(input_mat_files_dir,'*.mat'))
mat_files = sorted(mat_files)

if len(mat_files) == 0:
    print(f'No mat files found for {task_name}')
    quit()
convertor = Convertor()
reader = Reader()
dataset_dict = {}
for mat_file in tqdm(mat_files):
    subject_name = os.path.basename(mat_file).split('_')[0].replace('results','').strip()
    dataset_dict[subject_name] = []
    
    if version == 'v1':
        matdata = io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)['sentenceData']
        matdata = matdata.tolist()
    elif version == 'v2':
        #matdata = h5py.File(mat_file,'r')
        try:
            data_dict = mat73.loadmat(mat_file)
            matdata = data_dict["sentenceData"]
            length = len(matdata["word"])
        except:
            matdata = io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)['sentenceData']
            matdata = matdata.tolist()
            version = 'v1'

    if version == 'v2':
        dataset_dict[subject_name] = reader.read_matlab_v2(length,matdata,task_name,subject_name)
    else:
        dataset_dict[subject_name] = reader.read_matlab_v1(task_name,subject_name,matdata)
    version = args["version"]


    # converted_data = convertor.read_matlab(mat_file)
    # with open(f'{output_dir}/{subject_name}.json', 'w') as f:
    #     json.dump(converted_data, f, indent=4)


    # print(dataset_dict.keys())
    # print(dataset_dict[subject_name][0].keys())
    # print(dataset_dict[subject_name][0]['content'])
    # print(dataset_dict[subject_name][0]['word'][0].keys())
    # print(dataset_dict[subject_name][0]['word'][0]['word_level_EEG']['FFD'])


"""output"""
output_name = f'{task_name}-dataset.pickle'
# with open(os.path.join(output_dir,'task1-SR-dataset.json'), 'w') as out:
#     json.dump(dataset_dict,out,indent = 4)
with open(os.path.join(output_dir,output_name), 'w') as out:
    json.dump(dataset_dict, out, indent=4)

with open(os.path.join(output_dir,output_name), 'wb') as handle:
    pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('write to:', os.path.join(output_dir,output_name))


"""sanity check"""
# check dataset
with open(os.path.join(output_dir,output_name), 'rb') as handle:
    whole_dataset = pickle.load(handle)
print('subjects:', whole_dataset.keys())

if version == 'v1':
    print('num of sent:', len(whole_dataset['ZAB']))
    print()


