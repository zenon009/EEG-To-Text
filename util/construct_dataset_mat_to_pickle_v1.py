import sys

import mat73


sys.path.insert(1,"/home/wrb15144/zenon/EEG-To-Text/")
import scipy.io as io
import h5py
import os
import json
from glob import glob
from tqdm import tqdm
import numpy as np
import pickle
import argparse

from util.Convertor import Convertor
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
args = vars(parser.parse_args())


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
dataset_dict = {}
for mat_file in tqdm(mat_files):
    subject_name = os.path.basename(mat_file).split('_')[0].replace('results','').strip()
    dataset_dict[subject_name] = []
    
    if version == 'v1':
        matdata = io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)['sentenceData']
    elif version == 'v2':
        #matdata = h5py.File(mat_file,'r')
        data_dict = mat73.loadmat(mat_file)
        matdata = data_dict["sentenceData"]
        length = len(matdata["word"])


    # converted_data = convertor.read_matlab(mat_file)
    # with open(f'{output_dir}/{subject_name}.json', 'w') as f:
    #     json.dump(converted_data, f, indent=4)

    for sent_index in range(length):
        word_data = matdata["word"][sent_index]
        if not isinstance(word_data, float):
            # sentence level:
            sent_obj = {'content':matdata["content"][sent_index]}
            sent_obj['sentence_level_EEG'] = {'mean_t1':matdata["mean_t1"][sent_index], 'mean_t2':matdata["mean_t2"][sent_index],
                                              'mean_a1':matdata["mean_a1"][sent_index], 'mean_a2':matdata["mean_a2"][sent_index], 'mean_b1':matdata["mean_b1"][sent_index],
                                              'mean_b2':matdata["mean_b2"][sent_index], 'mean_g1':matdata["mean_g1"][sent_index], 'mean_g2':matdata["mean_g2"][sent_index]}
            sent_obj['sentence_level_EEG'] = convert_dictionary(sent_obj['sentence_level_EEG'])
            if task_name == 'task1-SR':
                sent_obj['answer_EEG'] = {'answer_mean_t1':matdata["answer_mean_t1"][sent_index],
                                          'answer_mean_t2':matdata["answer_mean_t2"][sent_index],
                                          'answer_mean_a1':matdata["answer_mean_a1"][sent_index],
                                          'answer_mean_a2':matdata["answer_mean_a2"][sent_index],
                                          'answer_mean_b1':matdata["answer_mean_b1"][sent_index],
                                          'answer_mean_b2':matdata["answer_mean_b2"][sent_index],
                                          'answer_mean_g1':matdata["answer_mean_g1"][sent_index],
                                          'answer_mean_g2':matdata["answer_mean_g2"][sent_index]}
                sent_obj['answer_EEG'] = convert_dictionary(sent_obj['answer_EEG'])

            # word level:
            sent_obj['word'] = []

            word_tokens_has_fixation = []
            word_tokens_with_mask = []
            word_tokens_all = []
            word_length = len(word_data["content"])
            for word_index in range(word_length):
                word_obj = {'content':word_data["content"][word_index]}
                word_tokens_all.append(word_data["content"][word_index])
                # TODO: add more version of word level eeg: GD, SFD, GPT
                if word_data["nFixations"][word_index] is not None:
                    word_obj['nFixations'] = word_data["nFixations"][word_index].tolist()
                else:
                    word_obj['nFixations'] = []
                if word_data["nFixations"][word_index] is not None:
                    if word_data["nFixations"][word_index].tolist() > 0:
                        word_obj['word_level_EEG'] = {'FFD':{'FFD_t1':word_data["FFD_t1"][word_index], 'FFD_t2':word_data["FFD_t2"][word_index], 'FFD_a1':word_data["FFD_a1"][word_index],
                                                             'FFD_a2':word_data["FFD_a2"][word_index], 'FFD_b1':word_data["FFD_b1"][word_index], 'FFD_b2':word_data["FFD_b2"][word_index], 'FFD_g1':word_data["FFD_g1"][word_index],
                                                             'FFD_g2':word_data["FFD_g2"][word_index]}}
                        word_obj['word_level_EEG']['FFD'] = convert_dictionary(word_obj['word_level_EEG']['FFD'])
                        word_obj['word_level_EEG']['TRT'] = {'TRT_t1':word_data["TRT_t1"][word_index],
                                                             'TRT_t2':word_data["TRT_t2"][word_index],
                                                             'TRT_a1':word_data["TRT_a1"][word_index],
                                                             'TRT_a2':word_data["TRT_a2"][word_index],
                                                             'TRT_b1':word_data["TRT_b1"][word_index],
                                                             'TRT_b2':word_data["TRT_b2"][word_index],
                                                             'TRT_g1':word_data["TRT_g1"][word_index],
                                                             'TRT_g2':word_data["TRT_g2"][word_index]}
                        word_obj['word_level_EEG']['TRT'] = convert_dictionary(word_obj['word_level_EEG']['TRT'])
                        word_obj['word_level_EEG']['GD'] = {'GD_t1':word_data["GD_t1"][word_index],
                                                            'GD_t2':word_data["GD_t2"][word_index],
                                                            'GD_a1':word_data["GD_a1"][word_index],
                                                            'GD_a2':word_data["GD_a2"][word_index],
                                                            'GD_b1':word_data["GD_b1"][word_index],
                                                            'GD_b2':word_data["GD_b2"][word_index],
                                                            'GD_g1':word_data["GD_g1"][word_index],
                                                            'GD_g2':word_data["GD_g2"][word_index]}
                        word_obj['word_level_EEG']['GD'] = convert_dictionary(word_obj['word_level_EEG']['GD'])
                    sent_obj['word'].append(word_obj)
                    word_tokens_has_fixation.append(word_data["content"][word_index])
                    word_tokens_with_mask.append(word_data["content"][word_index])
                else:
                    word_tokens_with_mask.append('[MASK]')
                    # if a word has no fixation, use sentence level feature
                    # word_obj['word_level_EEG'] = {'FFD':{'FFD_t1':matdatamean_t1, 'FFD_t2':matdatamean_t2, 'FFD_a1':matdatamean_a1, 'FFD_a2':matdatamean_a2, 'FFD_b1':matdatamean_b1, 'FFD_b2':matdatamean_b2, 'FFD_g1':matdatamean_g1, 'FFD_g2':matdatamean_g2}}
                    # word_obj['word_level_EEG']['TRT'] = {'TRT_t1':matdatamean_t1, 'TRT_t2':matdatamean_t2, 'TRT_a1':matdatamean_a1, 'TRT_a2':matdatamean_a2, 'TRT_b1':matdatamean_b1, 'TRT_b2':matdatamean_b2, 'TRT_g1':matdatamean_g1, 'TRT_g2':matdatamean_g2}

                    # NOTE:if a word has no fixation, simply skip it
                    continue

            sent_obj['word_tokens_has_fixation'] = word_tokens_has_fixation
            sent_obj['word_tokens_with_mask'] = word_tokens_with_mask
            sent_obj['word_tokens_all'] = word_tokens_all

            dataset_dict[subject_name].append(sent_obj)


        else:
            print(f'missing sent: subj:{subject_name} content:{matdata["content"][sent_index]}, return None')
            dataset_dict[subject_name].append(None)

            continue
    # print(dataset_dict.keys())
    # print(dataset_dict[subject_name][0].keys())
    # print(dataset_dict[subject_name][0]['content'])
    # print(dataset_dict[subject_name][0]['word'][0].keys())
    # print(dataset_dict[subject_name][0]['word'][0]['word_level_EEG']['FFD'])
    with open(mat_file.replace('.mat', '.json'), 'w') as out:
        json.dump(dataset_dict, out, indent=4)

"""output"""
output_name = f'{task_name}-dataset.pickle'
# with open(os.path.join(output_dir,'task1-SR-dataset.json'), 'w') as out:
#     json.dump(dataset_dict,out,indent = 4)

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


