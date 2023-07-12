import os

import h5py
import numpy as np
from scipy import io
from tqdm import tqdm


class Reader:
    def __init__(self):
        print("Reader object created")

    def check_if_numpy_array(self,data):
        if isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data

    def convert_dictionary(self,dict):
        for key in dict.keys():
            dict[key] = self.check_if_numpy_array(dict[key])
        return dict
    def read_matlab_v1(self,  task_name,subject_name, matdata):
        dataset_dict = []

        for sent in matdata:
            word_data = sent.word

            # sentence level:
            sent_obj = {'content': sent.content}
            sent_obj['sentence_level_EEG'] = {'mean_t1': sent.mean_t1, 'mean_t2': sent.mean_t2,
                                              'mean_a1': sent.mean_a1, 'mean_a2': sent.mean_a2,
                                              'mean_b1': sent.mean_b1, 'mean_b2': sent.mean_b2,
                                              'mean_g1': sent.mean_g1, 'mean_g2': sent.mean_g2}
            sent_obj["sentence_level_EEG"] = self.convert_dictionary(sent_obj["sentence_level_EEG"])

            if task_name == 'task1-SR':
                sent_obj['answer_EEG'] = {'answer_mean_t1': sent.answer_mean_t1,
                                          'answer_mean_t2': sent.answer_mean_t2,
                                          'answer_mean_a1': sent.answer_mean_a1,
                                          'answer_mean_a2': sent.answer_mean_a2,
                                          'answer_mean_b1': sent.answer_mean_b1,
                                          'answer_mean_b2': sent.answer_mean_b2,
                                          'answer_mean_g1': sent.answer_mean_g1,
                                          'answer_mean_g2': sent.answer_mean_g2}
                sent_obj["answer_EEG"] = self.convert_dictionary(sent_obj["answer_EEG"])

            # word level:
            sent_obj['word'] = []

            word_tokens_has_fixation = []
            word_tokens_with_mask = []
            word_tokens_all = []
            if type(word_data) is not float:
                for word in word_data:
                    word_obj = {'content': word.content}
                    word_tokens_all.append(word.content)
                    # TODO: add more version of word level eeg: GD, SFD, GPT
                    word_obj['nFixations'] = word.nFixations
                    if word.nFixations > 0:
                        word_obj['word_level_EEG'] = {
                            'FFD': {'FFD_t1': word.FFD_t1, 'FFD_t2': word.FFD_t2, 'FFD_a1': word.FFD_a1,
                                    'FFD_a2': word.FFD_a2, 'FFD_b1': word.FFD_b1, 'FFD_b2': word.FFD_b2,
                                    'FFD_g1': word.FFD_g1, 'FFD_g2': word.FFD_g2}}
                        word_obj['word_level_EEG']['FFD'] = self.convert_dictionary(word_obj['word_level_EEG']['FFD'])
                        word_obj['word_level_EEG']['TRT'] = {'TRT_t1': word.TRT_t1, 'TRT_t2': word.TRT_t2,
                                                             'TRT_a1': word.TRT_a1, 'TRT_a2': word.TRT_a2,
                                                             'TRT_b1': word.TRT_b1, 'TRT_b2': word.TRT_b2,
                                                             'TRT_g1': word.TRT_g1, 'TRT_g2': word.TRT_g2}
                        word_obj['word_level_EEG']['TRT'] = self.convert_dictionary(word_obj['word_level_EEG']['TRT'])
                        word_obj['word_level_EEG']['GD'] = {'GD_t1': word.GD_t1, 'GD_t2': word.GD_t2,
                                                            'GD_a1': word.GD_a1, 'GD_a2': word.GD_a2,
                                                            'GD_b1': word.GD_b1, 'GD_b2': word.GD_b2,
                                                            'GD_g1': word.GD_g1, 'GD_g2': word.GD_g2}
                        word_obj['word_level_EEG']['GD'] = self.convert_dictionary(word_obj['word_level_EEG']['GD'])
                        sent_obj['word'].append(word_obj)
                        word_tokens_has_fixation.append(word.content)
                        word_tokens_with_mask.append(word.content)
                    else:
                        word_tokens_with_mask.append('[MASK]')
                        # if a word has no fixation, use sentence level feature
                        # word_obj['word_level_EEG'] = {'FFD':{'FFD_t1':sent.mean_t1, 'FFD_t2':sent.mean_t2, 'FFD_a1':sent.mean_a1, 'FFD_a2':sent.mean_a2, 'FFD_b1':sent.mean_b1, 'FFD_b2':sent.mean_b2, 'FFD_g1':sent.mean_g1, 'FFD_g2':sent.mean_g2}}
                        # word_obj['word_level_EEG']['TRT'] = {'TRT_t1':sent.mean_t1, 'TRT_t2':sent.mean_t2, 'TRT_a1':sent.mean_a1, 'TRT_a2':sent.mean_a2, 'TRT_b1':sent.mean_b1, 'TRT_b2':sent.mean_b2, 'TRT_g1':sent.mean_g1, 'TRT_g2':sent.mean_g2}

                        # NOTE:if a word has no fixation, simply skip it
                        continue

                sent_obj['word_tokens_has_fixation'] = word_tokens_has_fixation
                sent_obj['word_tokens_with_mask'] = word_tokens_with_mask
                sent_obj['word_tokens_all'] = word_tokens_all

                dataset_dict.append(sent_obj)
            else:
                dataset_dict.append(sent_obj)


        return dataset_dict

    def read_matlab_v2(self,length,matdata,task_name,subject_name):
        dataset_dict = []
        for sent_index in range(length):
            word_data = matdata["word"][sent_index]
            if not isinstance(word_data, float):
                # sentence level:
                sent_obj = {'content': matdata["content"][sent_index]}
                sent_obj['sentence_level_EEG'] = {'mean_t1': matdata["mean_t1"][sent_index],
                                                  'mean_t2': matdata["mean_t2"][sent_index],
                                                  'mean_a1': matdata["mean_a1"][sent_index],
                                                  'mean_a2': matdata["mean_a2"][sent_index],
                                                  'mean_b1': matdata["mean_b1"][sent_index],
                                                  'mean_b2': matdata["mean_b2"][sent_index],
                                                  'mean_g1': matdata["mean_g1"][sent_index],
                                                  'mean_g2': matdata["mean_g2"][sent_index]}
                sent_obj['sentence_level_EEG'] = self.convert_dictionary(sent_obj['sentence_level_EEG'])
                if task_name == 'task1-SR':
                    sent_obj['answer_EEG'] = {'answer_mean_t1': matdata["answer_mean_t1"][sent_index],
                                              'answer_mean_t2': matdata["answer_mean_t2"][sent_index],
                                              'answer_mean_a1': matdata["answer_mean_a1"][sent_index],
                                              'answer_mean_a2': matdata["answer_mean_a2"][sent_index],
                                              'answer_mean_b1': matdata["answer_mean_b1"][sent_index],
                                              'answer_mean_b2': matdata["answer_mean_b2"][sent_index],
                                              'answer_mean_g1': matdata["answer_mean_g1"][sent_index],
                                              'answer_mean_g2': matdata["answer_mean_g2"][sent_index]}
                    sent_obj['answer_EEG'] = self.convert_dictionary(sent_obj['answer_EEG'])

                # word level:
                sent_obj['word'] = []

                word_tokens_has_fixation = []
                word_tokens_with_mask = []
                word_tokens_all = []
                word_length = len(word_data["content"])
                for word_index in range(word_length):
                    word_obj = {'content': word_data["content"][word_index]}
                    word_tokens_all.append(word_data["content"][word_index])
                    # TODO: add more version of word level eeg: GD, SFD, GPT
                    if word_data["nFixations"][word_index] is not None:
                        word_obj['nFixations'] = word_data["nFixations"][word_index].tolist()
                    else:
                        word_obj['nFixations'] = []
                    if word_data["nFixations"][word_index] is not None:
                        if word_data["nFixations"][word_index].tolist() > 0:
                            word_obj['word_level_EEG'] = {'FFD': {'FFD_t1': word_data["FFD_t1"][word_index],
                                                                  'FFD_t2': word_data["FFD_t2"][word_index],
                                                                  'FFD_a1': word_data["FFD_a1"][word_index],
                                                                  'FFD_a2': word_data["FFD_a2"][word_index],
                                                                  'FFD_b1': word_data["FFD_b1"][word_index],
                                                                  'FFD_b2': word_data["FFD_b2"][word_index],
                                                                  'FFD_g1': word_data["FFD_g1"][word_index],
                                                                  'FFD_g2': word_data["FFD_g2"][word_index]}}
                            word_obj['word_level_EEG']['FFD'] = self.convert_dictionary(word_obj['word_level_EEG']['FFD'])
                            word_obj['word_level_EEG']['TRT'] = {'TRT_t1': word_data["TRT_t1"][word_index],
                                                                 'TRT_t2': word_data["TRT_t2"][word_index],
                                                                 'TRT_a1': word_data["TRT_a1"][word_index],
                                                                 'TRT_a2': word_data["TRT_a2"][word_index],
                                                                 'TRT_b1': word_data["TRT_b1"][word_index],
                                                                 'TRT_b2': word_data["TRT_b2"][word_index],
                                                                 'TRT_g1': word_data["TRT_g1"][word_index],
                                                                 'TRT_g2': word_data["TRT_g2"][word_index]}
                            word_obj['word_level_EEG']['TRT'] = self.convert_dictionary(word_obj['word_level_EEG']['TRT'])
                            word_obj['word_level_EEG']['GD'] = {'GD_t1': word_data["GD_t1"][word_index],
                                                                'GD_t2': word_data["GD_t2"][word_index],
                                                                'GD_a1': word_data["GD_a1"][word_index],
                                                                'GD_a2': word_data["GD_a2"][word_index],
                                                                'GD_b1': word_data["GD_b1"][word_index],
                                                                'GD_b2': word_data["GD_b2"][word_index],
                                                                'GD_g1': word_data["GD_g1"][word_index],
                                                                'GD_g2': word_data["GD_g2"][word_index]}
                            word_obj['word_level_EEG']['GD'] = self.convert_dictionary(word_obj['word_level_EEG']['GD'])
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

                dataset_dict.append(sent_obj)


            else:
                print(f'missing sent: subj:{subject_name} content:{matdata["content"][sent_index]}, return None')
                dataset_dict.append([])

        return dataset_dict