import os
import sys
import configparser
import numpy as np
import time
import h5py
import random

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from src.utils.utilities import (read_lst, read_config, int16_to_float32)
from src.utils.target_process import TargetProcessor

random_state = np.random.RandomState(1234)

class BaseDataset(object):
	def __init__(self, config_path, config_name, mode, shuffle):
		hparams = read_config(config_path, config_name)
		self._params = self.__init_params__(hparams, mode, shuffle)
		self._data = self.__init_data__()
		self._tracks_id = self.__init_tracks_id__()

	def __init_params__(self, hparams, mode, shuffle):
		sample_rate = int(hparams['sample_rate'])
		file_lst = read_lst(hparams[f'{mode}_lst'])
		duration = int(hparams[f'{mode}_duration']) * sample_rate
		audios_num = len(file_lst)

		params = {}
		params['file_lst'] = file_lst
		params['shuffle'] = shuffle
		params['audios_num'] = audios_num
		params['mode'] = mode
		params['duration'] = duration
		params['sample_rate'] = sample_rate
		return params

	def __init_data__(self):
		params = self._params
		audios_num = params['audios_num']
		return [None] * audios_num

	def __init_tracks_id__(self):
		params = self._params
		audios_num = params['audios_num']
		return np.arange(audios_num)

	def get_audios_num(self):
		return self._params['audios_num']

	def __get_next_track_id__(self, pos = None):
		params = self._params
		audios_num = params['audios_num']

		if pos is None:
			current_id = self.current_id
			current_id = current_id + 1
			if current_id == split_audios_num:
				if params['shuffle']:
					random_state.shuffle(self._tracks_id)
				current_id = 0
			self.current_id  = current_id
		else:
			current_id = pos % audios_num
			if params['shuffle']:
				random_state.shuffle(self._tracks_id)

		nid = self._tracks_id[current_id]
		return nid

	def get_next_train_sample(self, pos = None):
		pass

	def get_sample_rate(self):
		return self._params['sample_rate']


