import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import random
import h5py
from prefetch_generator import BackgroundGenerator

sys.path.insert(1, os.path.join(sys.path[0], '../../..'))
from src.utils.utilities import (read_lst, read_config, int16_to_float32, encode_mu_law)
from src.utils.dataset_sample import BaseDataset
from src.utils.audio_utilities import write_audio

random_state = np.random.RandomState(1234)
SHIFT = 9

class UrmpDataset(BaseDataset):
	def __init__(self, name, config_path, config_name, mode, shuffle=False):
		super(UrmpDataset, self).__init__(config_path, config_name, mode, shuffle)
		self.config_name = config_name
		self.__init_urmp_params__(config_path, config_name, name)

	def __init_urmp_params__(self, config_path, config_name, name):
		hparams = read_config(config_path, config_name)
		params = self._params
		params['max_note_shift'] = int(hparams['max_note_shift'])
		params['frames_per_second'] = float(hparams['frames_per_second'])
		params['begin_note'] = int(hparams['begin_note'])
		params['classes_num'] = int(hparams['classes_num'])
		params['config_name'] = config_name
		self.tag = -1

	def get_frames_per_second(self):
		return self._params['frames_per_second']

	def get_classes_num(self):
		return self._params['classes_num']

	def get_duration(self):
		return self._params['duration']

	def get_samples_num(self):
		return len(self._params['file_lst'])


	def next_sample(self, pos=None, is_query=False):
		params = self._params
		duration = params['duration']
		mode = params['mode']
		classes_num = params['classes_num']
		frames_per_second = params['frames_per_second']
		config_name = params['config_name']


		def is_silence(x):
			return x.shape[-1] * 88 == x.sum()

		def frame_roll_mask(x, y):
			mask = np.ones_like(x)
			mask[x == 88] = 0
			mask[y == 88] = 1
			return mask

		def load_file(pos, track_id, shift_pitch):
			if self._data[track_id] is None:
				hdf5_path = params['file_lst'][track_id]
				datas = []
				for i in range(9):
					data = {}
					train_hdf5_path = str.replace(hdf5_path, '.h5', f'._TRAIN_shift_pitch_{i - 4}.h5')
					hf = h5py.File(train_hdf5_path, 'r')
					data = {'shift_waveform': int16_to_float32(hf['shift_waveform'][:])[None, :],
						'shift_dense_waveform' : int16_to_float32(hf['shift_dense_waveform'][:])[None, :],
						'frame_roll': hf['frame_roll'][:].astype(np.int)}
					datas.append(data)
				self._data[track_id] = datas
			return self._data[track_id][shift_pitch + 4]

		def load_cache_data(pos, track_id, other_nid, another_nid, is_query):
			hdf5_path = params['file_lst'][track_id]

			max_note_shift = params['max_note_shift']
			duration = params['duration']
			sample_rate = params['sample_rate']

			if is_query:
				shift_pitch = random_state.randint(0, SHIFT) - SHIFT // 2
				hf = load_file(pos, other_nid, shift_pitch)
				shift_dense_waveform = hf['shift_dense_waveform']
				st = random_state.randint(0, shift_dense_waveform.shape[1] - duration)
				query_waveform = shift_dense_waveform[:, st : st + duration].copy()

				shift_pitch = random_state.randint(0, SHIFT) - SHIFT // 2
				hf = load_file(pos, another_nid, shift_pitch)
				shift_dense_waveform = hf['shift_dense_waveform']
				st = random_state.randint(0, shift_dense_waveform.shape[1] - duration)
				another_query_waveform = shift_dense_waveform[:, st : st + duration].copy()

				return query_waveform, another_query_waveform

			else:

				shift_pitch = random_state.randint(0, SHIFT) - SHIFT // 2
				hf = load_file(pos, track_id, shift_pitch)
				waveform = hf['shift_waveform']
				frame_roll = hf['frame_roll']

				shift_pitch = random_state.randint(0, SHIFT) - SHIFT // 2
				hf = load_file(pos, track_id, shift_pitch)
				strong_waveform = hf['shift_waveform']
				another_frame_roll = hf['frame_roll']

				start_time = random_state.randint(0, int((waveform.shape[-1] - duration) / sample_rate))
				st = start_time * sample_rate
				frame_roll_st = int(start_time * frames_per_second)
				ed = frame_roll_st + int(duration // sample_rate * frames_per_second) + 1
				obj_frame_roll = frame_roll[frame_roll_st : ed].copy()
					
				another_start_time = random_state.randint(0, int((waveform.shape[-1] - duration) / sample_rate)) if is_silence(obj_frame_roll) else start_time
				another_st = another_start_time * sample_rate
				another_frame_roll_st = int(another_start_time * frames_per_second)
				another_ed = another_frame_roll_st + int(duration // sample_rate * frames_per_second) + 1
				another_frame_roll = another_frame_roll[another_frame_roll_st : another_ed].copy()

				ori_waveform = waveform[:, st : st + duration].copy()
				strong_waveform = strong_waveform[:, another_st : another_st + duration].copy()
		
				return (ori_waveform, strong_waveform, obj_frame_roll, another_frame_roll)

		def get_next_track(pos=None, is_query=False):
			nid = self.__get_next_track_id__(pos)
			other_nid = self.__get_next_track_id__(pos + 1)
			another_nid = self.__get_next_track_id__(pos + 2)
			return load_cache_data(pos, nid, other_nid, another_nid, is_query)

		tracks = get_next_track(pos, is_query)
		return tracks



class UrmpSample(Dataset):
	def __init__(self, config_path, mode):
		super(UrmpSample, self).__init__()
		self.__init_params__(config_path, mode)

	def __iter__(self):
		return BackgroundGenerator(super().__iter__())
		#return super().__iter__()

	def __init_params__(self, config_path, mode):
		hparams = read_config(config_path, 'hdf5s_data')
		instruments = hparams['instruments'].split(',')
		
		datasets = {}
		for instr in instruments:
			datasets[instr] = UrmpDataset(instr, config_path, instr, mode,  mode=='train')
			notes_num = datasets[instr].get_classes_num()
			duration = datasets[instr].get_duration()
			frames_per_second = datasets[instr].get_frames_per_second()
			sample_rate = datasets[instr].get_sample_rate()

		self._datasets = datasets
		datasets_index = []
		datasets_samples_num = [0]
		for d in datasets:
			datasets_index.append(d)
			n = datasets[d].get_samples_num()
			datasets_samples_num.append(n + datasets_samples_num[-1])


		self._datasets_index = datasets_index
		self.datasets_samples_num = datasets_samples_num
		classes_num = len(datasets_index)

		params = {}
		params['batch_size'] = int(hparams[f'{mode}_batch_size'])
		params['mode'] = mode
		params['notes_num'] = notes_num
		params['classes_num'] = classes_num
		params['duration'] = duration
		params['frames_per_second'] = frames_per_second
		params['sources_num'] = len(instruments)
		params['instruments'] = instruments
		params['sample_rate'] = sample_rate
		params['len'] = int(hparams['samples_num'])
		self._params = params
		
	def __get_train_sample__(self, index, instr_indexs, is_query):
		classes_num = self._params['classes_num']
		input_samples = []
		datasets = self._datasets
		datasets_index = self._datasets_index

		for instr in instr_indexs:
			dataset = datasets[datasets_index[instr]]
			inputs = dataset.next_sample(index, is_query)
			#print(len(inputs), "inputs shape urmpsample.py.")
			for i, input in enumerate(inputs):
				if len(input_samples) == i:
					input_samples.append([])
				input = np.expand_dims(input, 0)
				input_samples[i].append(input)

		for i, input in enumerate(input_samples):
			input_samples[i] = np.concatenate(input_samples[i], 0)

		#print(input_samples[0].shape, "input samples shape aka one training sample")
		#print(len(input_samples), "input samples len")
		return input_samples


	def __extract__(self, separated, query_separated, another_query_separated, target, r):
		separated = torch.from_numpy(separated[r]).float()
		query = torch.from_numpy(query_separated[r]).float()
		another_query = torch.from_numpy(another_query_separated[r]).float()
		target = torch.from_numpy(target[r]).long()
		return separated, query, another_query, target


	def __sample_class__(self):
		class_ratio = self.datasets_samples_num
		index = random_state.randint(class_ratio[-1])
		for i in range(len(class_ratio) - 1):
			if index < class_ratio[i + 1]:
				return i

	def __getitem__(self, index = 0):
		mode = self._params['mode']
		if mode == "train":
			classes_num = self.get_classes_num()
			UB = 2
			up_bound = classes_num if classes_num < UB else UB
			#mixtures_num = random.randint(low_bound, up_bound) 
			#up_bound = classes_num
			mixtures_num = 2
			selected_ids = []
			while len(selected_ids) < up_bound:
				id = self.__sample_class__()
				if not id in selected_ids:
					selected_ids.append(id)

			(separated, strong_separated, target, another_target) = self.__get_train_sample__(index, selected_ids[:mixtures_num], is_query=False)
			(query_separated, another_query_separated) = self.__get_train_sample__(index, selected_ids, is_query=True)
			mix = torch.from_numpy(separated).float().sum(0)
			strong_mix = torch.from_numpy(strong_separated).float().sum(0)
			separated = torch.from_numpy(separated).float()
			query_separated = torch.from_numpy(query_separated).float()
			another_query_separated = torch.from_numpy(another_query_separated).float()
			target = torch.from_numpy(target).long()
			another_target = torch.from_numpy(another_target).long()
			batch = (separated, query_separated, another_query_separated, target, another_target)
			return mix, strong_mix, batch
		else:
			return self.__get_vali_sample__(index)
		assert False

	def __len__(self):
		return self._params['len']

	def get_len(self):
		return self.__len__()

	def get_batch_size(self):
		return self._params['batch_size']

	def get_collate_fn(self):
		return default_collate

	def get_duration(self):
		return self._params['duration']	

	def get_frames_per_second(self):
		return self._params['frames_per_second']

	def get_classes_num(self):
		return self._params['classes_num']

	def get_sources_num(self):
		return self._params['sources_num']

	def get_instruments(self):
		return self._params['instruments']

	def get_sample_rate(self):
		return self._params['sample_rate']

