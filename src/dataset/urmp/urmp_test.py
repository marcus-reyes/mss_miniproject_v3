import numpy as np
import librosa
import sys
import os
import h5py

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from src.utils.utilities import (parse_frameroll2annotation, read_lst, read_config, write_lst, mkdir, int16_to_float32)

class UrmpTest(object):
	def __init__(self, data_lst, query_lst, sample_rate=16000):
		file_lst = read_lst(data_lst)
		query_lst = read_lst(query_lst)
		data_path = []
		for i, f in enumerate(file_lst):
			fs = f.split('\t')
			qs = query_lst[i].split('\t')
			instruments = fs[0].split(',')
			files = fs[1].split(',')
			query = qs[1].split(',')
			
			sample_name = str.replace(files[0].split('_')[-1], '.h5' ,'')
			#need ._TEST.h5 instead of just .h5. didn't work
			#sample_name = str.replace(files[0].split('_')[-1], '._TEST.h5' ,'')
			sample = {}
			for j, instr in enumerate(instruments):
				sample[instr] = {}
				sample[instr]['ref'] = files[j]
				sample[instr]['query'] = query[j]
			
			data_path.append({'sample_name' : sample_name, 'instrs' : sample})

		self.data_path = data_path
		self.sample_rate = sample_rate


	def vad(self, x, frame_roll, frames_per_second=100, sample_rate=16000, notes_num=88):
 
		frames_per_sample = frames_per_second * 1. / sample_rate

		if len(x.shape) == 2:
			y = x[0]
		else:
			y = x
		output = np.zeros_like(y)
		frame_roll_len = int(y.shape[-1] / sample_rate * frames_per_second + 1)
		frame_roll = frame_roll[ : frame_roll_len]
		new_frame_roll = np.zeros_like(frame_roll) + notes_num
		split_index = librosa.effects.split(y, top_db=18)
		st = 0
		ed = 0
		for index in split_index:
			ed = st + index[1] - index[0]
			output[st : ed] = y[index[0] : index[1]]
			offset = (index[1] - index[0]) * frames_per_sample
			ori_st = int(st * frames_per_sample)
			ori_ed = int(st * frames_per_sample + offset)
			obj_st = int(index[0] * frames_per_sample)
			obj_ed = int(index[0] * frames_per_sample + offset)
			offset = (ori_ed - ori_st) if ori_ed - ori_st < ed - st else obj_ed - obj_st
			new_frame_roll[ori_st : ori_st + offset] = frame_roll[obj_st : obj_st + offset]
			st = ed
		output = output[:ed]
		new_frame_roll = new_frame_roll[: int(ed * frames_per_sample)]

		if len(x.shape) == 2:
			output = output[None, :]
		return output, new_frame_roll


	def test_samples(self):
		for data in self.data_path:
			sample_name = data['sample_name']
			sample = data['instrs']
			samples = []
			mix = []
			for instr in sample:
				ref = sample[instr]['ref']
				queries = sample[instr]['query'].split(' ')
				query = []
				tr_query = []
				for q in queries:
				
					#add to change since they swapped conventison
					print(q, "q in urmp_test.py")
					q = q.replace(".h5", "._TEST.h5")
					print(q, "q post replacein urmp_test.py")
					with h5py.File(q, 'r') as hf:
						waveform = int16_to_float32(hf['waveform'][:])[None, :]
						frame_roll = hf['frame_roll'][:].astype(np.int)

					waveform, frame_roll = self.vad(waveform, frame_roll)
					query.append(waveform)
					tr_query.append(parse_frameroll2annotation(frame_roll))
				
				#add to change since they swapped conventison
				ref = ref.replace(".h5", "._TEST.h5")
				with h5py.File(ref, 'r') as hf:
					wav_ref = int16_to_float32(hf['waveform'][:])[None, :]
					tr_ref = hf['note_annotations_txt'][0].decode()
					frame_roll = hf['frame_roll'][:].astype(np.int)

				samples.append([instr, wav_ref, tr_ref, frame_roll, query, tr_query])

			ref_len = samples[0][1].shape[-1]
			for i, ref in enumerate(samples):
				ref_len = ref[1].shape[-1] if ref_len > ref[1].shape[-1] else ref_len
				
			samples = [[s[0], s[1][:, :ref_len]] + s[2:] for s in samples]
			mix = [s[1] for s in samples]
			mix = np.stack(mix, 0)
			test_sample = {'mix' : mix, 'sample_name': sample_name, 'instrs' : samples}
			yield test_sample
		
