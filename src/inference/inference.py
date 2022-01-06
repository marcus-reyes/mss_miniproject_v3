import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import time
import h5py

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from src.utils.utilities import (read_lst, read_config, mkdir, parse_frameroll2annotation, write_lst)
from src.inference.utilities import (align, onehot_tensor, wav2spec, spec2wav, save_audio, \
																		devide_into_batches, merge_batches, merge_from_list)
from src.dataset.urmp.load_urmp_data import load_test_data
from src.models.modelFactory import ModelFactory
from src.utils.weiMidi import WeiMidi

INSTR_NUM = 2

modes = {"AMT" : ["AMT"],
				"MSS" : ["MSS"],
				"MSS-AMT" : ["MSS-AMT"],
				"MSI" : ["MSI", "MSI-S"],
				"MSI-DIS" : ["MSI-DIS", "MSI-DIS-S"],
				"SYS" : ["SYS"]}

def device(x, dev="cuda"):
	x = torch.from_numpy(x).float()
	if dev == "cuda":
		x = x.cuda()
	return x

def load_gpu_model(conf_path, model_name, model_path):
	nnet = ModelFactory(conf_path, model_name)
	nnet.load_state_dict(torch.load(model_path), strict=True)
	return nnet.cuda()

def get_register(target):
	target = target[target < 88]
	register_seq = ((target + 9) / 12).astype(np.int)
	cnt = np.bincount(register_seq)
	return np.argmax(cnt)

def adapt_pitch(src_target, obj_target):
	adapt_register = (- get_register(src_target) + get_register(obj_target)) * 12
	if adapt_register < 0:
		adapt_register += 12
	elif adapt_register > 0:
		adapt_register -= 12
	print(adapt_register)
	src_target = src_target + adapt_register
	print(len(src_target[src_target > 88]))
	src_target[src_target == 88 + adapt_register] = 88
	src_target[src_target > 88] = 88
	src_target[src_target < 0] = 0
	return src_target

def move2cuda(batch):
	if type(batch) in [tuple, list]:
		batch = list(batch)
		batch_size = 23333
		for i in range(len(batch)):
			batch[i] = batch[i].float().cuda()
			batch_size = batch[i].shape[0] if batch[i].shape[0] < batch_size else batch_size
		for i in range(len(batch)):
			batch[i] = batch[i][:batch_size]
	else:
		batch = batch.float().cuda()
	return batch

class Inference():
	def __init__(self, model_conf_path, data_conf_path, model_name, model_path, output_dir, epoch):
		
		self.model_name = model_name

		self.test_data = load_test_data(data_conf_path)
		model_tag = "MSI-DIS" if model_name == "SYS" else model_name
		self.network = load_gpu_model(model_conf_path, model_tag, model_path)
		
		mkdir(output_dir)
		self.output_dir = output_dir
		self.score_path = os.path.join(output_dir, f"scores-{epoch}.json")


	def query(self, spec_batches, reduce_dim=True):
		hQuery = []
		self.network.eval()
		with torch.no_grad():
			for batch in spec_batches:
				batch = move2cuda(batch)
				hQuery.append(self.network.network.queryNet(batch, "inference"))
		hQuery = torch.cat(hQuery, 0).unsqueeze(1).transpose(1, -1).squeeze(-1).flatten(0, 1)
		if reduce_dim:
			hQuery = hQuery.mean(0)
		return hQuery
			

	def predict(self, batches, condition, mode="inference"):
		preds = []
		condition = condition[None, ...]
		self.network.eval()
		with torch.no_grad():
			for batch in batches:
				batch = move2cuda(batch)
				batch = batch + [condition] if type(batch) is list else [batch, condition]
				preds.append(self.network(batch, mode))
		return preds



	def getHQuery(self):
		output_dir = self.output_dir
		model_name = self.model_name
		results = {}

		for sample in self.test_data.test_samples():
			instrs = sample['instrs']
			for i in range(INSTR_NUM):
				instr_name, _, _, _, query, _ = instrs[i]
				query = device(query[0])
				query_spec, _, _ = wav2spec(query)				
				query_spec_batches = devide_into_batches(query_spec, duration_axis=-2)
				hQuery = self.query(query_spec_batches, False)
				if instr_name not in results:
					results[instr_name] = []
				results[instr_name].append(hQuery.cpu().numpy())
		
		for instr_name in results:
			path = f"{output_dir}/{instr_name}.npy"
			q = np.concatenate(results[instr_name], 0)
			np.save(path, q)


	def synthesis(self, path, track_id):
		output_dir = self.output_dir
		model_name = self.model_name	
		song = WeiMidi(path)
		song = song[track_id]
		song = song[25 * 100 : 35 * 100]
		#target = onehot_tensor(device(target).long())
		sample_index = 0

		for sample in self.test_data.test_samples():
			sample_index += 1
			mix = device(sample['mix'].sum(0))
			instrs = sample['instrs']
			sample_dir = f"{output_dir}/sample_{sample_index}"
			mkdir(sample_dir)
			print(mix.shape[-1] - 16000 * 45)
			mix = mix[:, 16000 * 35 : 16000 * 45]


			for i in range(INSTR_NUM):
				instr_name, audio, _, target, query, _ = instrs[i]
				query = device(query[0])
				
				target = adapt_pitch(song, target)
				target = onehot_tensor(device(target).long())


				mix_spec, mix_cos, mix_sin = wav2spec(mix)
				query_spec, _, _ = wav2spec(query)

				wav_len = mix.shape[-1]
				spec_len = mix_spec.shape[-2]

				if spec_len > target.shape[-1]:
					spec_len = target.shape[-1]
					mix_spec = mix_spec.transpose(0, -2)[:spec_len].transpose(0, -2)

				mix_spec_batches = devide_into_batches(mix_spec, duration_axis=-2)
				query_spec_batches = devide_into_batches(query_spec, duration_axis=-2)

				hQuery = self.query(query_spec_batches)

				target_batches = devide_into_batches(target, duration_axis=-1)
				batches = zip(mix_spec_batches, target_batches)

				preds = self.predict(batches, hQuery, "synthesis")

				synthesis_batches = merge_from_list(preds, index=0)
				synthesis_spec = merge_batches(synthesis_batches, duration_axis=-2)
				synthesis_spec = align(synthesis_spec, mix_cos, -2)
				synthesis_wav = spec2wav(synthesis_spec, mix_cos, mix_sin, 160000, syn_phase=1)

				synthesis_wav_path = f"{sample_dir}/{instr_name}_{model_name}.wav"
				save_audio(synthesis_wav[:, :160000], synthesis_wav_path)




	def inference(self):
		
		results = {}
		sample_index = -1
		output_dir = self.output_dir

		model_name = self.model_name
		for mode in modes[model_name]:
			results[mode] = []

		for sample in self.test_data.test_samples():
			sample_index += 1
			mix = device(sample['mix'].sum(0))
			instrs = sample['instrs']
			
			result = {}
			for mode in modes[model_name]:
				result[mode] = {}

			for i in range(INSTR_NUM):
				instr_name, audio, annotation, target, query, query_annotation = instrs[i]
			
				for mode in modes[model_name]:
					result[mode][instr_name] = {"separation" : [], "transcription" : []}
	
				query = device(query[0])
				query_annotation = query_annotation[0]

				mix_spec, mix_cos, mix_sin = wav2spec(mix)
				query_spec, _, _ = wav2spec(query)

				
				wav_len = mix.shape[-1]
				spec_len = mix_spec.shape[-2]
				
				mix_spec_batches = devide_into_batches(mix_spec, duration_axis=-2)
				query_spec_batches = devide_into_batches(query_spec, duration_axis=-2)

				hQuery = self.query(query_spec_batches)
				
				if model_name in ["AMT", "MSS", "MSS-AMT", "UNET"]:
					batches = mix_spec_batches
				else:
					target = onehot_tensor(device(target).long())
					target_batches = devide_into_batches(target, duration_axis=-1)
					batches = zip(mix_spec_batches, target_batches)
				
				preds = self.predict(batches, hQuery)
				
				sample_dir = f"{output_dir}/sample_{sample_index}"
				mkdir(sample_dir)

	
				if model_name in ["AMT", "MSS-AMT", "MSI", "MSI-DIS"]:
					transcription_batches = merge_from_list(preds, index=-1)					
					prob = merge_batches(transcription_batches, duration_axis=-1)
					est_annotation = parse_frameroll2annotation(np.argmax(prob.cpu().numpy(), 0))
					est_annotation_path = f"{sample_dir}/{instr_name}_est.txt" 
					write_lst(est_annotation_path, est_annotation)
					
					annotation = str.replace(annotation, '/gpfsnyu/home/ll4270/music/transcription/wei_transcription', '/scratch/gx219/wei_env/data-source/dataset')
					result[model_name][instr_name]['transcription'].append([est_annotation_path, annotation])


				if not model_name == "AMT":
					separated_batches = merge_from_list(preds, index=0)
					separated_spec = merge_batches(separated_batches, duration_axis=-2)
					separated_spec = align(separated_spec, mix_cos, -2)
					separated_wav = spec2wav(separated_spec, mix_cos, mix_sin, wav_len)

					separated_wav_path = f"{sample_dir}/{instr_name}_{model_name}.wav"
					mix_path = f"{sample_dir}/Mixture.wav"
					ref_path = f"{sample_dir}/{instr_name}_ref.wav"
					save_audio(separated_wav, separated_wav_path)
					save_audio(mix, mix_path)
					save_audio(torch.from_numpy(audio), ref_path)
					result[model_name][instr_name]['separation'].append([separated_wav_path, ref_path])

				if model_name in ["MSI", "MSI-DIS"]:
					separated_batches = merge_from_list(preds, index=1)
					separated_spec = merge_batches(separated_batches, duration_axis=-2)
					separated_spec = align(separated_spec, mix_cos, -2)
					separated_wav = spec2wav(separated_spec, mix_cos, mix_sin, wav_len)
					separated_wav_path = f"{sample_dir}/{instr_name}_{model_name}-S.wav"
					save_audio(separated_wav, separated_wav_path)
					result[f"{model_name}-S"][instr_name]['separation'].append([separated_wav_path, ref_path])


			for mode in modes[model_name]:
				results[mode].append(result[mode])

		return results
	
		

