import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import time
import h5py
from torchlibrosa.stft import STFT, ISTFT, magphase

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from src.utils.utilities import (read_lst, read_config)
from src.models.models import (AMTBaseline, MSSBaseline, MultiTaskBaseline, DisentanglementModel)

et = 1e-8


class ModelFactory(nn.Module):

	def __init__(self, conf_path, model_name):
		super(ModelFactory, self).__init__()

		hparams = read_config(conf_path, "FeatureExaction")
		window_size=int(hparams['window_size'])
		hop_size = int(hparams['hop_size'])
		pad_mode = hparams['pad_mode']
		window = hparams['window']

		self.stft = STFT(n_fft=window_size, hop_length=hop_size,
			win_length=window_size, window=window, center=True,
			pad_mode=pad_mode, freeze_parameters=True)

		self.istft = ISTFT(n_fft=window_size, hop_length=hop_size,
			win_length=window_size, window=window, center=True,
			pad_mode=pad_mode, freeze_parameters=True)


		if model_name in ['AMT', 'AMTBaseline']:
			network = AMTBaseline(conf_path)
		elif model_name in ['MSS', 'MSSBaseline']:
			network = MSSBaseline(conf_path)
		elif model_name in ['MSS-AMT', 'MultiTaskBaseline']:
			network = MultiTaskBaseline(conf_path)
		elif model_name in ['MSI', 'MSI-DIS', 'DisentanglementModel']:
			network = DisentanglementModel(conf_path)
	
		self.network = network

	def wav2spec(self, input):
		channels_num = input.shape[-2]

		def spectrogram(input):
			(real, imag) = self.stft(input)
			spec = (real ** 2 + imag ** 2) ** 0.5
			return spec

		spec_list = []

		for channel in range(channels_num):
			spec = spectrogram(input[:, channel, :])
			spec_list.append(spec)

		spec = torch.cat(spec_list, 1)[:, :, :, :-1]
		return spec

	def forward(self, input, mode):
		if mode == "wav2spec":
			spec = self.wav2spec(input)
			return spec
		return self.network(input, mode)
		

if __name__ == '__main__':
	model_name = 'MSI-DIS'
	conf_path = 'conf/model.cfg'
	model = ModelFactory(conf_path, model_name)
	model.cuda()
	query_spec = model(torch.zeros(1, 1, 16000 * 3).cuda(), 'wav2spec')
	mix_spec = model(torch.zeros(1, 1, 16000 * 3).cuda(), 'wav2spec')
	print(query_spec.shape, mix_spec.shape)
	out = model(query_spec, 'query')
	print(out.shape)
	args = (mix_spec, mix_spec, out)
	sep, prob, target = model(args, 'transfer')
	print(sep.shape, prob.shape, target.shape)

	
