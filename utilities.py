from torchlibrosa.stft import STFT, ISTFT, magphase
import torch
import torchaudio.functional as AF
import librosa
import torch.nn.functional as F
#Use the wav2spec functionality they use. Note that their stft is based on librosa
def wav2spec(config, input):
		channels_num = input.shape[-2]


		spec_list = []

		for channel in range(channels_num):
			spec = spectrogram(input[:, channel, :], config)
			spec_list.append(spec)

		spec = torch.cat(spec_list, 1)[:, :, :, :-1]
		return spec

#n_fft is the same as win_length
#hop_size is hop_length
#Why do they insist on changing the names just copy the input parameters
'''
config_spec = {'center' : True, \
				'freeze_parameters' : True, \
				'n_fft' : 256, \
				'hop_length' : 160, \
				'pad_mode' : reflect, \
				'window' : hann, \
				'win_length' : 256}
'''
def spectrogram(input, config):
	n_fft = config['n_fft']
	hop_length = config['hop_length']
	win_length = config['win_length']
	window = config['window']
	center = config['center']
	pad_mode = config['pad_mode']
	freeze_parameters = config['freeze_parameters']
	
	stft_process = STFT(n_fft = n_fft, \
						hop_length = hop_length, \
						win_length = win_length, \
						window = window, \
						center = center, \
						pad_mode = pad_mode, \
						freeze_parameters = freeze_parameters)
	(real, imag) = stft_process(input)
	spec = (real ** 2 + imag ** 2) ** 0.5
	return spec

'''
#n_fft is the same as win_length
config_s2w = {'fps' : 100, \
			'samp_rate' : 16000, \
			'window' : "hann", \
			'n_fft' : 128, \
			'hop_length' : 160, \
			'win_length' : 128, \
			'power' : 1, \
			'normalized' : False, \
			'n_iter' : 200, \
			'momentum' : 0, \
			'rand_init' : False}
'''
def spec2wav(input, config_s2w):

	print(input.shape, "spec2wav input utilities.py")
	#configs for griffinlim
	fps = config_s2w['fps']
	samp_rate = config_s2w['samp_rate']
	n_fft = config_s2w['n_fft']
	hop_length = config_s2w['hop_length']
	win_length = config_s2w['win_length']
	power = config_s2w['power']
	normalized = config_s2w['normalized']
	#length = config_s2w['wav_len'] cant be part of config since it's obtained later
	n_iter = config_s2w['n_iter']
	momentum = config_s2w['momentum']
	rand_init = config_s2w['rand_init']

	window = config_s2w['window']
	fft_window = librosa.filters.get_window(window, win_length, fftbins = True)
	fft_window = librosa.util.pad_center(fft_window, n_fft)
	fft_window = torch.from_numpy(fft_window)
	
	
	#Don't quite follow why they need to pad. I assume this is just so dimensions match
	input = F.pad(input, (0,1), "constant", 0)
	
	
	print(input.shape, "spec2wav input presqueeze utilities.py")
	#fixing the input size -mlreyes
	#Squeeze only the second dimention
	input = torch.squeeze(input, dim = 1)
	
	
	print(input.shape, "spec2wav input utilities.py")
	print(input.transpose(1,2).shape, "did i chnagespec2wav input utilities.py")
	
	wav_len = int((input.shape[-2] - 1)/ fps * samp_rate)
	wav = AF.griffinlim(input.transpose(1,2), \
				window = fft_window, \
				n_fft = n_fft, \
				hop_length = hop_length, \
				win_length = win_length, \
				power = power, \
				#normalized = False, \???? unexpected keyword.
				length = wav_len, \
				n_iter = n_iter, \
				momentum = momentum, \
				rand_init = rand_init)
				
	#corresponding unsqueeze -mlreyes
	wav = wav.unsqueeze(1)
	return wav