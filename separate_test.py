from src.dataset.urmp.load_urmp_data import load_test_data

from src.utils.multiEpochsDataLoader import MultiEpochsDataLoader as DataLoader

import torchaudio
import torch

from reyes_tan_models_upgraded import *
testdata = load_test_data(r"/home/jeff/Downloads/stock_data/Dataset/reproduce_data/hdf5s/urmp-rec")

print(testdata.test_samples())
config_enc = {'num_blocks' : 3, \
			'in_channels' : 1, \
			'momentum' : 0.01, \
			'cond_dim' : 6, \
			'input_size' : 256, \
			}
			
config_dec = {'num_blocks' : 3, \
		'in_channels' : 16, \
		'momentum' : 0.01}
		

config_query = {'num_blocks' : 2, \
'in_channels' : 1, \
'input_size' : 256, \
'cond_dim' : 6, \
}
#n_fft is the same as win_length
config_spec = {'center' : True, \
		'freeze_parameters' : True, \
		'n_fft' : 512, \
		'hop_length' : 160, \
		'pad_mode' : "reflect", \
		'window' : "hann", \
		'win_length' : 512}
		
		
# should be 256 not 128
#n_fft is the same as win_length
config_s2w = {'fps' : 100, \
		'samp_rate' : 16000, \
		'window' : "hann", \
		'n_fft' : 512, \
		'hop_length' : 160, \
		'win_length' : 512, \
		'power' : 1, \
		'normalized' : False, \
		'n_iter' : 200, \
		'momentum' : 0, \
		'rand_init' : False}

enc_net = Encoder(config_enc)
dec_net = Decoder(config_dec)
query_net = Query(config_query)


config_tra = {'num_blocks' : 2, \
		'output_dim' : 89, \
		'in_channels' : enc_net.latent_rep_channels[-1][0], \
		'input_size' : enc_net.latent_rep_channels[-1][1], \
		'momentum' : 0.01\
		}

tra_net = Transcriptor(config_tra)
pitch_net = Pitch(tra_net.notes_num, enc_net.latent_rep_channels)
tim_net = Timbre(enc_net.latent_rep_channels, 0.01)

checkpoint = torch.load('checkpoint_21.pt')


print("Initialized models")

enc_net.load_state_dict(checkpoint['enc_state_dict'])
dec_net.load_state_dict(checkpoint['dec_state_dict'])
query_net.load_state_dict(checkpoint['query_state_dict'])
tra_net.load_state_dict(checkpoint['tra_state_dict'])
tim_net.load_state_dict(checkpoint['tim_state_dict'])
pitch_net.load_state_dict(checkpoint['pitch_state_dict'])

print("loaded weights")
i = 0
for item in testdata.test_samples():
	
	###
	mix = torch.from_numpy(item['mix'])
	unified_mix = (0.5*mix[0,:,:] + 0.5*mix[1,:,:]).unsqueeze(dim = 0)
	#both need to be 3d
	part1 = torch.from_numpy(item['mix'][0,:,:]).unsqueeze(dim = 0)
	part2 = torch.from_numpy(item['mix'][1,:,:]).unsqueeze(dim = 0)
	
	print(mix.shape, part1.shape, part2.shape)
	
	unified_mix_spec = wav2spec(config_spec, unified_mix)
	query1 = wav2spec(config_spec, part1)
	query2 = wav2spec(config_spec, part2)
	
	print(unified_mix_spec.shape, query1.shape, query2.shape)
	query1_out = query_net(query1)
	query2_out = query_net(query2)
	
	out1, out_conc1 = enc_net(unified_mix_spec, query1_out)
	out2, out_conc2 = enc_net(unified_mix_spec, query2_out)
	
	tra_out1 = F.softmax(tra_net(out1), 1)
	tra_out2 = F.softmax(tra_net(out2), 1)
	
	pitch_out1 = pitch_net(tra_out1)
	pitch_out2 = pitch_net(tra_out2)

	
	tim_out1 = tim_net(out_conc1 + [out1])
	tim_out2 = tim_net(out_conc2 + [out2])
	
	in_to_dec1 = multipleEntanglement(pitch_out1, tim_out1)
	in_to_dec2 = multipleEntanglement(pitch_out2, tim_out2)
	
	sep1 = dec_net(in_to_dec1[-1], in_to_dec1[:-1])
	sep2 = dec_net(in_to_dec2[-1], in_to_dec2[:-1])
	
	sep1wav = spec2wav(sep1, config_s2w).detach().squeeze(dim = 0)
	sep2wav = spec2wav(sep2, config_s2w).detach().squeeze(dim = 0)
	
	print(sep1wav.shape, sep2wav.shape)
	torchaudio.save("testsep1_"+str(i)+"_.wav", sep1wav, 16000)
	torchaudio.save("testsep2_"+str(i)+"_.wav", sep2wav, 16000)
	torchaudio.save("origmix_"+str(i)+"_.wav", unified_mix.squeeze(dim = 0), 16000)
	torchaudio.save("origsep1_"+str(i)+"_.wav", mix[0,:,:], 16000)
	torchaudio.save("origsep2_"+str(i)+"_.wav", mix[1,:,:], 16000)
	
	i += 1
	if i == 2:
		break
	
	
#print(len(testdata))
