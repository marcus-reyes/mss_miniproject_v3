import torch.nn as nn
import torch
import torch.nn.functional as F

from reyes_tan_utilities import *

from src.dataset.urmp.urmp_sample import *
from src.utils.multiEpochsDataLoader import MultiEpochsDataLoader as DataLoader

from src.inference.inference import merge_batches

import torchaudio

from reyes_tan_models_upgraded import *

#check if we can load the state_dicts
if __name__ == "__main__":
	config_enc = {'num_blocks' : 1, \
					'in_channels' : 1, \
					'momentum' : 0.01, \
					'cond_dim' : 6, \
					'input_size' : 128, \
					}
					
	config_dec = {'num_blocks' : 1, \
				'in_channels' : 4, \
				'momentum' : 0.01}
				

	config_query = {'num_blocks' : 2, \
		'in_channels' : 1, \
		'input_size' : 128, \
		'cond_dim' : 6, \
		}
	#n_fft is the same as win_length
	config_spec = {'center' : True, \
				'freeze_parameters' : True, \
				'n_fft' : 256, \
				'hop_length' : 160, \
				'pad_mode' : "reflect", \
				'window' : "hann", \
				'win_length' : 256}
				
				
	# should be 256 not 128
	#n_fft is the same as win_length
	config_s2w = {'fps' : 100, \
				'samp_rate' : 16000, \
				'window' : "hann", \
				'n_fft' : 256, \
				'hop_length' : 160, \
				'win_length' : 256, \
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
	
	
	#checkpoint = torch.load('checkpoint_0.pt')
	
	
	print("Initialized models")
	'''
	enc_net.load_state_dict(checkpoint['enc_state_dict'])
	dec_net.load_state_dict(checkpoint['dec_state_dict'])
	query_net.load_state_dict(checkpoint['query_state_dict'])
	tra_net.load_state_dict(checkpoint['tra_state_dict'])
	tim_net.load_state_dict(checkpoint['tim_state_dict'])
	pitch_net.load_state_dict(checkpoint['pitch_state_dict'])

	print("loaded weights")
	'''	
	
	##Do a runthrough
	urmp_data = UrmpSample('utilities_taken_as_is/urmp.cfg', 'train')

	
	urmp_loader = DataLoader(urmp_data, \
			batch_size = 2, \
			shuffle = False, \
			num_workers = 1, \
			pin_memory = True, \
			persistent_workers = False,
			collate_fn = urmp_data.get_collate_fn())
	parameters = {}
	parameters['query'] = list(query_net.parameters())
	parameters['enc'] = list(enc_net.parameters())
	parameters['dec'] = list(dec_net.parameters())
	parameters['tim'] = list(tim_net.parameters())
	parameters['tra'] = list(tra_net.parameters())
	parameters['pitch'] = list(pitch_net.parameters())
	
	optimizers = []
	#since resume epoch is 0
	for param in parameters:
		optimizer = torch.optim.Adam(parameters[param], \
						lr = 5e-4/(2**(0 // 100)))
		optimizers.append({'mode' : param, 'opt' : optimizer, 'name' : param})		
	loss_list_1 = []
	loss_list_2 = []
	for i_batch, urmp_batch in enumerate(urmp_loader):
		
		if i_batch == 1:
			break
		
		#splitting the data taken as is since this is just processing
		mix, another_mix, batch = urmp_batch
		separated, query, another_query, pitch_target, another_pitch_target = batch
		
		#obtain the losses
		for j in range(len(optimizers)):
			mode = optimizers[j]['mode']
			op = optimizers[j]['opt']
			op.zero_grad()
			
			#query: obtain the loss similar to how they did it.
			latent_vectors = []
			hQuery = []
			if mode == "query":
				print(query.shape, "testing out query shape")
				for i in range(query.shape[1]):
					query_spec = wav2spec(config_spec, query[:,i])
					a_query_spec = wav2spec(config_spec, another_query[:,i])
					
					h = query_net(query_spec)
					hc = query_net(a_query_spec)
					
					latent_vectors.append([h, hc])
					
					print(h.size(), hc.size(), "h hc size from training loop")
				
				sim = 0.
				for i in range(query.shape[1]):
					next_i = (i + 1) % query.shape[1]
					sim += torch.mean((latent_vectors[i][0] - latent_vectors[i][1])**2, dim = -1) + \
						torch.relu(1./8. - torch.mean((latent_vectors[i][0] - latent_vectors[next_i][1])**2, dim = -1))
						
				sim_loss = sim.mean()/query.shape[1]
				print(sim_loss)
			
			#enc and dec tra tim pitch just choose one
			elif mode == "enc": 
				spec_losses = []
				mix_spec = wav2spec(config_spec, mix)
				another_mix_spec = wav2spec(config_spec, another_mix)
				target = onehot(pitch_target, 1, 89)
				another_target = onehot(another_pitch_target, 1, 89)
				
				pitch_transcription = []
				another_pitch_transcription = []
				for i in range(separated.shape[1]):
					source_spec = wav2spec(config_spec, separated[:,i])
					
					query_spec = wav2spec(config_spec, query[:,i])
					hQuery = query_net(query_spec)
					
					out, out_conc = enc_net(mix_spec, hQuery)
					tra_out = tra_net(out)
					tra_out_2 = F.softmax(tra_out, 1)
					
					out_2, out_conc_2 = enc_net(another_mix_spec, hQuery)
					
					pitch_out = pitch_net(tra_out_2)
					tim_out = tim_net(out_conc_2 + [out_2])
					
					in_to_dec = multipleEntanglement(pitch_out, tim_out)
					
					in_short_dec = in_to_dec[-1]
					in_to_dec = in_to_dec[:-1]
					
					sep = dec_net(in_short_dec, in_to_dec)
					
					#need sep and tra_out for loss
					
					pitch_transcription.append(tra_out)
					spec_loss = torch.abs(sep - align(source_spec, sep, -2))
					spec_losses.append(spec_loss)
					
				spec_loss = torch.stack(spec_losses, 1)
				spec_loss = spec_loss.mean()
				
				transcription = torch.stack(pitch_transcription, 2)
				pitch_loss = nn.CrossEntropyLoss()(transcription, align(pitch_target, transcription, -1))
				
				print(spec_loss, pitch_loss, "spec and pitch loss msi-dis")

				'''
				spec_loss.backward(retain_graph = True)
				op.step()
				print(spec_loss)
				loss_list_2.append(spec_loss)
				del spec_loss
				'''

		#step needs to be all at once for the enc aand dec
		others_loss = spec_loss + pitch_loss
		others_loss.backward(retain_graph = True)
		sim_loss.backward(retain_graph = True)
		for j in range(len(optimizers)):
			
			mode = optimizers[j]['mode']
			op = optimizers[j]['opt']
			
			if mode == "query":
			
				op.step()
				loss_list_1.append(sim_loss)
				op.zero_grad()
			else:
				op.step()
				loss_list_2.append(spec_loss)
				op.zero_grad()
					
