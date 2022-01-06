import torch.nn as nn
import torch
import torch.nn.functional as F

from reyes_tan_utilities import *

from src.dataset.urmp.urmp_sample import *
from src.utils.multiEpochsDataLoader import MultiEpochsDataLoader as DataLoader

from src.inference.inference import merge_batches

import torchaudio
#reyes_models patterned after https://github.com/anonymous-16/a-unified-model-for-zero-shot-musical-source-separation-transcription-and-synthesis/tree/main/src/models

#Temporary
torch.autograd.set_detect_anomaly(True)

class Encoder(nn.Module):
	
	def __init__(self, config):
		#needed else "AttributeError: cannot assign module before Module.__init__() call"
		super(Encoder, self).__init__()
		
		#Unpack the config
		
		#Channels
		self.in_channels = config['in_channels']
		self.out_channels = 2
		
		#Momentum
		self.momentum = config['momentum']
		
		#Conditioning dimension from querynet
		self.cond_dim = config['cond_dim']
		
		#input size. Based on frequency. In this case 256/2 or 128
		self.input_size = config['input_size']
		
		#bn0
		self.bn0 = nn.BatchNorm2d(1, momentum = 0.01)
		
		
		#initialize the blocks
		#Each block is 
		#conv - bn -film
		self.num_blocks = config['num_blocks']
		self.conv_layers = nn.ModuleList()
		self.bn_layers = nn.ModuleList()
		self.film_layers = nn.ModuleList()
		
		#for transcriptor later
		self.latent_rep_channels = []
		
		#Source code has an extra one
		for i in range(self.num_blocks + 1):
			self.conv_layers.append(nn.Conv2d(in_channels = self.in_channels, \
							out_channels = self.out_channels, \
							kernel_size = (3,3), \
							stride = (1,1), \
							padding = (1,1), \
							bias = False))
			self.bn_layers.append(nn.BatchNorm2d(self.out_channels))
			
			self.film_layers.append(FiLM1DLayer(self.cond_dim, \
							self.out_channels, \
							self.input_size))
							
			self.latent_rep_channels.append([self.out_channels, self.input_size])
			
			self.in_channels = self.out_channels
			self.out_channels *= 2
			self.input_size //= 2
			
	def forward(self, input, condition):
	
		#There model seems to be not a strict Unet in that the concatenation only happens once
		#Update. It is a list which is unpacked decoder side. It is indeed a strcit Unet
		x = input
		x = self.bn0(x)
		concat_tensors = []
		for i in range(self.num_blocks):
		
			#normal "layers"
			x = self.conv_layers[i](x)
			x = self.bn_layers[i](x)
			x = F.relu_(x)
			
			#film layers
			#print(x.shape, "pre film for concat testing")
			x = self.film_layers[i](x, condition)
			
			#print(x.shape, "post film for concat testing")
			#others
			concat_tensors.append(x)
			x = F.avg_pool2d(x, kernel_size = (1, 2))
			
		x = self.conv_layers[self.num_blocks](x)
		x = self.bn_layers[self.num_blocks](x)
		x = F.relu_(x)
		
		
		x = self.film_layers[self.num_blocks](x, condition)
		return x, concat_tensors
		
		
class Decoder(nn.Module):

	def __init__(self, config):
		super(Decoder, self).__init__()
		
		#Unpack config
		
		
		layers = nn.ModuleList()
		self.in_channels = config['in_channels']
		self.momentum = config['momentum']
		self.num_blocks = config['num_blocks']

		#Layers lists
		self.conv_tr_layers = nn.ModuleList()
		self.bn_layers = nn.ModuleList()
		
		#
		self.conv_layers = nn.ModuleList()
		self.bn_layers_2 = nn.ModuleList()
		
		#Decoder building blocks
		self.in_channels = self.in_channels
		self.out_channels = self.in_channels//2
		for i in range(self.num_blocks):
			
			self.conv_tr_layers.append(torch.nn.ConvTranspose2d(in_channels = self.in_channels, \
				out_channels = self.out_channels, \
				kernel_size = (3,3), \
				stride = (1,2), \
				padding = (0,0), \
				output_padding = (0,0),
				bias = False)
				)
			
			self.bn_layers.append(nn.BatchNorm2d(self.out_channels, momentum = self.momentum))
			
			
			
			#The 2 "channels" are the dec output and the corresponding concatenation
			self.conv_layers.append(nn.Conv2d(in_channels = self.out_channels*2, \
				out_channels = self.out_channels, \
				kernel_size = (3, 3), \
				stride = (1,1), \
				padding = (1,1), \
				bias = False))
			
			self.bn_layers_2.append(\
				nn.BatchNorm2d(self.out_channels,\
					momentum = self.momentum)
				)
				
			#the next in channels will be divded by 2
			self.in_channels = self.out_channels
			self.out_channels = self.out_channels//2
			
		#The last layers
		#Bottom as they call it
		self.bottom = nn.Conv2d(in_channels = self.in_channels, \
							out_channels = self.out_channels, \
							kernel_size = (1,1), \
							stride = (1,1), \
							bias = True)
	def forward(self, input, concat_tensors):
		
		x = input
		for i in range(self.num_blocks):
		
			#print(x.size())
			x = self.conv_tr_layers[i](x)
			x = self.bn_layers[i](x)
			x = F.relu_(x)
			
			
			#Pruning to match
			x = x[:, :, 1:-1, : -1]
			
			#print(x.size(), "x size per dec before the torchcat")
			#print(concat_tensors[-i-1].size(), "x size per dec before the torchcat")
			#print(concat_tensors[-i-1].size())
			#Since they the unpacking is FIFO use -i-1
			x = torch.cat((x, concat_tensors[-i-1]), dim = 1)
			
			x = self.conv_layers[i](x)
			#print(x.size(), "x per dec cycle post conv")
			x = self.bn_layers_2[i](x)
			#print(x.size(), "x per dec cycle post bn")
			x = F.relu_(x)
			#print(x.size(), "x per dec cycle")
			
		x = self.bottom(x)
		return x

class Query(nn.Module):

	def __init__(self, config):
		super(Query, self).__init__()

		#Unpack config
		self.num_blocks = config['num_blocks']
		self.in_channels = config['in_channels']
		self.input_size = config['input_size']
		self.output_size = self.input_size
		self.cond_dim = config['cond_dim']
		self.out_channels = 2
		
		#Layers
		self.conv_layers = nn.ModuleList()
		self.bn_layers = nn.ModuleList()
		
		for i in range(self.num_blocks):
			self.conv_layers.append(nn.Conv2d(\
				in_channels = self.in_channels, \
				out_channels = self.out_channels, \
				kernel_size = (3,3), \
				stride = (1,1), \
				padding = (1,1), \
				bias = False\
				))
				
			self.bn_layers.append(nn.BatchNorm2d(\
				self.out_channels\
				))
			
			self.in_channels = self.out_channels
			self.out_channels *= 2
			self.output_size //= 2
		
		self.bottom = nn.Conv1d(\
			in_channels = self.in_channels * self.output_size, \
			out_channels = self.cond_dim, \
			kernel_size = 1\
			)

	def forward(self, input):
		x = input
		
		for i in range(self.num_blocks):
			x = self.conv_layers[i](x)
			x = self.bn_layers[i](x)
			
			#insert relu_
			x = F.avg_pool2d(x, kernel_size = (2,2))
			
		x = x.transpose(-1, -2).flatten(1,2)
		x = self.bottom(x)
		x = torch.tanh(x)
		
		#query mode in their code reduces dimension
		
		x = x.mean(-1)
		x = x.unsqueeze(dim = 2)
		return x
		
		

class Transcriptor(nn.Module):

	def __init__(self, config):
		super(Transcriptor, self).__init__()
		
		self.num_blocks = config['num_blocks']
		self.output_dim = config['output_dim']
		self.in_channels = config['in_channels']
		self.input_size = config['input_size']
		self.out_channels = self.in_channels * 2
		self.momentum = config['momentum']
		
		self.conv_layers = nn.ModuleList()
		self.bn_layers = nn.ModuleList()
		self.conv_layers_2 = nn.ModuleList()
		self.bn_layers_2 = nn.ModuleList()
		
		
		
		for i in range(self.num_blocks):
			self.conv_layers.append(nn.Conv2d(\
							in_channels = self.in_channels, \
							out_channels = self.out_channels, \
							kernel_size = (3,3), \
							stride = (1,1), \
							padding = (1,1), \
							bias = False))
			self.bn_layers.append(nn.BatchNorm2d(\
							self.out_channels, \
							momentum = self.momentum))
			
			self.conv_layers_2.append(nn.Conv2d(\
							in_channels = self.out_channels, \
							out_channels = self.out_channels, \
							kernel_size = (3,3), \
							stride = (1,1), \
							padding = (1,1), \
							bias = False))
			self.bn_layers_2.append(nn.BatchNorm2d(\
							self.out_channels, \
							momentum = self.momentum))
							
							
			self.in_channels = self.out_channels
			self.out_channels *= 2
			self.input_size //= 2
		self.bottom = nn.Conv1d(\
			in_channels = self.in_channels * self.input_size, \
			out_channels = self.output_dim, \
			kernel_size = 1, \
			stride = 1, \
			bias = True)
			
		self.notes_num = self.output_dim
			
	def forward(self, input):
	
		x = input
		
		for j in range(self.num_blocks):
			x = self.conv_layers[j](x)
			x = self.bn_layers[j](x)
			x = F.relu_(x)
			x = self.conv_layers_2[j](x)
			x = self.bn_layers_2[j](x)
			x = F.relu_(x)
			x = F.avg_pool2d(x, kernel_size = (1,2))
			
		x = x.transpose(-1, -2).flatten(1, 2)
		x = self.bottom(x)
		
		return x
		
#simple enough not to need a config

class Pitch(nn.Module):

	def __init__(self, notes_num, latent_rep_channels):
	
		super(Pitch, self).__init__()
		self.notes_num = notes_num
		self.latent_rep_channels = latent_rep_channels
		self.layers = nn.ModuleList()
		for latent_rep in latent_rep_channels:
			D = latent_rep[0] * latent_rep[1]
			K = 2 * D
			
			self.layers.append(nn.Conv1d(\
						in_channels = self.notes_num, \
						out_channels = K, \
						kernel_size = 1, \
						stride = 1, \
						bias = False))
	
	def forward(self, input):
		output_tensors = []
		x = input
		#print(x.size(), "x size insidepitch")
		
		for i, layer in enumerate(self.layers):
		
			#the shape changes therefore maintain input's originality
			x = self.layers[i](input)
			x = x.reshape((x.shape[0], \
						self.latent_rep_channels[i][0] * 2, \
						self.latent_rep_channels[i][1], \
						x.shape[-1])).transpose(-1, -2)
			output_tensors.append(x)
			
		return output_tensors
	
#again doesn't really need config
class Timbre(nn.Module):
	
	def __init__(self, latent_rep_channels, momentum):
		super(Timbre, self).__init__()
		
		self.conv_layers = nn.ModuleList()
		self.bn_layers = nn.ModuleList()
		
		output_dim = 0
		for latent_rep in latent_rep_channels:
			channels = latent_rep[0]
			output_dim += (latent_rep[0] * latent_rep[1])
			
			self.conv_layers.append(nn.Conv2d(\
						in_channels = channels, \
						out_channels = channels, \
						kernel_size = (3,3), \
						stride = (1,1), \
						padding = (1,1), \
						bias = False))
						
			self.bn_layers.append(nn.BatchNorm2d(\
						channels, \
						momentum = momentum))
			
		self.output_dim = output_dim
		
	def forward(self, input):
		output_tensors = []
		x = input
		
		for j in range(len(self.conv_layers)):
			x = self.conv_layers[j](input[j])
			x = self.bn_layers[j](x)
			x = F.relu_(x)
			output_tensors.append(x)
			
		return output_tensors
#Taken almost as is from source code since this is a building block that is based on another paper. LinearBlock1d was changed to nn.Conv1d as part of the rebuilding/unwrapping of all the multiple layers of networks that they have.
class FiLM1DLayer(nn.Module):
	"""A 1D FiLM Layer. Conditioning a 1D tensor on a 3D tensor using the FiLM.
		Input : [B x input_dim], [B x channels_dim x T x feature_size]
		Output : [B x channels_dim x T x feature_size]
	
		Parameters
		-----------
		input_dim : int
			The number of channels of the input 1D tensor.
		channels_dim : int
			The number of channels of the input 3D tensor.
		feature_size : int
			The feature size of the input 3D tensor.
	
		Properties
		----------
		channels_dim : int
			See in Parameters.
		feature_size :
			See in Parameters.
		gamma : `LinearBlock1D`
			gamma in FiLM..
		beta : `LinearBlock1D`
			beta in FiLM
		
	"""

	def __init__(self, input_dim, channels_dim, feature_size):
		super(FiLM1DLayer, self).__init__()

		#linear block 1d is just conv1d
		self.gamma = nn.Conv1d(in_channels = input_dim, \
					out_channels = channels_dim * feature_size, \
					kernel_size = 1)
		self.beta = nn.Conv1d(in_channels = input_dim, \
					out_channels = channels_dim * feature_size, \
					kernel_size = 1)

		self.channels_dim = channels_dim
		self.feature_size = feature_size

	def forward(self, input, condition):
		"""
		Parameters
		----------
		input : tensor
			The input 3D tensor.
			[B x channels_dim x T x feature_size]
		condition:	tensor
			The input 1D (condition) tensor.
			[B x input_dim]
		Returns
		-------
				:	tensor
			[B x channels_dim x T x feature_size]		
		"""	
		x = input
		y = condition
		channels_dim = self.channels_dim
		feature_size = self.feature_size
		g = self.gamma(y).reshape((y.shape[0], channels_dim, feature_size, -1)).transpose(-1, -2)
		b = self.beta(y).reshape((y.shape[0], channels_dim, feature_size, -1)).transpose(-1, -2)
		return x * g + b
		
#taken as is since this is just data manipulation
def align(a, b, dim):
	return a.transpose(0, dim)[:b.shape[dim]].transpose(0, dim)
	
#entangle and multipleentanglement were taken as is since there wasn't much unwrapping to be done
def entangle(p, ti):
	"""
		Entanglement operation discribed in Section 3.3 in the paper.
		[p_gamma p_beta] = p
		z = p_gamma * ti + p_beta
		Note that D = C * F and K = 2 * C * F
		Parameters
		----------
		p : tensor
			[B x (2*C) x T x F]
			the disentangled pitch representation
		ti : tensor
			[B x C x T x F]
			the disentangled timbre representation
		Returns
		---------
			: tensor
			[B x C x T x F]
	"""
	p_gamma = p[:, :ti.shape[1]]
	p_beta = p[:, ti.shape[1]:]
	return p_gamma * ti + p_beta

#onehot also taken as is
def onehot(x, dim, classes_num):
	x = x.unsqueeze(dim)
	shape = list(x.shape)
	shape[dim] = classes_num
	y = torch.zeros(shape).to(x.device).scatter_(dim, x, 1)
	return y
def multipleEntanglement(p_tensors, ti_tensors):
	"""
		Entanglement operation for a list of pitch and timbre representations. See more details in func `entangle`.
		Parameters
		----------
		p_tensors : list of tensor
			A list of N pitch representations, the shape of each tensor in which is [B x (2*C) x T x F]
		ti_tensors : list of tensor
			A list of N timbre representations, the shape of each tensor in which is [B x C x T x F]
		Returns
		---------
			: list of tensor
			A list of N tensors, the shape of each tensor in which is [B x C x T x F]
	"""

	tensors = []
	for i, p in enumerate(p_tensors):
		tensors.append(entangle(p, ti_tensors[i]))
	return tensors

#also taken as is
def move_data2cuda(urmp_batch):
	mix, another_mix, batch = urmp_batch
	separated, query, another_query, pitch_target, another_pitch_target = batch
	batch = [separated, query, another_query, pitch_target, another_pitch_target]
	for i, b in enumerate(batch):
		batch[i] = b.cuda()
	mix = mix.cuda()
	another_mix = another_mix.cuda()
	return mix, another_mix, batch	
if __name__ == "__main__":
	
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

	urmpsamp = torch.randn((2, 1, 48000))
	urmpspec = torch.randn((2,1,301,256))
	newinitial = wav2spec(config_spec, urmpsamp)
	
	print(urmpsamp.shape, newinitial.shape, "urmpsamp, newitinital shapes reyesmodels.py")
	
	enc_net = Encoder(config_enc)
	initial = torch.randn((2,1,301,256))
	initial_cond = torch.randn((2,6,1))
	out, out_conc = enc_net(initial, initial_cond)
	print(out.shape, "out of enc shape")
	dec_net = Decoder(config_dec)
			
	config_tra = {'num_blocks' : 2, \
					'output_dim' : 89, \
					'in_channels' : enc_net.latent_rep_channels[-1][0], \
					'input_size' : enc_net.latent_rep_channels[-1][1], \
					'momentum' : 0.01\
					}
					
	tra_net = Transcriptor(config_tra)
	
	pitch_net = Pitch(tra_net.notes_num, enc_net.latent_rep_channels)
	tim_net = Timbre(enc_net.latent_rep_channels, 0.01)
	query_net = Query(config_query)
	
	queryout = query_net(urmpspec)
	#queryout = queryout.unsqueeze(dim = 2)
	
	print(queryout.shape, "query shape main rt_models.py")
	initial = torch.randn((2,1,301,256))
	initial_cond = torch.randn((2,6,1))
	out, out_conc = enc_net(initial, initial_cond)
		
	print(out.shape, "out of enc shape")
	final = dec_net(out, out_conc)
	
	print(final.size(), "final size")
	
	finalwav = spec2wav(final, config_s2w)
	print(finalwav.size(), "final wav size reyesmodels.py")
	
	
	tra_out = tra_net(out)
	print(tra_out.size(), "tra_out size")
	pitch_out = pitch_net(tra_out)
	print(len(pitch_out))
	print(pitch_out[0].size(), "pitch_out_size")
	
	tim_out = tim_net(out_conc + [out])
	print(len(tim_out), "tim out len")
	print(tim_out[0].shape, "tim out shape")
	
	in_to_dec = multipleEntanglement(pitch_out, tim_out)
	in_short_dec = in_to_dec[-1]
	in_to_dec = in_to_dec[:-1]
	
	sep = dec_net(in_short_dec, in_to_dec)
	print(sep.shape)
	#exit()
	### Data loading taken from their code

	urmp_data = UrmpSample('utilities_taken_as_is/urmp.cfg', 'train')
	print(urmp_data)
	
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
	
	
	sep_spec = merge_batches(final, duration_axis =-2)
	print(sep_spec.shape)
	sep_spec = sep_spec.unsqueeze(dim = 0)
	print(sep_spec.shape)
	finalwav = spec2wav(sep_spec, config_s2w)
	print(finalwav.shape)
	
	#wavs are 2d
	finalwav = finalwav.detach().squeeze(dim = 0)
	#Their sample rate is 16k apparently
	print(type(finalwav), "finalwa type")
	print(finalwav.dtype, "finalwav dtype")
	torchaudio.save("first.wav", finalwav.type(torch.float32), 16000)
	loss_list_1 = []
	loss_list_2 = []
	print(len(urmp_loader) -1, "length of urmp loader")
	
	file1 = open("losses1.txt", 'w')
	file2 = open("losses2.txt", 'w')
	file3 = open("losses3.txt", 'w')
	for epoch in range(200):
		print("Starting epoch ", epoch)
		for i_batch, urmp_batch in enumerate(urmp_loader):
			if i_batch % 200 == 0:
				print("Starting i_batch", i_batch)
			#splitting the data taken as is since this is just processing
			#urmp_batch = move_data2cuda(urmp_batch)
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
					#print(query.shape, "testing out query shape")
					for i in range(query.shape[1]):
						query_spec = wav2spec(config_spec, query[:,i])
						a_query_spec = wav2spec(config_spec, another_query[:,i])
						
						h = query_net(query_spec)
						hc = query_net(a_query_spec)
						
						latent_vectors.append([h, hc])
						
						#print(h.size(), hc.size(), "h hc size from training loop")
					
					sim = 0.
					for i in range(query.shape[1]):
						next_i = (i + 1) % query.shape[1]
						sim += torch.mean((latent_vectors[i][0] - latent_vectors[i][1])**2, dim = -1) + \
							torch.relu(1./8. - torch.mean((latent_vectors[i][0] - latent_vectors[next_i][1])**2, dim = -1))
							
					sim_loss = sim.mean()/query.shape[1]
					#print(sim_loss)
				
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
					
					#print(spec_loss, pitch_loss, "spec and pitch loss msi-dis")

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
					op.zero_grad()
				else:
					op.step()
					op.zero_grad()
					
			file1.write(str(pitch_loss))
			file1.write("\n")
			file2.write(str(sim_loss))
			file2.write("\n")
			file3.write(str(spec_loss))
			file3.write("\n")
			file1.close()
			file2.close()
			file3.close()
			file1 = open("losses1.txt", 'a')
			file2 = open("losses2.txt", 'a')
			file3 = open("losses3.txt", 'a')
			del others_loss
			del sim_loss

			#print("===========================================Done with ",i_batch,"================")
		
			
		torch.save({\
				'epoch': epoch, \
				'enc_state_dict': enc_net.state_dict(), \
				'dec_state_dict': dec_net.state_dict(), \
				'query_state_dict': query_net.state_dict(), \
				'tra_state_dict' : tra_net.state_dict(), \
				'tim_state_dict' : tim_net.state_dict(), \
				'pitch_state_dict' : pitch_net.state_dict(), \
				'optimizers': optimizers},\
				"checkpoint_" + str(epoch) + ".pt")
	file1.close()
	file2.close()		
	file3.close()
		
	'''
		urmp_batch = urmp_batch[0]
		print(urmp_batch.shape, "size of urmpbatch[0] r_t_models.oy")
		urmp_batch = urmp_batch[1,:,:]
		print(urmp_batch.shape, "size after taking only first of abtch r_t_models.py")
		torchaudio.save("datasampleorig2.wav", urmp_batch, 16000)
		
		urmp_batch = urmp_batch.unsqueeze(dim = 0)
		urmp_batch = wav2spec(config_spec, urmp_batch)
		print(urmp_batch.shape, "size after wav2spec r_t_models.py")
		
		urmp_batch = spec2wav(urmp_batch, config_s2w)
		print(urmp_batch.shape, "size after spec2wav r_t_models.py")
		urmp_batch = urmp_batch.detach().squeeze(dim = 0)
		torchaudio.save("datasample2.wav", urmp_batch, 16000)
	'''
		
	'''

	for i_batch, urmp_batch in enumerate(urmp_loader):
		print(i_batch, "Doing this batch")
		loss = []
		
		out, out_conc = enc(initial)
		final = dec(out, out_conc)
		
		for j in range(len(optimizers)):
			op = optimizers[j]['opt']
			
			op.zero_grad()
			loss.append(torch.abs(out[0,0,0,0])) #hard code for now.
			loss[j].backward(retain_graph = True)
			
		#apparently you can't take a step yet until you process everything
		for j in range(len(optimizers)):
			op = optimizers[j]['opt']
			op.step()
			op.zero_grad()
		del loss
		
		if i_batch == 10:
			break
	print("reached sample")
	'''
	
	
	
	
	
	
	
	
	
	
	
	
