from src.dataset.urmp.load_urmp_data import load_test_data

from src.utils.multiEpochsDataLoader import MultiEpochsDataLoader as DataLoader

import torchaudio
import torch
testdata = load_test_data(r"D:\Downloads\URMP\Dataset\reproduce_data\urmp")

print(testdata.test_samples())
		
for item in testdata.test_samples():
	print(type(item))
	for k, v in item.items():
		print(k)
		
	print(type(item['instrs']))
	print(len(item['instrs']))
	print(item['instrs'][0])
	
	print(item['mix'].shape)
	print(type(item['mix']))
	#numtowav = item['mix'].astype('float16')
	numtowav = torch.from_numpy(item['mix'])
	print(type(numtowav))
	torchaudio.save("testsample1.wav", numtowav[0,:,:], 16000)
	torchaudio.save("testsample2.wav", numtowav[1,:,:], 16000)
	break
#print(len(testdata))