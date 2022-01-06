import torch
import torch.nn as nn
import sys
import os
from src.models.modelFactory import ModelFactory

sys.path.insert(1, os.path.join(sys.path[0], '..'))


def load_gpu_model(conf_path, model_name, model_path):
	nnet = ModelFactory(conf_path, model_name)
	nnet.load_state_dict(torch.load(model_path), strict=False)
	return nnet.cuda()



