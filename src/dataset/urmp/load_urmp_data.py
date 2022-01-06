from src.dataset.urmp.urmp_test import UrmpTest
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

def load_test_data(data_path):
	test_data = UrmpTest(f'{data_path}/testset/test.lst', f'{data_path}/testset/query.lst')
	return test_data
