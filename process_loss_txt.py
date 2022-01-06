#process losses.txt

import matplotlib.pyplot as plt
import numpy as np
file1 = open('losses1.txt', 'r')
file2 = open('losses2.txt', 'r')
file3 = open('losses3.txt', 'r')

f1_cont = file1.readlines()
f2_cont = file2.readlines()
f3_cont = file3.readlines()

cont_list = [f1_cont, f2_cont, f3_cont]
data_list = []
for i in range(3):
	data = []
	for line in cont_list[i]:
		data.append(float(line[7:13]))
	data_list.append(data)
	
list_of_sum_list = []
for list in data_list:
	i = 0
	sum_list = []
	sum = 0
	for item in list:
		sum += item
		i += 1
		if i == 3200:
			i = 0
			sum_list.append(sum/3200)
			sum = 0
	list_of_sum_list.append(sum_list)
	sum_list = []
print(sum_list)

plt.figure()
plt.plot(list_of_sum_list[0])
plt.ylim(0, 1.1*max(list_of_sum_list[0]))
plt.ylabel("Pitch Loss")
plt.xlabel("Epoch")
plt.savefig('PitchLoss.png')

plt.figure()
plt.plot(list_of_sum_list[1])
plt.ylim(0, 1.1*max(list_of_sum_list[1]))
plt.ylabel("Query Loss")
plt.xlabel("Epoch")
plt.savefig('QueryLoss.png')

plt.figure()
plt.plot(list_of_sum_list[2])
plt.ylim(0, 1.1*max(list_of_sum_list[2]))
plt.ylabel("Spectrogram Loss")
plt.xlabel("Epoch")
plt.savefig('SpectrogramLoss.png')