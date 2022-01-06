# EE 286 Final Project: Music Source Seperation
Main Motivation/Reference: https://github.com/Kikyo-16/A-unified-model-for-zero-shot-musical-source-separation-transcription-and-synthesis
This project is a partial reimplementation of the project by Lin et al. Since the source code is already in python the main focus was understanding the code as well as extracting the music source separation component. Some code unwrapping was also done for the model construction and training portion.
1.  Lin et al. A Unified Model for Zero-shot Music Source Separation, Transcription and Synthesis. 2021. https://arxiv.org/abs/2108.03456




## Quick Start
All commands are run once you change directory from the main project. 

### 1. Requirements.
The source code does not have a requirments.txt. As such we manually installed the requirements when indicated as missing by the command line.

### 2a. Data Preperation 
Download the dataset from [URMP homepage](http://www2.ece.rochester.edu/projects/air/projects/URMP.html).After this, you will have to edit the folder "15_Surprise_tpt_tpt_tbn". Some files in the dataset are mistakenly labelled as tpt when they should be tbn. Then in urmp_generate_dataset.py lines 44 and 46 the "resolution" should be changed to "ref". Finally run generate_features.py and generate_dataset.py. Adjust the arguments as needed.



### 3. Train the Network
Run the following command to train the network. 
```
python3 reyes_tan_models_upgraded.py
```

### 4. Evaluate the Network
Run the following command to produce separated music samples as well as graph the outputs
```
python3 separate_test.py
python3 process_loss_text.py
```
