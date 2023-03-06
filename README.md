# CNN_Classification_Earthquake_Collapse_Explosion

**Please report confusions/errors to yzcd4@umsystem.edu 

# Versions:
python 3.6.9 \
numpy 1.19.5 \
Tensorflow 1.14.0 \
Keras 2.2.5 

------------------------------------

# Data

Waveform data can be found at 10.6084/m9.figshare.22207498 \
	unzip under the Data/

unzip list.zip to release the lists of events 

in each *.phase.info.ls under list: \
	1st column is station name \
 	2nd column is Picking phases \
	3rd and 4th columns are Picking time \
	5th column is magnitude \
	6th column is event type 
    
list.error contains events have different labels between human and CNN in the original training \
    1st column is original human label \
    2nd column is event name \
    3rd, 4th, and 5th columns are output of CNN with majority voting. They represent possibilities of earthquake, collapse, and explosion, respectively. \
    6th column is new label after manually checking \
    7th column is 0 (remove) and 1 (keep)

# Models

Trained models with origianl dataset are under Models/Original \
Trained models with refined dataset are under Models/Refined \
New trained models with Train_original/train.py are under Models/New_1 \
New trained models with Train_refined/train.py are under Models/New_2 

# Training

Use Train_original/train.py to train CNN with original labels \
Use Train_refined/train.py to train CNN with refined labels

# Loading

Use Load_original/load.py to test with original labels \
Use Load_refined/load.py to test with refined labels \
change model_rout in line 21 to load different models
