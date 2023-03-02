# CNN_Classification_Earthquake_Collapse_Explosion

**Please report confusions/errors to yzcd4@umsystem.edu

# Versions:
python 3.6.9 \
numpy 1.19.5 \
Tensorflow 1.14.0 \
Keras 2.2.5 

------------------------------------

# Data

unzip list.zip to release the lists of events

# Models

Trained models with origianl dataset are under Models/Original \
Trained models with refined dataset are under Models/Refined \
New trained models with Train_original/train.py are under Models/New_1 \
New trained models with Train_refined/train.py are under Models/New_2 

# Training

Use Train_original/train.py to train CNN with original labels \
Use Train_refined/train.py to train CNN with refined labels

# Loading

change model_rout in line 21 to load different models
