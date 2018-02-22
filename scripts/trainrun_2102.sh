# Testing the effect of larger resnet models and different datasets

# Training script to run
TRAIN_SCRIPT="train_gazenet.py"

# Datasets
D1="gaze_real_150118"
D2="gaze_real_180220"
D3="gaze_fake_100118"

# Only Real Data
python train_gazenet.py --datasets=gaze_real_150118,gaze_real_180220 --saveas=only_real --log=only_real --test=True
python train_gazenet.py --datasets=gaze_real_150118,gaze_real_180220 --saveas=only_real_rn50 --log=only_real_rn50 --test=True --feature_extractor=resnet50

# Only Synthetic Data
python train_gazenet.py --datasets=gaze_fake_100118 --saveas=only_fake --log=only_fake --test=True
python train_gazenet.py --datasets=gaze_fake_100118 --saveas=only_fake_rn50 --log=only_fake_rn50 --test=True --feature_extractor=resnet50

# All Data
python train_gazenet.py --datasets=gaze_real_150118,gaze_real_180220,gaze_fake_100118 --saveas=all --log=all --test=True
python train_gazenet.py --datasets=gaze_real_150118,gaze_real_180220,gaze_fake_100118 --saveas=all_rn50 --log=all_rn50 --test=True --feature_extractor=resnet50
