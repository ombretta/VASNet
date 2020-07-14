import os
import math 

text = "#!/bin/sh\n\
#SBATCH --partition=general\n\
#SBATCH --qos=short\n\
#SBATCH --time=2:30:00\n\
#SBATCH --ntasks=1\n\
#SBATCH --cpus-per-task=2\n\
#SBATCH --mem=2000\n\
#SBATCH --gres=gpu:1\n\
module use /opt/insy/modulefiles\n\
module load cuda/10.0 cudnn/10.0-7.6.0.64\n\
srun python main.py "

train = True
if train: text += "--train "

features_type = "i3d" #"google"

ten_seconds_features = False
three_seconds_features = False

if features_type == "i3d":
    if three_seconds_features:
        text += " --datasets datasets/datasets_list3.txt --output-dir=i3d_features_3s"
    elif ten_seconds_features:
        text += " --datasets datasets/datasets_list2.txt --output-dir=i3d_features_10s"
    else:
        text += " --datasets datasets/datasets_list.txt --output-dir=i3d_features"
else: text += " --output-dir=google_features"

learning_rate = [0.00005, 0.005, 0.0005, 0.00005]
weight_decay = [0.01, 0.001, 0.0001, 0.000001] #[0.001, 0.0001, 0.00001, 0.000001]
epochs_max = 300

for lr in learning_rate:
    for l2_req in weight_decay:
        
        name_extension = "_lr" + str(lr) + "_l2req" + str(l2_req)
        full_text = text + name_extension + " --lr=" + str(lr) + \
            " --l2_req=" + str(l2_req) + " --epochs_max=" + str(epochs_max)
            
        filename = "VASNet_" + features_type + "_lr" + str(lr) + "_l2req" + str(l2_req) +".sbatch"
        
        print(full_text)
 
        with open(filename, "w") as file:
            file.write(full_text)
        
        os.system("sbatch " + filename)
