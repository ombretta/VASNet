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
srun python main.py --train "

features_type = "google" #"i3d"

if features_type == "i3d":
    
    text += " --datasets datasets/datasets_list.txt --output-dir=i3d_features"
else: text += " --output-dir=google_features"

learning_rate = [0.005, 0.0005, 0.00005]
weight_decay = [0.001, 0.0001, 0.00001]

for lr in learning_rate:
    for l2_req in weight_decay:
        full_text = text + "--lr=" + str(lr) + "--l2_req=" + str(l2_req)
            
        filename = "VASNet_" + features_type + "_lr" + str(lr) + "_l2req" + str(l2_req) +".sbatch"
        
        with open(filename, "w") as file:
            file.write(full_text)
        
        os.system("sbatch " + filename)
