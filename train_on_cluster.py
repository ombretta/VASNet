import os
import math 

text = "#!/bin/sh\n\
#SBATCH --partition=general\n\
#SBATCH --qos=long\n\
#SBATCH --time=6:00:00\n\
#SBATCH --ntasks=1\n\
#SBATCH --cpus-per-task=2\n\
#SBATCH --mem=4000\n\
#SBATCH --gres=gpu:1\n\
module use /opt/insy/modulefiles\n\
module load cuda/10.0 cudnn/10.0-7.6.0.64\n\
srun python main.py --train "

features_type = "google" #"i3d"

if features_type == "i3d":
    
    text += " --datasets datasets/datasets_list.txt --output-dir=i3d_features"
else: text += " --output-dir=google_features"
            
filename = "VASNet_" + features_type + ".sbatch"

with open(filename, "w") as file:
    file.write(text)

os.system("sbatch " + filename)
