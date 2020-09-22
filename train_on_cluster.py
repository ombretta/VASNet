import os
import math 

text = "#!/bin/sh\n\
#SBATCH --partition=general\n\
#SBATCH --qos=long\n\
#SBATCH --time=24:00:00\n\
#SBATCH --ntasks=1\n\
#SBATCH --cpus-per-task=2\n\
#SBATCH --mem=16000\n\
#SBATCH --gres=gpu:1\n\
module use /opt/insy/modulefiles\n\
module load cuda/10.0 cudnn/10.0-7.6.0.64\n\
srun python main.py "

train = False
if train: text += "--train "

features_type = "google"

ten_seconds_features = False
three_seconds_features = False
finetune = True
store_intermediate_results = True
backbone = "I3D_afterMaxPool3d" #"I3D"
fps = 2

if features_type == "i3d":
    if finetune and fps==16:
        text += " --finetune --datasets datasets/raw_datasets_list.txt --output-dir=i3d_features_30s_finetuned"
    elif three_seconds_features:
        text += " --datasets datasets/datasets_list3.txt --output-dir=i3d_features_3s"
    elif ten_seconds_features:
        text += " --datasets datasets/datasets_list2.txt --output-dir=i3d_features_10s"
    elif fps != 16 and finetune:
        text += " --finetune --datasets datasets/i3d_"+str(fps)+"fps_afterMaxPool3d_datasets_list.txt --output-dir=i3d_features_30s_"+str(fps)+"fps_finetuned"
    elif fps != 16:
        text += " --datasets datasets/i3d_"+str(fps)+"fps_mixed5c_datasets_list.txt --output-dir=i3d_features_30s_"+str(fps)+"fps"
    else:
        text += " --datasets datasets/datasets_list.txt --output-dir=i3d_features"

else: text += " --output-dir=google_features"

learning_rate =[0.00005, 0.0005, 0.005, 0.05]
weight_decay = [0.01, 0.001, 0.0001, 0.00001, 0.000001] 
epochs_max = 500
coeffs = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5] # coeff for the stochastic regularization term

for lr in learning_rate:
    for l2_req in weight_decay:
        for coeff in coeffs:
        
            name_extension = "_lr" + str(lr) + "_l2req" + str(l2_req)
            if coeff>=0: name_extension = name_extension + "_regcoeff" + str(coeff)

            full_text = text + name_extension + " --lr=" + str(lr) + \
                " --l2_req=" + str(l2_req) + " --epochs_max=" + str(epochs_max)
            
            if coeff>0: full_text = full_text + " --coeff=" + str(coeff)
            if backbone != "I3D" and features_type!="google": full_text = full_text + " --backbone="+backbone+" "
                
            filename = "VASNet_" + features_type + "_lr" + str(lr) + "_l2req" + str(l2_req) + "_regcoeff" + str(coeff) + ".sbatch"
            
            print(full_text)
     
            with open(filename, "w") as file:
                file.write(full_text)
            
            os.system("sbatch " + filename)
