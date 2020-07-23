import os
import time
import psutil
import h5py
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

import sys
sys.path.append("/Users/ombretta/Documents/Code/i3d_breakfast/src/")

from i3dpt import I3D

def i3d_features(sample, model, device):
    sample = np.expand_dims(sample, axis=0).transpose(0, 4, 1, 2, 3)
    with torch.no_grad():
        sample_var = torch.autograd.Variable(torch.from_numpy(sample).to(device))
        _, mixed_5c = model.extract(sample_var)
        out_tensor = mixed_5c.data.cpu()
    return out_tensor.numpy()


def get_features(sample, model, device):
    
    timesteps = sample.shape[0]
    all_features = np.zeros([math.ceil(timesteps/8), 1024])
    i = 0
    
    i3d_input_interval = 30 # 16fps*30s = 30 s at a time
    # Sample features for 30 seconds at a time 
    while i < timesteps:
        features = torch.from_numpy(i3d_features(sample[i:i+8*2*i3d_input_interval], model, device))
        print("features", features.shape)
        features = F.adaptive_avg_pool3d(features, (None, 1, 1))
        features = features.squeeze(3).squeeze(3).squeeze(0)
        features = features.permute(1,0)
        all_features[round(i/8):round(i/8)+features.shape[0]] = features.numpy()
        i += 8*2*i3d_input_interval
    return all_features

    
def show_frame(frame):
    plt.figure()
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    
def get_video_properties(cap):
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    return frameCount, frameWidth, frameHeight, fps


def main():
    
    start_time = time.time()
    
    dataset_name = "SumMe"
   
    # Load pretrained i3d model 
    root = "../kinetics_i3d_pytorch/"
    rgb_pt_checkpoint = root+'model/model_rgb.pth'
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
    i3d_rgb = I3D(num_classes=400, modality='rgb')
    i3d_rgb.eval()
    i3d_rgb.load_state_dict(torch.load(rgb_pt_checkpoint))
    i3d_rgb.to(device)
    
    h5_features_file = "./"+dataset_name+"_raw_30s_.hdf5"
    raw_h5_features_file = "./"+dataset_name+"_raw_30s_.hdf5"
    
    # Select video from path 
    videos_path = "./"+dataset_name+"/videos/"
    
    scaled_frameHeight = 224
    
    counter = 0
    with h5py.File(h5_features_file, 'w')  as f1, h5py.File(raw_h5_features_file, 'w') as f2:
        for file in [f for f in os.listdir(videos_path) if "webm" in f][:4]:
            
            print(videos_path+file)
            counter += 1
            
            cap = cv2.VideoCapture(videos_path+file)
            frameCount, frameWidth, frameHeight, fps = get_video_properties(cap)
            
            print("Original", frameCount, frameWidth, frameHeight, fps)
            
            scaled_frameWidth = round(224/frameHeight*frameWidth)
            if scaled_frameWidth%2 == 1: scaled_frameWidth -= 1
            
            print(scaled_frameWidth, 224, videos_path+file.replace(" ", "\ "), scaled_frameWidth)
            
            # Create temp downsampled version with ffmpeg: scale=x:224
            i3d_fps = 16 
            
            video_name = file.split(".")[0]
            video = "./tmp_"+dataset_name+str(counter)+"_"+video_name+".mp4" 
            os.system("ffmpeg -i '{0}' -vf \"scale={3}:224,fps={2}\" '{1}'".format(videos_path+file, video, i3d_fps, scaled_frameWidth)) 
            
            # Read video and properties
            cap = cv2.VideoCapture(video)
            frameCount, frameWidth, frameHeight, fps = get_video_properties(cap)
            
            
            # os.remove(video) # Remove tmp.mp4
            print("Downsampled", frameCount, frameWidth, frameHeight, fps)
            
            # Read all video frames 
            buf = np.empty((frameCount, 224, 224, 3), np.dtype('float32'))
            
            fc = 0
            ret = True
            
            offset = round((scaled_frameWidth-scaled_frameHeight)/2)
            
            while ((fc < frameCount)  and ret):
                ret, frame_bgr = cap.read()
                frame_rgb = frame_bgr[:, :, [2, 1, 0]] # Swap bgr to rgb
                buf[fc] = frame_rgb[:,offset:offset+224,:] # crop center of the frame to have input 224x224
                fc += 1
            
            cap.release()
            
            # "Normalize" before extracting the features, to have values from -1 to 1
            norm_buf = (buf - 127.5)/127.5 # same as original code
            
            # Save video raw features in h5 file
            f2.create_dataset(video_name, data=norm_buf, compression="gzip", compression_opts=9)
            
            # Extract features 
            # i3d_features = get_features(norm_buf, i3d_rgb, device)
            
            # print("i3d_features", i3d_features.shape)
            
            # Save feature matrix in h5 file
            # f1.create_dataset(video_name, data=i3d_features, compression="gzip", compression_opts=9)
                        
    print(counter)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    # Check memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
    print('\nmemory use:', memoryUse, "GB")


if __name__== "__main__":
    main()