#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 14:39:56 2020

@author: ombretta
"""

import h5py 
import os

temp_results = "temp_results/"

for res_dir in [f for f in os.listdir(temp_results) if os.path.isdir(temp_results+f)]:
    print(res_dir)
        
    for epoch in os.listdir(temp_results+res_dir):
        print(epoch)
        temp_res = h5py.File(temp_results+res_dir+"/"+epoch, "r")
        
        print(temp_res)
        model_name = list(temp_res.keys())[0]
        print(model_name)
        
        for video in temp_res[model_name].keys():
            print(video)
            print(temp_res[model_name][video].keys())
            # for attr in temp_res[model_name][video].keys():
            #     print(temp_res[model_name][video][attr])
            
            print(temp_res[model_name][video]["machine_summary"][...][:30])
            print(temp_res[model_name][video]["machine_summary"][...][-30:])
            print(temp_res[model_name][video]["machine_summary"][...].sum())
            print(temp_res[model_name][video]["machine_summary"][...].shape)
            print(temp_res[model_name][video]["machine_summary"][...].sum()/temp_res[model_name][video]["machine_summary"][...].shape)
            print(temp_res[model_name][video]["score"][...])
    print()
    
    
    
    
    #%%
    
now = time.time()
a = f['cam01_P13_friedegg'][...]
check_time(now)

now = time.time()
b = f['cam01_P13_friedegg'][...][:,:8,:,:]
check_time(now)

