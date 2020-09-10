import os

dirs = [d for d in os.listdir(".") if os.path.exists(d+"/runs")]

for d in dirs:
    if not os.path.exists(runs+d): os.mkdir(runs+d)
    if not os.path.exists(runs+d+"/runs"): os.mkdir(runs+d+"/runs")
    for f in os.listdir(d+"/runs"):
        text = "cp -r "+d+"/runs/"+f+" "+runs+d+"/runs/"+f
        print(text)
        os.system(text)
