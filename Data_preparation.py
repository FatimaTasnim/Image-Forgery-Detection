import os
import glob
import shutil
import numpy as np
import pandas as pd
from PIL import Image

src_path = "C:/Users/BS304/Documents/CodeFiles/Forged_Image_Detection/Image_Forged/benchmark_data/VCL/CoMoFoD_small_v2"
dst_path = "C:/Users/BS304/Documents/CodeFiles/Forged_Image_Detection/Image_Forged/benchmark_data/SelectedImages/"

## Forged files are renamed with _copy name extention to get the same format as previous dataset and moved to the source folder
## the fromat of previous dataset was original name = name and forged name = name_copy
files = glob.glob("C:/Users/BS304/Documents/CodeFiles/Forged_Image_Detection/Image_Forged/benchmark_data/VCL/CoMoFoD_small_v2/*_F*")
print(len(files))
i = 1
for f in files:
    
    name = os.path.split(f)
    name = name[1]
    old_name =  src_path + "/" + name
    name = name.replace(".png", '')
        
    name = name + "_copy.png"
    new_name = src_path + "/" + name
    os.rename(old_name, new_name) 
    dest = shutil.move(new_name, dst_path")  
    i += 1
print(i , " files replaced and moved to the main folder")

## Original files are renamed replacing (O, F) to get the same format as previous dataset and moved to the source folder
files = glob.glob("C:/Users/BS304/Documents/CodeFiles/Forged_Image_Detection/Image_Forged/benchmark_data/VCL/CoMoFoD_small_v2/*_O*")
print(len(files))
i = 1
for f in files:
    
    name = os.path.split(f)
    name = name[1]
    old_name =  src_path + "/" + name
    name = name.replace("O", "F")
    new_name = src_path + "/" + name
    os.rename(old_name, new_name) 
    dest = shutil.move(new_name, dst_path)  
    i += 1
print(i , " files replaced and moved to the main folder")

# Image Labeling and saving to the csv file
# label format: 0 -> Original, 1-> Forged

files = glob.glob("Image_Forged/benchmark_data/SelectedImages/*copy.png")
list = [[]]
i = 0
for f in files:
    x = os.path.split(f)
    y = x[1].replace('_copy','')
    list.append((x[1], 1))
    list.append((y, 0))
    i += 1
df = pd.DataFrame(list)
df.to_csv("classifications_v1.csv")
print(i, " images labeled successfully\n\n")
print("Labeled data: ", df.head())

### Resizing the images to (512 * 512) and save all image in PNG format
path = "Image_Forged/benchmark_data/SelectedImages/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((512,512), Image.ANTIALIAS)
            imResize.save(f + '.png', 'PNG', quality=90)

resize()