import os
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from PIL import Image

labels = {
'0': 'Nucleoplasm',
'1': 'Nuclear membrane',
'2': 'Nucleoli',
'3': 'Nucleoli fibrillar center',
'4': 'Nuclear speckles',
'5': 'Nuclear bodies',
'6': 'Endoplasmic reticulum',
'7': 'Golgi apparatus',
'8': 'Peroxisomes',
'9': 'Endosomes',
'10': 'Lysosomes',
'11': 'Intermediate filaments',
'12': 'Actin filaments',
'13': 'Focal adhesion sites',
'14': 'Microtubules',
'15': 'Microtubule ends',
'16': 'Cytokinetic bridge',
'17': 'Mitotic spindle',
'18': 'Microtubule organizing center',
'19': 'Centrosome',
'20': 'Lipid droplets',
'21': 'Plasma membrane',
'22': 'Cell junctions',
'23': 'Mitochondria',
'24': 'Aggresome',
'25': 'Cytosol',
'26': 'Cytoplasmic bodies',
'27': 'Rods & rings',
}

train_labels = pd.read_csv('data/train.csv')
train_labels['Target_array'] = train_labels['Target'].apply(lambda x: x.split())
label_freq = {labels[key]: train_labels['Target_array'].apply(lambda x: key in x).sum() for key in labels.keys()}
label_freq = pd.DataFrame.from_dict(label_freq, orient='index')
#plt.figure()
#bp = sns.barplot(y=label_freq.iloc[:,0], x=label_freq.index.values)# , order=label_freq.iloc[:,0].values.sort())
#bp.set_xticklabels(bp.get_xticklabels(), rotation=90)
#plt.show()

trainzip = zipfile.ZipFile('data/train.zip')
imlist = trainzip.filelist
# Example image
im1 = trainzip.open(imlist[0])
im1 = Image.open(im1)
im1_arr = np.array(im1)

def get_img_array(myzipfile, imgid):
    img_arr = np.zeros(shape=(4, 512, 512), dtype=np.float32)
    img_green = Image.open(myzipfile.open(f'{imgid}_green.png'))
    img_blue = Image.open(myzipfile.open(f'{imgid}_blue.png'))
    img_red = Image.open(myzipfile.open(f'{imgid}_red.png'))
    img_yellow = Image.open(myzipfile.open(f'{imgid}_yellow.png'))
    img_arr[0, :] = np.divide(np.array(img_green), 255)
    img_arr[1, :] = np.divide(np.array(img_blue), 255)
    img_arr[2, :] = np.divide(np.array(img_red), 255)
    img_arr[3, :] = np.divide(np.array(img_yellow), 255)
    return img_arr


res = get_img_array(trainzip, train_labels.loc[0, 'Id'])
print(res)