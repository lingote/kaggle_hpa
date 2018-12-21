import os
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split

labels = {
'0': 'Nucleoplasm', '1': 'Nuclear membrane', '2': 'Nucleoli', '3': 'Nucleoli fibrillar center',
'4': 'Nuclear speckles', '5': 'Nuclear bodies', '6': 'Endoplasmic reticulum', '7': 'Golgi apparatus',
'8': 'Peroxisomes', '9': 'Endosomes', '10': 'Lysosomes', '11': 'Intermediate filaments',
'12': 'Actin filaments', '13': 'Focal adhesion sites', '14': 'Microtubules', '15': 'Microtubule ends',
'16': 'Cytokinetic bridge', '17': 'Mitotic spindle', '18': 'Microtubule organizing center', '19': 'Centrosome',
'20': 'Lipid droplets', '21': 'Plasma membrane', '22': 'Cell junctions', '23': 'Mitochondria', '24': 'Aggresome',
'25': 'Cytosol', '26': 'Cytoplasmic bodies', '27': 'Rods & rings',
}

train_labels = pd.read_csv('data/train.csv')
train_labels['Target_array'] = train_labels['Target'].apply(lambda x: x.split())
label_freq = {labels[key]: train_labels['Target_array'].apply(lambda x: key in x).sum() for key in labels.keys()}
label_freq = pd.DataFrame.from_dict(label_freq, orient='index')


def get_datalist(mode='train'):
    train_labels = pd.read_csv(f'data/{mode}.csv')
    one_hot_labels = np.zeros((train_labels.shape[0], len(labels)))
    for row in train_labels.iterrows():
        one_hot_labels[row[0], [int(j) for j in row[1]['Target'].split()]] = 1
    return pd.DataFrame(np.column_stack((train_labels['Id'].values, one_hot_labels)),
                       columns=np.concatenate((['Id'], list(labels.values()))))

#res = get_data()
#print(res)


#plt.figure()
#bp = sns.barplot(y=label_freq.iloc[:,0], x=label_freq.index.values)# , order=label_freq.iloc[:,0].values.sort())
#bp.set_xticklabels(bp.get_xticklabels(), rotation=90)
#plt.show()

trainzip = zipfile.ZipFile('data/train.zip')
#imlist = trainzip.filelist
## Example image
#im1 = trainzip.open(imlist[0])
#im1 = Image.open(im1)
#im1_arr = np.array(im1)

def get_img_array(myzipfile, imgid, shape=(299,299)):
    """
    Reads image from zip file and converts it
    to numpy array
    Args:
        myzipfile: ZipFile object
        imgid (str): Image ID
    """
    img_arr = np.zeros(shape=(512, 512, 3), dtype=np.float32)
    img_green = Image.open(myzipfile.open(f'{imgid}_green.png'))
    img_blue = Image.open(myzipfile.open(f'{imgid}_blue.png'))
    img_red = Image.open(myzipfile.open(f'{imgid}_red.png'))
    img_yellow = Image.open(myzipfile.open(f'{imgid}_yellow.png'))
    img_arr[:,:,0] = np.divide(np.array(img_green), 255)
    img_arr[:,:,1] = np.divide(np.array(img_blue), 255)/2 + np.divide(np.array(img_yellow), 255)/2
    img_arr[:,:,2] = np.divide(np.array(img_red), 255)/2 + np.divide(np.array(img_red), 255)/2
    img_arr = cv2.resize(img_arr, shape)
    return img_arr


#res = get_img_array(trainzip, train_labels.loc[0, 'Id'])
#print(res)
#data_list = get_datalist()

#x_train, x_test, y_train, y_test = train_test_split(train_labels)

def prep_train_data(data_list, batch_size=5, shape=(299,299,3), augment=False):
    while True:
        random_indexes = np.random.choice(len(data_list), size=batch_size)
        batch_img = np.zeros(shape+(batch_size,))
        batch_labels = np.zeros((len(data_list), 28))
        for pos, i in enumerate(random_indexes):
            img = get_img_array(trainzip, data_list.iloc[i,0], (shape[0], shape[1]))
            batch_img[pos] = img
            batch_labels[pos] = data_list.iloc[i]
            yield batch_img, batch_labels