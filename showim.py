import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import zipfile

trainzip = zipfile.ZipFile('data/train.zip')

imglist = trainzip.filelist

# Example show random pick
rndfile = np.random.choice(trainzip.filelist).filename

img1 = Image(trainzip.open(rndfile))
color_maps = ['Greens', 'Oranges', 'Blues', 'Reds']
plt.imshow(np.array(img1), cmap='Orange')