PNEUMONIA Detection


Steps
    1. Import Libraries
    2. Loading the Dataset
    3. Data Visualization and Preprocessing
    4. Data Augmentation adn Resizing Images
    5. CNN Architecture
    6. Incremental Unfreezing and Fine- Tuning
    7. Evaluating the MOdel
    8. Flask Web Application




STEP 1 
```python  
import os, shutil
import random
import numpy as np
import pandas as pd
import cv2
import skimage
import skimage.segmentation
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline
plt.style.use("ggplot")
```


STEP 2
Loading teh dataset

STEP 3 
Data Visualization for PNEUMONIA  as well as NORMAL
we notice that our dataser is imbalance

STEP 4
Data Augmentation adn Resizing Images
making image data generator 
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D,Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
```