
import os,glob
import numpy as np
import os
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
from tensorflow.keras import layers as L
import glob
import pickle
import tensorflow as tf
import argparse
import tensorflow as tf

from radiomics import featureextractor 
import os

from tqdm import tqdm
for i in tqdm(range(5)):
    pass
import glob
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow_addons as tfa
#import cv2 as cv
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import SimpleITK as sitk
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121
import re
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import datetime
import keras
from tensorflow.keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,LeakyReLU,Add,Average,Lambda,MaxPool2D,Dropout,UpSampling2D,Concatenate,Multiply,GlobalAveragePooling2D,Dense,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import concatenate,Flatten,Layer,ReLU, MaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from numpy import loadtxt
import numpy as np
#from keras_cv.layers import RandomCutout
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.xception import preprocess_input
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from skimage import data, exposure
from skimage.transform import radon, rescale
from skimage.filters import roberts, sobel, scharr, prewitt
from tensorflow_addons.optimizers import AdamW
from skimage import feature
import os,glob
import numpy as np
import cv2 as cv

import glob
import pickle
import tensorflow as tf
import pickle
import argparse
import re
import datetime
from tensorflow.keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,LeakyReLU,Add,Average,Lambda,MaxPool2D,Dropout,UpSampling2D,Concatenate,Multiply,GlobalAveragePooling2D,Dense,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import concatenate,Flatten,ConvLSTM2D,LayerNormalization,GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
from sklearn.svm import LinearSVC
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as rp
from tensorflow.keras.applications.xception import preprocess_input as xp
import sklearn.metrics as metrics
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os,glob
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
import keras
import numpy as np
import glob
import pickle
#import clahe
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix , classification_report
from matplotlib import pyplot as plt
import tensorflow as tf
import argparse
import re
import datetime
import pandas as pd
from sklearn.metrics import accuracy_score
from sympy.solvers import solve
from sympy import Symbol
import seaborn as sns
import numpy as np
from tensorflow.keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,LeakyReLU,Add,Average,Lambda,MaxPool2D,Dropout,UpSampling2D,Concatenate,Multiply,GlobalAveragePooling2D,Dense,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import concatenate,Flatten,Layer,ReLU, MaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from numpy import loadtxt
import pickle
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import f1_score
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from skimage import data, exposure
from skimage.transform import radon, rescale
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage import feature
import os,glob
import numpy as np
import cv2
import glob
import pickle
import tensorflow as tf
import pickle
import argparse
import re
import datetime
from tensorflow.keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,LeakyReLU,Add,Average,Lambda,MaxPool2D,Dropout,UpSampling2D,Concatenate,Multiply,GlobalAveragePooling2D,Dense,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import concatenate,Flatten,ConvLSTM2D,LayerNormalization,GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
import tensorflow.keras.backend as K
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from skimage import data, exposure
from tensorflow.keras import layers as L
from PIL import Image
from numpy import asarray
from sklearn.utils import shuffle
import os
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf   
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from tensorflow.keras.metrics import Recall, Precision
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from tensorflow.keras.metrics import Recall, Precision
from skimage import data, exposure
from tensorflow.keras.layers import Layer as L
from PIL import Image
from numpy import asarray
from sklearn.utils import shuffle
import os
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import pandas as pd

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf

import tensorflow as tf
from tensorflow.keras import layers as L
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.models import Model
from vit_keras import vit, visualize
#import tensorflow_probability as tfp
glcm_feature_list=["Autocorrelation","ClusterProminence","ClusterShade","ClusterTendency","Contrast","Correlation","DifferenceAverage",
                      "DifferenceEntropy","DifferenceVariance","Id","Idm","Idmn","Idn","Imc1","Imc2","InverseVariance","JointAverage",
                      "JointEnergy","JointEntropy","MCC","MaximumProbability","SumAverage","SumEntropy","SumSquares"]

glrlm_feature_list=["GrayLevelNonUniformity","GrayLevelNonUniformityNormalized","GrayLevelVariance","HighGrayLevelRunEmphasis",
                       "LongRunEmphasis","LongRunHighGrayLevelEmphasis", "LongRunLowGrayLevelEmphasis","LowGrayLevelRunEmphasis",
                       "RunEntropy","RunLengthNonUniformity","RunLengthNonUniformityNormalized","RunPercentage","RunVariance",
                       "ShortRunEmphasis","ShortRunHighGrayLevelEmphasis","ShortRunLowGrayLevelEmphasis"]

gldm_feature_list = ["DependenceEntropy","DependenceNonUniformity","DependenceNonUniformityNormalized","DependenceVariance",
                        "GrayLevelNonUniformity","GrayLevelVariance","HighGrayLevelEmphasis","LargeDependenceEmphasis","LargeDependenceHighGrayLevelEmphasis",
                        "LargeDependenceLowGrayLevelEmphasis","LowGrayLevelEmphasis","SmallDependenceEmphasis","SmallDependenceHighGrayLevelEmphasis",
                        "SmallDependenceLowGrayLevelEmphasis"]

glszm_feature_list=["GrayLevelNonUniformity","GrayLevelNonUniformityNormalized","GrayLevelVariance","HighGrayLevelZoneEmphasis",
                       "LargeAreaEmphasis","LargeAreaHighGrayLevelEmphasis","LargeAreaLowGrayLevelEmphasis","LowGrayLevelZoneEmphasis",
                       "SizeZoneNonUniformity","SizeZoneNonUniformityNormalized","SmallAreaEmphasis","SmallAreaHighGrayLevelEmphasis",
                       "SmallAreaLowGrayLevelEmphasis","ZoneEntropy","ZonePercentage","ZoneVariance"]
    
train_my_dict={}
for feature_name in glcm_feature_list:
    train_my_dict[feature_name+"glcm"]=[]

for feature_name in glrlm_feature_list:
    train_my_dict[feature_name+"glrlm"]=[]    
    
for feature_name in gldm_feature_list:
    train_my_dict[feature_name+"gldm"]=[]   
    
for feature_name in glszm_feature_list:
    train_my_dict[feature_name+"glszm"]=[] 
val_my_dict={}
for feature_name in glcm_feature_list:
    val_my_dict[feature_name+"glcm"]=[]

for feature_name in glrlm_feature_list:
    val_my_dict[feature_name+"glrlm"]=[]    
    
for feature_name in gldm_feature_list:
    val_my_dict[feature_name+"gldm"]=[]   
    
for feature_name in glszm_feature_list:
    val_my_dict[feature_name+"glszm"]=[] 

#inserting radiomic features in dictionary for validation data
def val_radiomic_features(data):
        #GLCM FEATURES
    glcm_features = {
        "Autocorrelation": data["original_glcm_Autocorrelation"],
        "ClusterProminence": data["original_glcm_ClusterProminence"],
        "ClusterShade": data["original_glcm_ClusterShade"],
        "ClusterTendency": data["original_glcm_ClusterTendency"],
        "Contrast": data["original_glcm_Contrast"],
        "Correlation": data["original_glcm_Correlation"],
        "DifferenceAverage": data["original_glcm_DifferenceAverage"],
        "DifferenceEntropy": data["original_glcm_DifferenceEntropy"],
        "DifferenceVariance": data["original_glcm_DifferenceVariance"],
        "Id": data["original_glcm_Id"],
        "Idm": data["original_glcm_Idm"],
        "Idmn": data["original_glcm_Idmn"],
        "Idn": data["original_glcm_Idn"],
        "Imc1": data["original_glcm_Imc1"],
        "Imc2": data["original_glcm_Imc2"],
        "InverseVariance": data["original_glcm_InverseVariance"],
        "JointAverage": data["original_glcm_JointAverage"],
        "JointEnergy": data["original_glcm_JointEnergy"],
        "JointEntropy": data["original_glcm_JointEntropy"],
        "MCC": data["original_glcm_MCC"],
        "MaximumProbability": data["original_glcm_MaximumProbability"],
        "SumAverage": data["original_glcm_SumAverage"],
        "SumEntropy": data["original_glcm_SumEntropy"],
        "SumSquares": data["original_glcm_SumSquares"]
    }
    for feature_name, feature_value in glcm_features.items():
        val_my_dict[feature_name+"glcm"].append(feature_value)
       
        
    #GLRLM FEATURES
    glrlm_features = {
        "GrayLevelNonUniformity": data["original_glrlm_GrayLevelNonUniformity"],
        "GrayLevelNonUniformityNormalized": data["original_glrlm_GrayLevelNonUniformityNormalized"],
        "GrayLevelVariance": data["original_glrlm_GrayLevelVariance"],
        "HighGrayLevelRunEmphasis": data["original_glrlm_HighGrayLevelRunEmphasis"],
        "LongRunEmphasis": data["original_glrlm_LongRunEmphasis"],
        "LongRunHighGrayLevelEmphasis": data["original_glrlm_LongRunHighGrayLevelEmphasis"],
        "LongRunLowGrayLevelEmphasis": data["original_glrlm_LongRunLowGrayLevelEmphasis"],
        "LowGrayLevelRunEmphasis": data["original_glrlm_LowGrayLevelRunEmphasis"],
        "RunEntropy": data["original_glrlm_RunEntropy"],
        "RunLengthNonUniformity": data["original_glrlm_RunLengthNonUniformity"],
        "RunLengthNonUniformityNormalized": data["original_glrlm_RunLengthNonUniformityNormalized"],
        "RunPercentage": data["original_glrlm_RunPercentage"],
        "RunVariance": data["original_glrlm_RunVariance"],
        "ShortRunEmphasis": data["original_glrlm_ShortRunEmphasis"],
        "ShortRunHighGrayLevelEmphasis": data["original_glrlm_ShortRunHighGrayLevelEmphasis"],
        "ShortRunLowGrayLevelEmphasis": data["original_glrlm_ShortRunLowGrayLevelEmphasis"]
    }
    for feature_name, feature_value in glrlm_features.items():
        val_my_dict[feature_name+"glrlm"].append(feature_value)
        
    gldm_features = {
        "DependenceEntropy": data["original_gldm_DependenceEntropy"],
        "DependenceNonUniformity": data["original_gldm_DependenceNonUniformity"],
        "DependenceNonUniformityNormalized": data["original_gldm_DependenceNonUniformityNormalized"],
        "DependenceVariance": data["original_gldm_DependenceVariance"],
        "GrayLevelNonUniformity": data["original_gldm_GrayLevelNonUniformity"],
        "GrayLevelVariance": data["original_gldm_GrayLevelVariance"],
        "HighGrayLevelEmphasis": data["original_gldm_HighGrayLevelEmphasis"],
        "LargeDependenceEmphasis": data["original_gldm_LargeDependenceEmphasis"],
        "LargeDependenceHighGrayLevelEmphasis": data["original_gldm_LargeDependenceHighGrayLevelEmphasis"],
        "LargeDependenceLowGrayLevelEmphasis": data["original_gldm_LargeDependenceLowGrayLevelEmphasis"],
        "LowGrayLevelEmphasis": data["original_gldm_LowGrayLevelEmphasis"],
        "SmallDependenceEmphasis": data["original_gldm_SmallDependenceEmphasis"],
        "SmallDependenceHighGrayLevelEmphasis": data["original_gldm_SmallDependenceHighGrayLevelEmphasis"],
        "SmallDependenceLowGrayLevelEmphasis": data["original_gldm_SmallDependenceLowGrayLevelEmphasis"]
    }
    for feature_name, feature_value in gldm_features.items():
        val_my_dict[feature_name+"gldm"].append(feature_value)
    
    #GLSZM FEATURES
    glszm_features = {
        "GrayLevelNonUniformity": data["original_glszm_GrayLevelNonUniformity"],
        "GrayLevelNonUniformityNormalized": data["original_glszm_GrayLevelNonUniformityNormalized"],
        "GrayLevelVariance": data["original_glszm_GrayLevelVariance"],
        "HighGrayLevelZoneEmphasis": data["original_glszm_HighGrayLevelZoneEmphasis"],
        "LargeAreaEmphasis": data["original_glszm_LargeAreaEmphasis"],
        "LargeAreaHighGrayLevelEmphasis": data["original_glszm_LargeAreaHighGrayLevelEmphasis"],
        "LargeAreaLowGrayLevelEmphasis": data["original_glszm_LargeAreaLowGrayLevelEmphasis"],
        "LowGrayLevelZoneEmphasis": data["original_glszm_LowGrayLevelZoneEmphasis"],
        "SizeZoneNonUniformity": data["original_glszm_SizeZoneNonUniformity"],
        "SizeZoneNonUniformityNormalized": data["original_glszm_SizeZoneNonUniformityNormalized"],
        "SmallAreaEmphasis": data["original_glszm_SmallAreaEmphasis"],
        "SmallAreaHighGrayLevelEmphasis": data["original_glszm_SmallAreaHighGrayLevelEmphasis"],
        "SmallAreaLowGrayLevelEmphasis": data["original_glszm_SmallAreaLowGrayLevelEmphasis"],
        "ZoneEntropy": data["original_glszm_ZoneEntropy"],
        "ZonePercentage": data["original_glszm_ZonePercentage"],
        "ZoneVariance": data["original_glszm_ZoneVariance"]
    }
    for feature_name, feature_value in glszm_features.items():
        val_my_dict[feature_name+"glszm"].append(feature_value)

#radiomic features dictionary for saving the radiomic features in dictionary
def train_radiomic_features(data):
        #GLCM FEATURES
    glcm_features = {
        "Autocorrelation": data["original_glcm_Autocorrelation"],
        "ClusterProminence": data["original_glcm_ClusterProminence"],
        "ClusterShade": data["original_glcm_ClusterShade"],
        "ClusterTendency": data["original_glcm_ClusterTendency"],
        "Contrast": data["original_glcm_Contrast"],
        "Correlation": data["original_glcm_Correlation"],
        "DifferenceAverage": data["original_glcm_DifferenceAverage"],
        "DifferenceEntropy": data["original_glcm_DifferenceEntropy"],
        "DifferenceVariance": data["original_glcm_DifferenceVariance"],
        "Id": data["original_glcm_Id"],
        "Idm": data["original_glcm_Idm"],
        "Idmn": data["original_glcm_Idmn"],
        "Idn": data["original_glcm_Idn"],
        "Imc1": data["original_glcm_Imc1"],
        "Imc2": data["original_glcm_Imc2"],
        "InverseVariance": data["original_glcm_InverseVariance"],
        "JointAverage": data["original_glcm_JointAverage"],
        "JointEnergy": data["original_glcm_JointEnergy"],
        "JointEntropy": data["original_glcm_JointEntropy"],
        "MCC": data["original_glcm_MCC"],
        "MaximumProbability": data["original_glcm_MaximumProbability"],
        "SumAverage": data["original_glcm_SumAverage"],
        "SumEntropy": data["original_glcm_SumEntropy"],
        "SumSquares": data["original_glcm_SumSquares"]
    }
    for feature_name, feature_value in glcm_features.items():
        train_my_dict[feature_name+"glcm"].append(feature_value)
       
        
    #GLRLM FEATURES
    glrlm_features = {
        "GrayLevelNonUniformity": data["original_glrlm_GrayLevelNonUniformity"],
        "GrayLevelNonUniformityNormalized": data["original_glrlm_GrayLevelNonUniformityNormalized"],
        "GrayLevelVariance": data["original_glrlm_GrayLevelVariance"],
        "HighGrayLevelRunEmphasis": data["original_glrlm_HighGrayLevelRunEmphasis"],
        "LongRunEmphasis": data["original_glrlm_LongRunEmphasis"],
        "LongRunHighGrayLevelEmphasis": data["original_glrlm_LongRunHighGrayLevelEmphasis"],
        "LongRunLowGrayLevelEmphasis": data["original_glrlm_LongRunLowGrayLevelEmphasis"],
        "LowGrayLevelRunEmphasis": data["original_glrlm_LowGrayLevelRunEmphasis"],
        "RunEntropy": data["original_glrlm_RunEntropy"],
        "RunLengthNonUniformity": data["original_glrlm_RunLengthNonUniformity"],
        "RunLengthNonUniformityNormalized": data["original_glrlm_RunLengthNonUniformityNormalized"],
        "RunPercentage": data["original_glrlm_RunPercentage"],
        "RunVariance": data["original_glrlm_RunVariance"],
        "ShortRunEmphasis": data["original_glrlm_ShortRunEmphasis"],
        "ShortRunHighGrayLevelEmphasis": data["original_glrlm_ShortRunHighGrayLevelEmphasis"],
        "ShortRunLowGrayLevelEmphasis": data["original_glrlm_ShortRunLowGrayLevelEmphasis"]
    }
    for feature_name, feature_value in glrlm_features.items():
        train_my_dict[feature_name+"glrlm"].append(feature_value)
        
    gldm_features = {
        "DependenceEntropy": data["original_gldm_DependenceEntropy"],
        "DependenceNonUniformity": data["original_gldm_DependenceNonUniformity"],
        "DependenceNonUniformityNormalized": data["original_gldm_DependenceNonUniformityNormalized"],
        "DependenceVariance": data["original_gldm_DependenceVariance"],
        "GrayLevelNonUniformity": data["original_gldm_GrayLevelNonUniformity"],
        "GrayLevelVariance": data["original_gldm_GrayLevelVariance"],
        "HighGrayLevelEmphasis": data["original_gldm_HighGrayLevelEmphasis"],
        "LargeDependenceEmphasis": data["original_gldm_LargeDependenceEmphasis"],
        "LargeDependenceHighGrayLevelEmphasis": data["original_gldm_LargeDependenceHighGrayLevelEmphasis"],
        "LargeDependenceLowGrayLevelEmphasis": data["original_gldm_LargeDependenceLowGrayLevelEmphasis"],
        "LowGrayLevelEmphasis": data["original_gldm_LowGrayLevelEmphasis"],
        "SmallDependenceEmphasis": data["original_gldm_SmallDependenceEmphasis"],
        "SmallDependenceHighGrayLevelEmphasis": data["original_gldm_SmallDependenceHighGrayLevelEmphasis"],
        "SmallDependenceLowGrayLevelEmphasis": data["original_gldm_SmallDependenceLowGrayLevelEmphasis"]
    }
    for feature_name, feature_value in gldm_features.items():
        train_my_dict[feature_name+"gldm"].append(feature_value)
    
    #GLSZM FEATURES
    glszm_features = {
        "GrayLevelNonUniformity": data["original_glszm_GrayLevelNonUniformity"],
        "GrayLevelNonUniformityNormalized": data["original_glszm_GrayLevelNonUniformityNormalized"],
        "GrayLevelVariance": data["original_glszm_GrayLevelVariance"],
        "HighGrayLevelZoneEmphasis": data["original_glszm_HighGrayLevelZoneEmphasis"],
        "LargeAreaEmphasis": data["original_glszm_LargeAreaEmphasis"],
        "LargeAreaHighGrayLevelEmphasis": data["original_glszm_LargeAreaHighGrayLevelEmphasis"],
        "LargeAreaLowGrayLevelEmphasis": data["original_glszm_LargeAreaLowGrayLevelEmphasis"],
        "LowGrayLevelZoneEmphasis": data["original_glszm_LowGrayLevelZoneEmphasis"],
        "SizeZoneNonUniformity": data["original_glszm_SizeZoneNonUniformity"],
        "SizeZoneNonUniformityNormalized": data["original_glszm_SizeZoneNonUniformityNormalized"],
        "SmallAreaEmphasis": data["original_glszm_SmallAreaEmphasis"],
        "SmallAreaHighGrayLevelEmphasis": data["original_glszm_SmallAreaHighGrayLevelEmphasis"],
        "SmallAreaLowGrayLevelEmphasis": data["original_glszm_SmallAreaLowGrayLevelEmphasis"],
        "ZoneEntropy": data["original_glszm_ZoneEntropy"],
        "ZonePercentage": data["original_glszm_ZonePercentage"],
        "ZoneVariance": data["original_glszm_ZoneVariance"]
    }
    for feature_name, feature_value in glszm_features.items():
        train_my_dict[feature_name+"glszm"].append(feature_value)

#getting ready the function for feature extraction of validation images
def val_my_masks(image,mask):
    # Assuming you have a numpy array of input images called 'image_array'
    try:
        # Convert numpy array to SimpleITK image
        sitk_image = sitk.GetImageFromArray(image)

        sitk_mask = sitk.GetImageFromArray(mask)
        # Initialize the feature extractor
        extractor = featureextractor.RadiomicsFeatureExtractor()


        #glszm, glcm
        # Enable the GLSZM feature class
        extractor.enableFeatureClassByName('glcm')

        # Specify the parameters for GLSZM feature extraction (if desired)
        # For example:
        extractor.settings['distances'] = [1, 2, 3]  # Specify the distances
        extractor.settings['force2D'] = True  # Set to True if input is 2D image, False for 3D

        # Extract the GLSZM features
        feature_vector = extractor.execute(sitk_image,sitk_mask)
        data=feature_vector
        val_radiomic_features(data)
    except:
        for feature_name in glcm_feature_list:
            val_my_dict[feature_name+"glcm"].append(0)

        for feature_name in glrlm_feature_list:
            val_my_dict[feature_name+"glrlm"].append(0)   

        for feature_name in gldm_feature_list:
            val_my_dict[feature_name+"gldm"].append(0)     

        for feature_name in glszm_feature_list:
            val_my_dict[feature_name+"glszm"].append(0) 

#getting ready the function for feature extraction of training images
def train_my_masks(image,mask):
    # Assuming you have a numpy array of input images called 'image_array'
    try:
        # Convert numpy array to SimpleITK image
        sitk_image = sitk.GetImageFromArray(image)

        sitk_mask = sitk.GetImageFromArray(mask)
        # Initialize the feature extractor
        extractor = featureextractor.RadiomicsFeatureExtractor()


        #glszm, glcm
        # Enable the GLSZM feature class
        extractor.enableFeatureClassByName('glcm')

        # Specify the parameters for GLSZM feature extraction (if desired)
        # For example:
        extractor.settings['distances'] = [1, 2, 3]  # Specify the distances
        extractor.settings['force2D'] = True  # Set to True if input is 2D image, False for 3D

        # Extract the GLSZM features
        feature_vector = extractor.execute(sitk_image,sitk_mask)
        data=feature_vector
        train_radiomic_features(data)
    except:
        for feature_name in glcm_feature_list:
            train_my_dict[feature_name+"glcm"].append(0)

        for feature_name in glrlm_feature_list:
            train_my_dict[feature_name+"glrlm"].append(0)   

        for feature_name in gldm_feature_list:
            train_my_dict[feature_name+"gldm"].append(0)     

        for feature_name in glszm_feature_list:
            train_my_dict[feature_name+"glszm"].append(0)  


    
batch_size = 16
num_epochs = 100
image_size = 256  # We'll resize input images to this size
patch_size = 16  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024] 
num_classes = 5
#applying watershed segmentation on training images to get the mask
def train_my_mask_and_radiomic(pic):
    img=pic.copy()
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv.watershed(img,markers)
    markers[markers != 2] = 0
    markers[markers == 2] = 1
    mask=markers.copy()
    image=img.copy()
    img=img/255.0
    mask=np.stack((mask,)*3,axis=-1)
    train_my_masks(image,mask)
    return img

#applying watershed segmentation on validation images to get the mask
def val_my_mask_and_radiomic(pic):
    img=pic.copy()
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv.watershed(img,markers)
    markers[markers != 2] = 0
    markers[markers == 2] = 1
    mask=markers.copy()
    image=img.copy()
    img=img/255.0
    mask=np.stack((mask,)*3,axis=-1)
    val_my_masks(image,mask)
    return img
#loading, resizing the validation images and also getting the radiomic data of the validation images
def val_no_data_augmentation(normal_files,covid_files,pneumonia_files,tb_files,cat_files):
    aug_normal=[]
    aug_covid=[]
    aug_pneumonia=[]
    aug_tb=[]
    aug_cat=[]
    for ele in tqdm(normal_files):
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
        
        pic = cv2.resize(x,(image_size,image_size),interpolation = cv2.INTER_CUBIC)
        pic=val_my_mask_and_radiomic(pic)
        aug_normal.append(pic)
    print("category 1 done")
    for ele in tqdm(covid_files):
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
        
        pic = cv2.resize(x,(image_size,image_size),interpolation = cv2.INTER_CUBIC)
        pic=val_my_mask_and_radiomic(pic)
        aug_covid.append(pic)
    print("category 2 done")
    for ele in tqdm(pneumonia_files):
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
      
        pic = cv2.resize(x,(image_size,image_size),interpolation = cv2.INTER_CUBIC)
        pic=val_my_mask_and_radiomic(pic)
        aug_pneumonia.append(pic)
    print("category 3 done")
    for ele in tqdm(tb_files):
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
        
        pic = cv2.resize(x,(image_size,image_size),interpolation = cv2.INTER_CUBIC)
        pic=val_my_mask_and_radiomic(pic)
        aug_tb.append(pic)  
    print("category 4 done")
    for ele in tqdm(cat_files):
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
        
        pic = cv2.resize(x,(image_size,image_size),interpolation = cv2.INTER_CUBIC)
        pic=val_my_mask_and_radiomic(pic)
        aug_cat.append(pic) 
    print("category 5 done")
    for i in range(len(aug_normal)):
        aug_normal[i]=aug_normal[i].reshape((image_size,image_size,3))
    
    for i in range(len(aug_covid)):
        aug_covid[i]=aug_covid[i].reshape((image_size,image_size,3))
    for i in range(len(aug_pneumonia)):
        aug_pneumonia[i]=aug_pneumonia[i].reshape((image_size,image_size,3))
    for i in range(len(aug_tb)):
        aug_tb[i]=aug_tb[i].reshape((image_size,image_size,3))    
    for i in range(len(aug_cat)):
        aug_cat[i]=aug_cat[i].reshape((image_size,image_size,3)) 

    return aug_normal,aug_covid,aug_pneumonia, aug_tb, aug_cat

#loading, resizing the validation images and also getting the radiomic data of the validation images
def train_no_data_augmentation(normal_files,covid_files,pneumonia_files,tb_files,cat_files): #Loading, resizing the image. Getting radiomic features of the image.
    aug_normal=[]
    aug_covid=[]
    aug_pneumonia=[]
    aug_tb=[]
    aug_cat=[]
    for ele in tqdm( normal_files):
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
        
        pic = cv2.resize(x,(image_size,image_size),interpolation = cv2.INTER_CUBIC)
        pic=train_my_mask_and_radiomic(pic)
        aug_normal.append(pic)
    print("category 1 done")
    for ele in tqdm(covid_files):
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
        
        pic = cv2.resize(x,(image_size,image_size),interpolation = cv2.INTER_CUBIC)
        pic=train_my_mask_and_radiomic(pic)
        aug_covid.append(pic)
    print("category 2 done")
    for ele in tqdm(pneumonia_files):
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
      
        pic = cv2.resize(x,(image_size,image_size),interpolation = cv2.INTER_CUBIC)
        pic=train_my_mask_and_radiomic(pic)
        aug_pneumonia.append(pic)
    print("category 3 done")
    for ele in tqdm(tb_files):
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
        
        pic = cv2.resize(x,(image_size,image_size),interpolation = cv2.INTER_CUBIC)
        pic=train_my_mask_and_radiomic(pic)
        aug_tb.append(pic)  
    print("category 4 done")
    for ele in tqdm(cat_files):
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
        
        pic = cv2.resize(x,(image_size,image_size),interpolation = cv2.INTER_CUBIC)
        pic=train_my_mask_and_radiomic(pic)
        aug_cat.append(pic) 
    print("category 5 done")
    for i in range(len(aug_normal)):
        aug_normal[i]=aug_normal[i].reshape((image_size,image_size,3))
    
    for i in range(len(aug_covid)):
        aug_covid[i]=aug_covid[i].reshape((image_size,image_size,3))
    for i in range(len(aug_pneumonia)):
        aug_pneumonia[i]=aug_pneumonia[i].reshape((image_size,image_size,3))
    for i in range(len(aug_tb)):
        aug_tb[i]=aug_tb[i].reshape((image_size,image_size,3))    
    for i in range(len(aug_cat)):
        aug_cat[i]=aug_cat[i].reshape((image_size,image_size,3)) 
    

    return aug_normal,aug_covid,aug_pneumonia, aug_tb, aug_cat
#function if anyone wants to apply data augmentation like rotation, zoom, vertical flip, etc
def my_augmenter(x):
    my_list=[]
    datagen=ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
    )
    counter=0
    for i in datagen.flow(x):
        if counter>=6:
            break
        #i=i/255.0
        img=i[0].astype(np.uint8).copy()
#         plt.imshow(img)
#         plt.show()
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        kernel = np.ones((3,3),np.uint8)
        opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
        # sure background area
        sure_bg = cv.dilate(opening,kernel,iterations=3)
        # Finding sure foreground area
        dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
        ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg,sure_fg)
        # Marker labelling
        ret, markers = cv.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0
        markers = cv.watershed(img,markers)
        markers[markers != 2] = 0
        markers[markers == 2] = 1
        mask=markers.copy()
        image=img.copy()
        img=img/255.0
        mask=np.stack((mask,)*3,axis=-1)
        train_my_masks(image,mask)
        my_list.append(img)
        counter+=1
    return my_list



    
#function if want to use normal augmentation with radiomics
def data_augmentation(normal_files,covid_files,pneumonia_files,tb_files,cat_files):
    aug_normal=[]
    aug_covid=[]
    thresh_hold=7
    aug_pneumonia=[]
    aug_tb=[]
    aug_cat=[]
    
    #x = tf.keras.preprocessing.image.load_img("/content/IM-0001-0001.jpeg")
    
   
    #category 1
    counter=0
    for location in tqdm(normal_files):
        counter=0
        x = Image.open(location)
        x = asarray(x)
        x=cv.resize(x,(image_size,image_size),interpolation = cv.INTER_CUBIC)
        x=x.reshape((1,)+x.shape)
        aug_normal=my_augmenter(x)
        #x=x/255.0
        
    
    #category 2
    for location in tqdm(tb_files):
        counter=0

        x = Image.open(location)
        x = asarray(x)
       
        x=cv2.resize(x,(image_size,image_size),interpolation = cv2.INTER_CUBIC)
        x=x/255.0
        
        x=x.reshape((1,)+x.shape)
        #x=x/255.0
        aug_tb=my_augmenter(x)
            
    #category 3
    counter=0
    for location in tqdm(covid_files):
        counter=0
        x = Image.open(location)
        x = asarray(x)
    
        x=cv2.resize(x,(image_size,image_size),interpolation = cv2.INTER_CUBIC)
        x=x/255.0

        #x=img_to_array(x)
        x=x.reshape((1,)+x.shape)
        #x=x/255.0
        aug_covid = my_augmenter(x)
        
    #category 4
    counter=0
    for location in tqdm(pneumonia_files):
        counter=0
        x = Image.open(location)
        x = asarray(x)
    

        x=cv2.resize(x,(image_size,image_size),interpolation = cv2.INTER_CUBIC)
        x=x/255.0

        #x=img_to_array(x)
        x=x.reshape((1,)+x.shape)
        #x=x/255.0
        aug_pneumonia = my_augmenter(x)
          
    
    #category 5
    counter=0
    for location in tqdm(cat_files):
        counter=0
        x = Image.open(location)
        x = asarray(x)
    

        x=cv2.resize(x,(image_size,image_size),interpolation = cv2.INTER_CUBIC)
        x=x/255.0
  
        #x=img_to_array(x)
        x=x.reshape((1,)+x.shape)
        #x=x/255.0
        aug_cat = my_augmenter(x)

    for ele in normal_files:
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
        
        pic = cv2.resize(x,(image_size,image_size),interpolation = cv2.INTER_CUBIC)
        pic=train_my_mask_and_radiomic(pic)
        aug_normal.append(pic)
    for ele in covid_files:
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
        
        pic = cv2.resize(x,(image_size,image_size),interpolation = cv2.INTER_CUBIC)
        pic=train_my_mask_and_radiomic(pic)
        
        
        aug_covid.append(pic)
    for ele in pneumonia_files:
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
      
        pic = cv2.resize(x,(image_size,image_size),interpolation = cv2.INTER_CUBIC)
        pic=train_my_mask_and_radiomic(pic)
        aug_pneumonia.append(pic)
    for ele in tb_files:
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
      
        pic = cv2.resize(x,(image_size,image_size),interpolation = cv2.INTER_CUBIC)
        pic=train_my_mask_and_radiomic(pic)
        aug_tb.append(pic)    
    for ele in cat_files:
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
      
        pic = cv2.resize(x,(image_size,image_size),interpolation = cv2.INTER_CUBIC)
        pic=train_my_mask_and_radiomic(pic)
        aug_cat.append(pic)  
    
    for i in range(len(aug_normal)):
        aug_normal[i]=aug_normal[i].reshape((image_size,image_size,3))
    
    for i in range(len(aug_covid)):
        aug_covid[i]=aug_covid[i].reshape((image_size,image_size,3))
    for i in range(len(aug_pneumonia)):
        aug_pneumonia[i]=aug_pneumonia[i].reshape((image_size,image_size,3))
    for i in range(len(aug_tb)):
        aug_tb[i]=aug_tb[i].reshape((image_size,image_size,3))
    for i in range(len(aug_cat)):
        aug_cat[i]=aug_cat[i].reshape((image_size,image_size,3))
    
    return aug_normal,aug_covid,aug_pneumonia,aug_tb,aug_cat

def box(lamda):
    IM_SIZE=128
    r_x=int(np.random.uniform(0, IM_SIZE))
    
    
    r_y=int(np.random.uniform(0, IM_SIZE))
    
    r_w=IM_SIZE*np.sqrt(1 - lamda)
    r_h=IM_SIZE*np.sqrt(1 - lamda)

    r_x=np.clip(r_x - r_w // 2, 0, IM_SIZE)
    r_y=np.clip(r_y-r_h//2, 0, IM_SIZE)

    x_b_r=np.clip(r_x+r_w//2,0, IM_SIZE)
    y_b_r=np.clip(r_y+r_h//2,0, IM_SIZE)

    r_w=y_b_r-r_y
    if(r_w==0):
        r_w=1
    r_h=y_b_r-r_y
    if(r_h==0):
        r_h=1
     
    return int(r_y),int(r_x),int(r_h),int(r_w)

#function for using advance augmentation, for future work to determine how cutmix advance augmentation works
def cutmix(image1,label1,images,labels):
    np.random.seed(None)
    index = np.random.permutation(len(images))
    lamda=stats.beta(0.4, 0.4).rvs()
    r_y,r_x,r_h,r_w=box(lamda)
    image2 = images[index[0]]
    label2 = labels[index[0]]
    crop2=tf.image.crop_to_bounding_box(image2,r_y,r_x,r_h,r_w)
    pad2=tf.image.pad_to_bounding_box(crop2,r_y,r_x,IM_SIZE,IM_SIZE)
    crop1=tf.image.crop_to_bounding_box(image1,r_y,r_x,r_h,r_w)
    pad1=tf.image.pad_to_bounding_box(crop1,r_y,r_x,IM_SIZE,IM_SIZE)
    image=image1-pad1+pad2
    lamda=1-(r_h*r_w)/(IM_SIZE*IM_SIZE)
    label=lamda*label1+(1-lamda)*label2
    return image,label
#function for one hot encoding labels
def on_hot_encode_labels(lables):
    aug_list=[]
    for i in range(len(lables)):
        if lables[i]==0:
            aug_list.append(0)
        elif lables[i]==1:
            aug_list.append(1)
        elif lables[i]==2:
            aug_list.append(2)
        elif lables[i]==3:
            aug_list.append(3)
        elif lables[i]==4:
            aug_list.append(4)
    return aug_list
import cv2
IM_SIZE=256
#fundtion to apply mixup advance data augmentation technique in future
def mixup(image1,label1,images,labels):
    index = np.random.permutation(len(images))
    image2 = images[index[0]]
    
    label2 = labels[index[0]]
    lamda=np.random.beta(0.4, 0.4)
    
    label_1=label1
    label_2=label2
    image=lamda*image1+(1-lamda)*image2
    label=lamda*label_1+(1-lamda)*label_2
    
    return image,label

#function to apply cutout advance data augmentation technique in the future
def cutout(images,labels, pad_size=16):
    cut_image=[]
    cut_labels=[]
    for index in tqdm(range(len(images))):
        img=images[index]
        h, w, c = img.shape
        mask = np.ones((h + pad_size*2, w + pad_size*2, c))
        y = np.random.randint(pad_size, h + pad_size)
        x = np.random.randint(pad_size, w + pad_size)
        y1 = np.clip(y - pad_size, 0, h + pad_size*2)
        y2 = np.clip(y + pad_size, 0, h + pad_size*2)
        x1 = np.clip(x - pad_size, 0, w + pad_size*2)
        x2 = np.clip(x + pad_size, 0, w + pad_size*2)
        mask[y1:y2, x1:x2, :] = 0
        img_cutout = img * mask[pad_size:pad_size+h, pad_size:pad_size+w, :]
        cut_image.append(img_cutout)
        cut_labels.append(labels[index])
    return cut_image,cut_labels

#function to do normal augmentation without radiomics
def random_augment(image1,label1,images,labels):
    datagen=ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
    )
    x=image1.reshape((1,)+image1.shape)
    my_images=[]
    my_labels=[]
    counter=0
    for i in datagen.flow(x):
        if counter==7:
            break
        my_image=i
        my_image=my_image.reshape((256,256,3))
        my_images.append(my_image)
        my_labels.append(label1)
        counter+=1
        
        
    
    return my_images,my_labels
        
#function to apply advance augmentation without radiomice features for future    
def advance_data_aug(images_list,images_labels,full_data,full_label,param=2):
    images_list=np.array(images_list)
    images_labels=np.array(images_labels)
    
    # create the original array
    arr = full_label

    # define the value of the element to delete
    value_to_delete = images_labels[0]
    index=[]
    for i in range(len(arr)):
        if np.array_equal(arr[i],value_to_delete):
            index.append(i)
    my_full_data=[]
    my_full_label=[]
    for i in range(len(full_label)):
        if i in index:
            continue
        else:
            my_full_data.append(full_data[i])
            my_full_label.append(full_label[i])
    full_data=my_full_data.copy()
    full_label=my_full_label.copy()
    full_data=np.array(full_data)
    full_label=np.array(full_label)
    aug_list=[]
    aug_labels=[]
    print("adding original images")
    for i in range(len(images_list)):
        aug_labels.append(images_labels[i])
        aug_list.append(images_list[i])
    print(np.array(aug_list).shape,np.array(aug_labels).shape)
#     print("cutmix")
#     for i in range(2):
#         for j in tqdm(range(len(images_list))):
#             new_image,new_label=cutmix(images_list[j],images_labels[j],full_data,full_label)
#             aug_labels.append(new_label)
#             aug_list.append(new_image)
#     print(np.array(aug_list).shape,np.array(aug_labels).shape)
#     print("mixup")
#     for i in range(2):
#         for j in tqdm(range(len(images_list))):
#             new_image,new_label=mixup(images_list[j],images_labels[j],full_data,full_label)
#             aug_labels.append(new_label)
#             aug_list.append(new_image)
#     print(np.array(aug_list).shape,np.array(aug_labels).shape)
    print("random augmentation")
    for j in tqdm(range(len(images_list))):
        new_image,new_label=random_augment(images_list[j],images_labels[j],full_data,full_label)
        for index in range(len(new_image)):
            aug_labels.append(new_label[index])
            aug_list.append(new_image[index])
#     print(np.array(aug_list).shape,np.array(aug_labels).shape)
#     print("cutout")
#     aug_list=np.array(aug_list)
#     aug_labels=np.array(aug_labels)
#     for i in range(2):
#         im,la=cutout(images_list,images_labels) 
#         aug_list = np.concatenate([aug_list, im])
#         aug_labels=np.concatenate([aug_labels,la])
        
    print(np.array(aug_list).shape,np.array(aug_labels).shape)
    return aug_list,aug_labels

accuracy_gamma=[]
precision_gamma=[]
recall_gamma=[]
fscore_gamma=[]
accuracy_rank=[]
precision_rank=[]
recall_rank=[]
fscore_rank=[]
accuracy_sugeno=[]
precision_sugeno=[]
recall_sugeno=[]
fscore_sugeno=[]
accuracy_weighted=[]
precision_weighted=[]
recall_weighted=[]
fscore_weighted=[]
    
def making_training_and_testing_data_train(full_data,full_label):
    return full_data,full_label

#function to generate plots (training and validation curve)  
def my_plots(folder_path,history,my_model):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    my_path="training and validation accuracy curve of "+my_model+".png"
    plt.savefig(folder_path+my_path)
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylim([0, 1])

    #plt.ylim([-3, 3])
    plt.yticks(np.arange(0, 1.1, 0.25))
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    my_path="training and validation loss curve of "+my_model+".png"
    plt.savefig(folder_path+my_path)
    plt.show()
    
#other form of vit model
num_classes = 5
input_shape = (224,224,3)
batch_size = 256
num_epochs = 200
image_size = 224  # We'll resize input images to this size
patch_size = 16  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024] 

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
#Implement the patch encoding layer
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
#Build the ViT model

# Compute the mean and the variance of the training data for normalization.

def create_vit_classifier():
    
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes,activation="softmax")(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model
#compile and Run    

#one hot encoding training data labels 
def making_training_and_testing_data(full_data,full_label):
    train_label=[]
    for i in range(len(full_label)):
        if full_label[i]==0:
            train_label.append([1,0,0,0,0])
        elif full_label[i]==1:
            train_label.append([0,1,0,0,0])
        elif full_label[i]==2:
            train_label.append([0,0,1,0,0])
        elif full_label[i]==3:
            train_label.append([0,0,0,1,0])
        elif full_label[i]==4:
            train_label.append([0,0,0,0,1])
    full_label=np.array(train_label)
    return full_data,full_label

#storing all validation data in a list
def val_making_full_data(aug_normal,aug_covid,aug_pneumonia,aug_tb,aug_cat):
    
    aug_normal_labels=[]
    for i in range(len(aug_normal)):
        aug_normal_labels.append(0)
    print(np.shape(aug_normal),np.shape(aug_normal_labels))
    aug_covid_labels=[]
    for i in range(len(aug_covid)):
        aug_covid_labels.append(1)
    print(np.shape(aug_covid),np.shape(aug_covid_labels))
    aug_pneumonia_labels=[]
    for i in range(len(aug_pneumonia)):
        aug_pneumonia_labels.append(2)
    print(np.shape(aug_pneumonia),np.shape(aug_pneumonia_labels))  
    aug_tb_labels=[]
    for i in range(len(aug_tb)):
        aug_tb_labels.append(3)
    print(np.shape(aug_tb),np.shape(aug_tb_labels))  
    aug_cat_labels=[]
    for i in range(len(aug_cat)):
        aug_cat_labels.append(4)
    print(np.shape(aug_cat),np.shape(aug_cat_labels)) 

    full_data=[]
    full_label=[]
    for i in range(len(aug_normal)):
        full_data.append(aug_normal[i])
        full_label.append(aug_normal_labels[i])
    for i in range(len(aug_covid)):
        full_data.append(aug_covid[i])
        full_label.append(aug_covid_labels[i])
    for i in range(len(aug_pneumonia)):
        full_data.append(aug_pneumonia[i])
        full_label.append(aug_pneumonia_labels[i])
    for i in range(len(aug_tb)):
        full_data.append(aug_tb[i])
        full_label.append(aug_tb_labels[i])
    for i in range(len(aug_cat)):
        full_data.append(aug_cat[i])
        full_label.append(aug_cat_labels[i])
        
    full_data=np.array(full_data)
    full_label=np.array(full_label)
    
    full_data=shuffle(full_data,random_state=0)
    full_label=shuffle(full_label,random_state=0)
    global val_my_dict
    val_my_dict=pd.DataFrame(val_my_dict)
    val_my_dict = shuffle(val_my_dict, random_state=0)
    
    return full_data,full_label

#storing all training data in a list
def train_making_full_data(aug_normal,aug_covid,aug_pneumonia,aug_tb,aug_cat):
    
    aug_normal_labels=[]
    for i in range(len(aug_normal)):
        aug_normal_labels.append(0)
    print(np.shape(aug_normal),np.shape(aug_normal_labels))
    aug_covid_labels=[]
    for i in range(len(aug_covid)):
        aug_covid_labels.append(1)
    print(np.shape(aug_covid),np.shape(aug_covid_labels))
    aug_pneumonia_labels=[]
    for i in range(len(aug_pneumonia)):
        aug_pneumonia_labels.append(2)
    print(np.shape(aug_pneumonia),np.shape(aug_pneumonia_labels))  
    aug_tb_labels=[]
    for i in range(len(aug_tb)):
        aug_tb_labels.append(3)
    print(np.shape(aug_tb),np.shape(aug_tb_labels))  
    aug_cat_labels=[]
    for i in range(len(aug_cat)):
        aug_cat_labels.append(4)
    print(np.shape(aug_cat),np.shape(aug_cat_labels)) 

    full_data=[]
    full_label=[]
    for i in range(len(aug_normal)):
        full_data.append(aug_normal[i])
        full_label.append(aug_normal_labels[i])
    for i in range(len(aug_covid)):
        full_data.append(aug_covid[i])
        full_label.append(aug_covid_labels[i])
    for i in range(len(aug_pneumonia)):
        full_data.append(aug_pneumonia[i])
        full_label.append(aug_pneumonia_labels[i])
    for i in range(len(aug_tb)):
        full_data.append(aug_tb[i])
        full_label.append(aug_tb_labels[i])
    for i in range(len(aug_cat)):
        full_data.append(aug_cat[i])
        full_label.append(aug_cat_labels[i])
        
    full_data=np.array(full_data)
    full_label=np.array(full_label)
    
    full_data=shuffle(full_data,random_state=0)
    full_label=shuffle(full_label,random_state=0)
    global train_my_dict
    train_my_dict=pd.DataFrame(train_my_dict)
    train_my_dict = shuffle(train_my_dict, random_state=0)
    
    return full_data,full_label
from tensorflow.keras.applications.resnet50 import ResNet50
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as rp
from tensorflow.keras.applications.xception import preprocess_input as xp
import sklearn.metrics as metrics
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#normal function to make vit model
def build_mlp():
    model = keras.Sequential([
        keras.Input(shape=70, name='Extracted_Traditional_Features'),
        keras.layers.Dense(35, activation=tf.keras.activations.relu, name='Dense1'),
        keras.layers.Dense(17, activation=tf.keras.activations.relu, name='Dense2'),
       
    ])
    return model

def run_experiment(model,train_full_data,train_full_label,val_full_data,val_full_label,early_stopping):

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.CategoricalCrossentropy(from_logits = False), metrics=['accuracy'])
    

    history= model.fit(train_full_data,train_full_label,epochs=100,validation_data=(val_full_data, val_full_label),batch_size=16, callbacks=[early_stopping])
    
    return history,model

if __name__ == '__main__':  #straight away go to this
    #loading data from directory
    normal_dir = "/home/ssharma8/manas_user/data_vit_radiomic/im_Dyskeratotic/im_Dyskeratotic/CROPPED/" #give your normal cases data path here
    #vit_datasets/Dataset_ViT/ViT_dataset/Covid-19
    dir1 = os.path.join(normal_dir,"*.bmp")
    dir2 = os.path.join(normal_dir,"*.jpeg")
    dir = os.path.join(normal_dir,"*.jpg")
    normal_files = glob.glob(dir)
    normal_1 = glob.glob(dir1)
    normal_2 = glob.glob(dir2)
    normal_files.extend(normal_1)
    normal_files.extend(normal_2)
    normal_files=normal_files

    normal_dir = "/home/ssharma8/manas_user/data_vit_radiomic/im_Koilocytotic/im_Koilocytotic/CROPPED/"  #give your covid 19 cases data path here
    dir1 = os.path.join(normal_dir,"*.bmp")
    dir = os.path.join(normal_dir,"*.jpg")
    dir2 = os.path.join(normal_dir,"*.jpeg")
    covid_files = glob.glob(dir)
    covid_files2 = glob.glob(dir2)
    covid_files1 = glob.glob(dir1)
    covid_files.extend(covid_files2)
    covid_files.extend(covid_files1)
    covid_files=covid_files

    normal_dir = "/home/ssharma8/manas_user/data_vit_radiomic/im_Metaplastic/im_Metaplastic/CROPPED/" #give your pneumonia cases data path here
    dir1 = os.path.join(normal_dir,"*.bmp")
    dir2 = os.path.join(normal_dir,"*.jpeg")
    dir = os.path.join(normal_dir,"*.jpg")
    pneumonia_files = glob.glob(dir)
    pneumonia_1 = glob.glob(dir1)
    pneumonia_2 = glob.glob(dir2)
    pneumonia_files.extend(pneumonia_1)
    pneumonia_files.extend(pneumonia_2)
    pneumonia_files=pneumonia_files

    normal_dir = "/home/ssharma8/manas_user/data_vit_radiomic/im_Parabasal/im_Parabasal/CROPPED/" #give your pneumonia cases data path here
    dir1 = os.path.join(normal_dir,"*.bmp")
    dir2 = os.path.join(normal_dir,"*.jpeg")
    dir = os.path.join(normal_dir,"*.jpg")
    tb_files = glob.glob(dir)
    tb_1 = glob.glob(dir1)
    tb_2 = glob.glob(dir2)
    tb_files.extend(tb_1)
    tb_files.extend(tb_2)
    tb_files=tb_files
    
    normal_dir = "/home/ssharma8/manas_user/data_vit_radiomic/im_Superficial-Intermediate/im_Superficial-Intermediate/CROPPED/" #give your pneumonia cases data path here
    dir1 = os.path.join(normal_dir,"*.bmp")
    dir2 = os.path.join(normal_dir,"*.jpeg")
    dir = os.path.join(normal_dir,"*.jpg")
    cat_files = glob.glob(dir)
    cat_1 = glob.glob(dir1)
    cat_2 = glob.glob(dir2)
    cat_files.extend(cat_1)
    cat_files.extend(cat_2)
    cat_files=cat_files
    
    #shuffling and sorting files
    normal_files.sort()
    covid_files.sort()
    pneumonia_files.sort()
    tb_files.sort()
    cat_files.sort()
    normal_files=shuffle(normal_files,random_state=10)
    covid_files=shuffle(covid_files,random_state=10)
    pneumonia_files=shuffle(pneumonia_files,random_state=10)
    tb_files=shuffle(tb_files,random_state=10)
    cat_files=shuffle(cat_files,random_state=10)
    
    
    total_files=(len(normal_files)+len(covid_files)+len(pneumonia_files)+len(tb_files))+len(cat_files)
    
    temp_files=[]
    temp_labels=[]
    for i in range(len(pneumonia_files)):
        temp_files.append(pneumonia_files[i])
        temp_labels.append(0)
    
    for i in range(len(covid_files)):
        temp_files.append(covid_files[i])
        temp_labels.append(1)
    
    for i in range(len(normal_files)):
        temp_files.append(normal_files[i])
        temp_labels.append(2)
        
    for i in range(len(tb_files)):
        temp_files.append(tb_files[i])
        temp_labels.append(3)
    for i in range(len(cat_files)):
        temp_files.append(cat_files[i])
        temp_labels.append(4)
        
    temp_files=shuffle(temp_files,random_state=10)
    temp_labels=shuffle(temp_labels,random_state=10)  
    skf = StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
    
    
#inception_pred.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0002), loss=tf.keras.losses.CategoricalCrossentropy(from_logits = False) , metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', 
        patience=35, 

        min_delta=0.001, 
        mode='min', restore_best_weights=True)
        
       
    model_counter=0    
    accuracy_d=[]
    precision_d=[]
    recall_d=[]
    f_score_d=[]
    accuracy_m=[]
    precision_m=[]
    recall_m=[]
    f_score_m=[]
    accuracy_i=[]
    precision_i=[]
    recall_i=[]
    f_score_i=[]

    
    temp_files=np.array(temp_files)
    temp_labels=np.array(temp_labels)
    
    max_accurate_model_densnet=0
    max_accurate_model_MobileNet=0
    max_accurate_model_incep=0
    counter=-1
    my_fold=0
    model_counter=0
    #spliting data into training and testing
    X_train, X_val, y_train, y_val = train_test_split(temp_files, temp_labels, test_size=0.20, random_state=42,stratify=temp_labels)

    #vit model initialization
    vit_model = vit.vit_b16(
        image_size=224,  # Adjust according to your image size
        activation='softmax',
        pretrained=True,
        include_top=False,
        pretrained_top=False,
    )


    #training data
    train_normal_files=[]
    for i in range(len(X_train)):
        if y_train[i]==2:
            train_normal_files.append(X_train[i])

    train_covid_files=[]
    for i in range(len(X_train)):
        if y_train[i]==1:
            train_covid_files.append(X_train[i])

    train_pneumonia_files=[]
    for i in range(len(X_train)):
        if y_train[i]==0:
            train_pneumonia_files.append(X_train[i])

    train_tb_files=[]
    for i in range(len(X_train)):
        if y_train[i]==3:
            train_tb_files.append(X_train[i])


    train_cat_files=[]
    for i in range(len(X_train)):
        if y_train[i]==4:
            train_cat_files.append(X_train[i])


    #validation data
    val_normal_files=[]
    for i in range(len(X_val)):
        if y_val[i]==2:
            val_normal_files.append(X_val[i])

    val_covid_files=[]
    for i in range(len(X_val)):
        if y_val[i]==1:
            val_covid_files.append(X_val[i])

    val_pneumonia_files=[]
    for i in range(len(X_val)):
        if y_val[i]==0:
            val_pneumonia_files.append(X_val[i])

    val_tb_files=[]
    for i in range(len(X_val)):
        if y_val[i]==3:
            val_tb_files.append(X_val[i])

    val_cat_files=[]
    for i in range(len(X_val)):
        if y_val[i]==4:
            val_cat_files.append(X_val[i])




    #Loading and resizing the images of training data. This function is also extracting radiomic data of images.
    train_aug_normal,train_aug_covid,train_aug_pneumonia,train_aug_tb,train_aug_cat=train_no_data_augmentation(train_normal_files,train_covid_files,train_pneumonia_files,train_tb_files,train_cat_files)
    #Loading and resizing the images of validation data. This function is also extracting radiomic data of images.
    val_aug_normal,val_aug_covid,val_aug_pneumonia,val_aug_tb,val_aug_cat=val_no_data_augmentation(val_normal_files,val_covid_files,val_pneumonia_files,val_tb_files,val_cat_files)

    #combining the training data in one list
    train_full_data,train_full_label=train_making_full_data(train_aug_normal,train_aug_covid,train_aug_pneumonia,train_aug_tb,train_aug_cat)  #getting my full data
    #combining the validation data in one list
    val_full_data,val_full_label=val_making_full_data(val_aug_normal,val_aug_covid,val_aug_pneumonia,val_aug_tb,val_aug_cat)

    #onehot encoding the training data
    train_full_data,train_full_label= making_training_and_testing_data(train_full_data,train_full_label) #dividing full_data into train and test data
    #onehot encoding the validation data
    val_full_data,val_full_label=making_training_and_testing_data(val_full_data,val_full_label)
    
    
    train_radiomic_features=train_my_dict #training radiomic features dictionary
    val_radiomic_features=val_my_dict #validation radiomic features dictionary
    
    # Define the radiomic feature input shape
    radiomic_input_shape = (70,)  # Adjust based on your radiomic feature size
    num_classes=5
    # Define the classification head
    classification_head = Dense(num_classes, activation='softmax')  # Replace `num_classes` with the number of classes in your dataset

    # Define the input layers for ViT features and radiomic features
    vit_input = tf.keras.Input(shape=(224, 224, 3))  # Adjust based on your image size
    radiomic_input = tf.keras.Input(shape=radiomic_input_shape)

    # Extract ViT features from the input images
    vit_features = vit_model(vit_input)

    # Merge the ViT features with radiomic features
    merged_features = Concatenate()([vit_features, radiomic_input])
    reshaped_features = Reshape((-1, merged_features.shape[-1]))(merged_features)
    # Apply global average pooling to reduce dimensionality
    pooled_features = GlobalAveragePooling1D()(reshaped_features)

    # Pass the pooled features through the classification head
    output = classification_head(pooled_features)

    # Create the combined model
    Model = Model(inputs=[vit_input, radiomic_input], outputs=output)
    Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.CategoricalCrossentropy(from_logits = False), metrics=['accuracy'])
    tqdm_callback = tfa.callbacks.TQDMProgressBar()

    history= Model.fit([train_full_data, train_radiomic_features.astype("float32")],  # Replace with your training data
    train_full_label,  # Replace with your training labels
    batch_size=32,  # Adjust as needed
    epochs=100,# Adjust as needed
    validation_data=([val_full_data, val_radiomic_features.astype("float32")], val_full_label),  # Replace with your validation data
    callbacks=[tqdm_callback, early_stopping])

    #getting the results
    pred=Model.predict([val_full_data, val_radiomic_features.astype("float32")])
    predictions = np.argmax(pred,axis = 1)
    y_label=np.argmax(val_full_label,axis = 1).tolist()
    print("Accuracy is:",accuracy_score(predictions,y_label))
    accuracy_d.append(accuracy_score(predictions,y_label))
    print("Precision is:",metrics.precision_score(predictions,y_label,average='macro'))
    precision_d.append(metrics.precision_score(predictions,y_label,average='macro'))
    print("Recall is:",metrics.recall_score(predictions,y_label,average='macro'))
    recall_d.append(metrics.recall_score(predictions,y_label,average='macro'))
    print("F1 Score is:",f1_score(predictions, y_label, average='macro'))
    f_score_d.append(f1_score(predictions, y_label, average='macro'))


    numpy_path='/home/ssharma8/manas_user/history_vitradio'+str(model_counter)+'.npy'
    np.save(numpy_path,history.history)
    fold_number=[1,2,3,4,5]
        
    y_label=np.argmax(val_full_label,axis = 1).tolist()

    cm=metrics.confusion_matrix(predictions,y_label)
    cm_df = pd.DataFrame(cm)

    y_test_binarized=val_full_label

    zipped = list(zip(accuracy_d, precision_d, recall_d, f_score_d))

    df = pd.DataFrame(zipped, columns=['Accuracy', 'Precision', 'Recall', 'F_SCORE'])
    df.to_csv("/home/ssharma8/manas_user/vitradio_results.csv")
    model_counter+=1   
        
    zipped = list(zip(accuracy_d, precision_d, recall_d, f_score_d))

    df = pd.DataFrame(zipped, columns=['Accuracy', 'Precision', 'Recall', 'F_SCORE'])
    df.to_csv("/home/ssharma8/manas_user/vitradio_final_results.csv")
    
        
        
    
    
        
    
