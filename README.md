
from typing import List, Set, Dict, Tuple, Optional

import os
import time
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn import impute
from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import model_selection

from sklearn.metrics import accuracy_score
from sklearn import model_selection, metrics
import xgboost as xgb

from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

# Visualization Libraries
import matplotlib as mpl
import matplotlib.pylab as plt
import seaborn as sns
import missingno as msno
TARGET="quality"
class Config:
      path:str = "/content/wine_quality_dataset.csv"

      fast_render: bool = True
      calc_probability: bool = False
      seed: int = 42
      N_FOLDS: int = 5

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rcParams["font.size"] = 16

theme_colors = ["#ffd670", "#70d6ff", "#ff4d6d", "#8338ec", "#90cf8e"]
sns.set_palette(sns.color_palette(theme_colors))

sns.palplot(sns.color_palette(theme_colors), size=1.5)
plt.tick_params(axis="both", labelsize=0, length=0)


 






[ ]
# AIML-Project
