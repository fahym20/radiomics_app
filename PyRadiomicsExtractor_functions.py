# -*- coding: utf-8 -*-

import os
import re
from random import randint

dictionary = {}

# Création de l'arborescence de sortie
def createOutputDir(arborescence, outputParentPath, numAno, seriesDir):
    '''
    Crée l'arborescence de sortie en fonction de l'option ARBORESCENCE
    '''
    if arborescence == 1:
        outputDir = os.path.join(outputParentPath, numAno, seriesDir)
    else:
        outputDir = os.path.join(outputParentPath, numAno)

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
        
    return outputDir

# Formatage du temps
def to_hms(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return '{}h{:0>2}m{:0>2}s'.format(h, m, s)