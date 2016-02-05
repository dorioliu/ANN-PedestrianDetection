import commands
import os
import re
import cv2
import numpy as np
import pdb
import matplotlib.pyplot as plt
import sys

img_size_x = 64 # This is used for cifarnet and pednet
img_size_y = 128 # This is used for cifarnet and pednet
context_pad = 16

baseImagePath = '/media/data/santhosh/PedestrianDetection/data-USA/images/'
baseDetectionsPath = '/home/santhosh/Projects/ANNCourse/ANN/Project/Images/'
baseSavePath = '/home/santhosh/Projects/ANNCourse/DeeperLookAtPedDetection/code3.2.1/data-USA/res/CifarNet-10x/' 
DetDirs = ['set06', 'set07', 'set08', 'set09', 'set10']

for i in range(6, len(DetDirs)+6):
    DetDir = DetDirs[i-6]
    currDirPath = baseDetectionsPath+DetDir+'/' 
    RoiDirs = commands.getoutput('ls '+currDirPath).split('\n')
    print('=====>  Processing path: %s\n'%(DetDir))
    
    os.system('mkdir -p ' + baseSavePath+DetDir+'/')
    
    for j in range(len(RoiDirs)):
        
        fname = baseSavePath+'set%.2d'%(i)+'/'+'V%.3d.txt'%(j)
        saveFile = open(fname, 'wb')

        print('===> Processing sub-path: %s'%(RoiDirs[j]))
        currDirPath_ = currDirPath+RoiDirs[j]
        ROIs = commands.getoutput('ls ' + currDirPath_+'/ | grep .txt').split('\n')
        for k in range(len(ROIs)):

            ROI = ROIs[k]
            fread = open(currDirPath_ + '/' + ROI, 'r')
            
            line1 = re.sub('\n', '', fread.readline())
            line2 = re.sub('\n', '', fread.readline())
            saveString = line1 + ', ' + line2 + '\n'
            print(saveString)
            fread.close()
            saveFile.write(saveString)

        saveFile.close()
