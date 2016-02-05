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
baseDetectionsPath = '/home/santhosh/Projects/ANNCourse/DeeperLookAtPedDetection/Detections/'
baseSavePath = '/home/santhosh/Projects/ANNCourse/ANN/Project/Images/'

DetDirs = ['Detections06', 'Detections07', 'Detections08', 'Detections09', 'Detections10']
finalListImages = '/home/santhosh/Projects/ANNCourse/ANN/Project/ListOfImages.txt'
fileList = open(finalListImages, 'w')

for i in range(6, len(DetDirs)+6):
    DetDir = DetDirs[i-6]
    currDirPath = baseDetectionsPath+DetDir+'/' 
    RoiDirs = commands.getoutput('ls '+currDirPath+' | grep results').split('\n')
    print('=====>  Processing path: %s\n'%(DetDir))
    
    for j in range(len(RoiDirs)):
        
        fname = baseSavePath+'set%.2d'%(i)+'/'+'V%.3d.txt'%(j)
        
        os.system('mkdir -p '+baseSavePath+'set%.2d'%(i)+'/'+'V%.3d/'%(j))
        print('===> Processing sub-path: %s'%(RoiDirs[j]))
        currDirPath_ = currDirPath+RoiDirs[j]
        currSavePath = baseSavePath+'set%.2d'%(i)+'/'+'V%.3d/'%(j)
        images = commands.getoutput('ls '+currDirPath_ + ' | grep .txt').split('\n')
        
        if len(images[0]) > 0:
            #print(images)
            #print(len(images[0]))
            for jter in range(len(images)):
                image = images[jter]
                
                imno = int(re.sub('[^0-9]', '', image)) + 1
                imgPath = baseImagePath+'set%.2d'%(i)+'/V%.3d'%(j)+'/'+re.sub('.txt', '.jpg', image)
                img = cv2.imread(imgPath)
                rows, cols = img.shape[:2]
                if (not img is None):
                    ROIs = open(currDirPath_+'/'+image,'r').readlines()
                    for k in range(len(ROIs)):

                        ROI = re.sub('[\n ]','',ROIs[k]).split(',')
                        #pdb.set_trace()
                        ROI = map(float, ROI)
                        ht = ROI[3]
                        wt = ROI[2]
                        ht = ROI[3]
                        x1 = ROI[0]
                        y1 = ROI[1]
                        x2 = x1 + wt - 1 
                        y2 = y1 + ht - 1 

                        #context_scale = img_size_x - 2*context_pad
                        half_height = (y2-y1+1)/2.0
                        half_width = (x2-x1+1)/2.0
                        center_x = x1 + half_width
                        center_y = y1 + half_height
                        x_add = context_pad*wt/(img_size_x - 2*context_pad)
                        y_add = context_pad*ht/(img_size_y - 2*context_pad)
        
                        x1 = int(max(x1 - x_add, 0)) 
                        y1 = int(max(y1 - y_add, 0)) 
                        x2 = int(min(x2 + x_add, cols-1))
                        y2 = int(min(y2 + y_add, rows-1))

                        ROI_img = img[y1:y2, x1:x2]
                        sizes_w = np.size(ROI_img, 1)
                        sizes_h = np.size(ROI_img, 0)
                        
                        ROI_img = cv2.resize(ROI_img, (img_size_x, img_size_y))
                        
                        imSaveFile = currSavePath+re.sub('.txt', '', image)+'_%.3d.jpg'%(k)
                        cv2.imwrite(imSaveFile, ROI_img)
                        fileList.write(imSaveFile+' ')
                        fileList.write(re.sub('.jpg', '.txt', imSaveFile)+'\n')
                        fsave = open(re.sub('.jpg', '.txt', imSaveFile), 'w')
                        saveString = str(imno) + ', ' + str(ROI[0]) + ', ' + str(ROI[1]) + ', ' + str(ROI[2]) + ', ' + str(ROI[3])  + '\n'
                        fsave.write(saveString)
                        fsave.close()
                #cv2.imshow('Whole img', img)
                #cv2.waitKey(0)

fileList.close()
