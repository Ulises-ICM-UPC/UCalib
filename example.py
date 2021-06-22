'''
Created on 2021 by Gonzalo Simarro and Daniel Calvete
'''
# own modules
from ucalib import ucalib
#
pathFolderBasis = 'basis'# USER DEFINED (folder with the basis images and their calibrations)

eCrit=5.

ucalib.CalibrationOfBasis(pathFolderBasis,eCrit)


pathFolderImagesToAutoCalibrate = 'imagesToAutoCalibrate' # USER DEFINED (folder with the images to calibrate)

nORB = 10000 # USER DEFINED
fC, KC = 5., 4 # USER DEFINED

ucalib.AutoCalibrationOfImages(pathFolderBasis,pathFolderImagesToAutoCalibrate,nORB,fC,KC)
