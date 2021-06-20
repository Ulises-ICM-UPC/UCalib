'''
Created on 2021 by Gonzalo Simarro and Daniel Calvete
'''
# own modules
from ucalib import basis, images
#
pathFolderBasis = 'basis'# USER DEFINED (folder with the basis images and their calibrations)

basis.nonlinearCalibrationOfBasis(pathFolderBasis)


fC, KC = 5., 4 # USER DEFINED
pathFolderImagesToAutoCalibrate = 'imagesToAutoCalibrate' # USER DEFINED (folder with the images to calibrate)
nORB = 10000 # USER DEFINED


images.autoCalibration(pathFolderBasis,pathFolderImagesToAutoCalibrate,fC,KC,nORB)

