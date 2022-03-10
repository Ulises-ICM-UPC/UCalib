#'''
# Created on 2022 by Gonzalo Simarro, Daniel Calvete and Paola Souto
#'''
#
import os
import sys
#
sys.path.insert(0, 'ucalib')
import ucalib as ucalib
#
pathFolderMain = 'example' # USER DEFINED
assert os.path.exists(pathFolderMain)
#
#''' --------------------------------------------------------------------------
# Calibration of the basis
#''' --------------------------------------------------------------------------
#
pathFolderBasis = pathFolderMain + os.sep + 'basis' # USER DEFINED
eCritical, calibrationModel = 5., 'parabolic' # USER DEFINED (eCritical is in pixels, calibrationModel = 'parabolic', 'quartic' or 'full')
verbosePlot = True # USER DEFINED
#
print('Calibration of the basis')
ucalib.CalibrationOfBasisImages(pathFolderBasis, eCritical, calibrationModel, verbosePlot)
print('Calibration of the basis forcing a unique camera position and intrinsic parameters')
ucalib.CalibrationOfBasisImagesConstantXYZAndIntrinsic(pathFolderBasis, calibrationModel, verbosePlot)
#
#''' --------------------------------------------------------------------------
# (Auto)Calibration of the images
#''' --------------------------------------------------------------------------
#
#pathFolderBasis = pathFolderMain + os.sep + 'basis' # USER DEFINED
pathFolderImages = pathFolderMain + os.sep + 'images' # USER DEFINED
verbosePlot = True # USER DEFINED
nORB, fC, KC = 10000, 5., 4 # USER DEFINED
#
print('Autocalibration of the images')
ucalib.AutoCalibrationOfImages(pathFolderBasis, pathFolderImages, nORB, fC, KC, verbosePlot)
#
#''' --------------------------------------------------------------------------
# Plot planviews
#''' --------------------------------------------------------------------------
#
#pathFolderImages = pathFolderMain + os.sep + 'images' # USER DEFINED
pathFolderPlanviews = pathFolderMain + os.sep + 'planviews' # USER DEFINED
z0, ppm = 3.2, 1.0 # USER DEFINED
verbosePlot = True # USER DEFINED
#
print('Generation of planviews')
ucalib.PlanviewsFromImages(pathFolderImages, pathFolderPlanviews, z0, ppm, verbosePlot)
#
#''' --------------------------------------------------------------------------
# check basis images
#''' --------------------------------------------------------------------------
#
pathFolderBasisCheck = pathFolderMain + os.sep + 'basis_check' # USER DEFINED
#eCritical = 5. # USER DEFINED (eCritical is in pixels)
#
print('Checking of the basis')
ucalib.CheckGCPs(pathFolderBasisCheck, eCritical)
#
