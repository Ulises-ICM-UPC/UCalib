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
pathFolderMain = 'example'
assert os.path.exists(pathFolderMain)
#
#
#''' --------------------------------------------------------------------------
# Calibration of the basis
#''' --------------------------------------------------------------------------
#
pathFolderBasis = os.path.join(pathFolderMain, 'basis')
eCritical, calibrationModel = 5., 'parabolic' # eCritical expressed in pixels, calibrationModel in ['parabolic', 'quartic', 'full']
#givenVariablesDict = {}
givenVariablesDict = {'zc':142.5, 'k1a':-0.0025}
verbosePlot = True
#
print('Calibration of the basis')
ucalib.CalibrationOfBasisImages(pathFolderBasis, eCritical, calibrationModel, givenVariablesDict, verbosePlot)
print('Calibration of the basis forcing a unique camera position and intrinsic parameters')
ucalib.CalibrationOfBasisImagesConstantXYZAndIntrinsic(pathFolderBasis, calibrationModel, givenVariablesDict, verbosePlot)
#
#''' --------------------------------------------------------------------------
# (Auto)Calibration of the images
#''' --------------------------------------------------------------------------
#
#pathFolderBasis = os.path.join(pathFolderMain, 'basis')
pathFolderImages = os.path.join(pathFolderMain, 'images')
overwrite = False
verbosePlot = True
nORB, fC, KC = 10000, 5., 4
#
print('Autocalibration of the images')
ucalib.AutoCalibrationOfImages(pathFolderBasis, pathFolderImages, nORB, fC, KC, overwrite, verbosePlot)
#
#''' --------------------------------------------------------------------------
# Plot planviews
#''' --------------------------------------------------------------------------
#
#pathFolderImages = os.path.join(pathFolderMain, 'images')
pathFolderPlanviews = os.path.join(pathFolderMain, 'planviews')
z0, ppm = 3.2, 2.0
overwrite = False
verbosePlot = True
#
print('Generation of planviews')
ucalib.PlanviewsFromImages(pathFolderImages, pathFolderPlanviews, z0, ppm, overwrite, verbosePlot)
#
#''' --------------------------------------------------------------------------
# check basis images
#''' --------------------------------------------------------------------------
#
pathFolderBasisCheck = os.path.join(pathFolderMain, 'basis_check')
eCritical = 5. # eCritical expressed in pixels
#
print('Checking of the basis')
ucalib.CheckGCPs(pathFolderBasisCheck, eCritical)
#
