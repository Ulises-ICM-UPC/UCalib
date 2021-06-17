'''
Created on 2021 by Gonzalo Simarro, Daniel Calvete and Paola Souto
'''
#
import ulises as uli
#
import numpy as np
import os
from scipy import optimize
#
fC, KC = 5., 4 # USER DEFINED
pathFolderBasis = './basis' # USER DEFINED (folder with the basis images and their calibrations)
pathFolderImagesToAutoCalibrate = './imagesToAutoCalibrate' # USER DEFINED (folder with the images to calibrate)
nOfFeaturesORB = 10000 # USER DEFINED
#
unifiedVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']
#
# --------------------------------------------------------------------------------------------------
# --- load information from the basis (ORB and calibrations) ---------------------------------------
# --------------------------------------------------------------------------------------------------
#
fnsBasisImages, ncs, nrs, kpss, dess = uli.p190227ORBForBasis(pathFolderBasis, {'nOfFeatures':nOfFeaturesORB})
#
mainSets, Hs = [[] for item in range(2)]
for posFnBasisImage, fnBasisImage in enumerate(fnsBasisImages):
    fnCalTxt = fnBasisImage[0:fnBasisImage.rfind('.')] + 'cal.txt'
    assert os.path.exists(pathFolderBasis + os.sep + fnCalTxt)
    openedFile = open(pathFolderBasis + os.sep + fnCalTxt, 'r')
    listOfLines = openedFile.readlines()
    openedFile.close()
    unifiedVariablesDictionary = {}
    for line in listOfLines[0:14]:
        elements = line.split()[0:2]
        unifiedVariablesDictionary[elements[1]] = float(elements[0])
    assert set(unifiedVariablesDictionary.keys()) == set(unifiedVariablesKeys)
    nc, nr = int(listOfLines[14].split()[0]), int(listOfLines[15].split()[0])
    unifiedVariables = uli.Dictionary2Array(unifiedVariablesKeys, unifiedVariablesDictionary)
    mainSets.append(uli.UnifiedVariables2MainSet(nc, nr, unifiedVariables))
    Hs.append(np.dot(mainSets[0]['R'], np.transpose(mainSets[posFnBasisImage]['R'])))
#
# --------------------------------------------------------------------------------------------------
# --- automatic calibration of the images to calibrate ---------------------------------------------
# --------------------------------------------------------------------------------------------------
#
fnsImagesToAutoCalibrate = sorted([item for item in os.listdir(pathFolderImagesToAutoCalibrate) if item[item.rfind('.')+1:] in ['jpg', 'png', 'jpeg']])
for fnImageToAutoCalibrate in fnsImagesToAutoCalibrate:
    #
    print('... autocalibration of {:}'.format(fnImageToAutoCalibrate))
    #
    nc0, nr0, kps0, des0, ctrl0 = uli.ORBKeypoints(pathFolderImagesToAutoCalibrate + os.sep + fnImageToAutoCalibrate, {'nOfFeatures':nOfFeaturesORB})
    if not (ctrl0 and nc0 == nc and nr0 == nr):
        print('  ... not automatically calibratable (ORB failure)')
        continue
    #
    # find pairs in the space (u, v)
    uUas0F, vUas0F = [np.asarray([]) for item in range(2)]
    uUas1F, vUas1F = [np.asarray([]) for item in range(2)]
    for posFnBasisImage, fnBasisImage in enumerate(fnsBasisImages):
        # find matches
        cDs0, rDs0, cDsB, rDsB, ersB = uli.ORBMatches(kps0, des0, kpss[posFnBasisImage], dess[posFnBasisImage], options={})
        if len(cDs0) == 0:
            continue
        # update (recall that intrinsic is constant, we use [0])
        uDas0, vDas0 = uli.CR2UaVa(mainSets[0], cDs0, rDs0)[0:2] 
        uUas0, vUas0 = uli.UDaVDa2UUaVUa(mainSets[0], uDas0, vDas0)
        uUas0F, vUas0F = np.concatenate((uUas0F, uUas0)), np.concatenate((vUas0F, vUas0))
        # update (recall that intrinsic is constant, we use [0])
        uDasB, vDasB = uli.CR2UaVa(mainSets[0], cDsB, rDsB)[0:2]
        uUasB, vUasB = uli.UDaVDa2UUaVUa(mainSets[0], uDasB, vDasB)
        dens = Hs[posFnBasisImage][2, 0] * uUasB + Hs[posFnBasisImage][2, 1] * vUasB + Hs[posFnBasisImage][2, 2]
        uUas1 = (Hs[posFnBasisImage][0, 0] * uUasB + Hs[posFnBasisImage][0, 1] * vUasB + Hs[posFnBasisImage][0, 2]) / dens
        vUas1 = (Hs[posFnBasisImage][1, 0] * uUasB + Hs[posFnBasisImage][1, 1] * vUasB + Hs[posFnBasisImage][1, 2]) / dens
        uUas1F, vUas1F = np.concatenate((uUas1F, uUas1)), np.concatenate((vUas1F, vUas1))
    #
    if len(uUas0F) < KC:
        print('  ... not automatically calibratable (K <= {:} < KC)'.format(len(uUas0F)))
        continue
    #
    # apply RANSAC (in pixels)
    cUs0F, rUs0F = uli.UaVa2CR(mainSets[0], uUas0F, vUas0F)[0:2]
    cUs1F, rUs1F = uli.UaVa2CR(mainSets[0], uUas1F, vUas1F)[0:2]
    parametersRANSAC = {'e':0.8, 's':4, 'p':0.999999, 'errorC':2.} # should be fine (on the safe side)
    goodPositions, dsGoodPositions = uli.p190227FindGoodPositionsForHomographyHa01ViaRANSAC(cUs0F, rUs0F, cUs1F, rUs1F, parametersRANSAC)
    if any([item is None for item in [goodPositions, dsGoodPositions]]) or len(goodPositions) < KC:
        print('  ... not automatically calibratable (K <= {:} < KC)'.format(len(goodPositions)))
        continue
    uUas0F, vUas0F = uUas0F[goodPositions], vUas0F[goodPositions]
    uUas1F, vUas1F = uUas1F[goodPositions], vUas1F[goodPositions]
    cUs0F, rUs0F = cUs0F[goodPositions], rUs0F[goodPositions]
    cUs1F, rUs1F = cUs1F[goodPositions], rUs1F[goodPositions]
    #
    # grid selection through distorted pixels
    uDas0F, vDas0F = uli.UUaVUa2UDaVDa(mainSets[0], uUas0F, vUas0F)
    uDas1F, vDas1F = uli.UUaVUa2UDaVDa(mainSets[0], uUas1F, vUas1F)
    cDs0F, rDs0F = uli.UaVa2CR(mainSets[0], uDas0F, vDas0F)[0:2]
    cDs1F, rDs1F = uli.UaVa2CR(mainSets[0], uDas1F, vDas1F)[0:2]
    possGrid = uli.SelectPixelsInGrid(10, nc, nr, cDs0F, rDs0F, dsGoodPositions)[0]
    K = len(possGrid)
    #
    if K < KC:
        print('  ... not automatically calibratable (K = {:} < KC)'.format(K))
        continue
    #
    theArgs = {}
    theArgs['mainSet0'] = mainSets[0]
    theArgs['uUas0F'], theArgs['vUas0F'] = uUas0F[possGrid], vUas0F[possGrid]
    theArgs['uUas1F'], theArgs['vUas1F'] = uUas1F[possGrid], vUas1F[possGrid]
    #
    # find the angles
    x0 = np.asarray([mainSets[0]['ph'], mainSets[0]['sg'], mainSets[0]['ta']])
    error0 = uli.p190227ErrorForAngles(x0, theArgs)
    xN = optimize.minimize(uli.p190227ErrorForAngles, x0, args = (theArgs)).x
    f = uli.p190227ErrorForAngles(xN, theArgs)
    if f > fC:
        print('  ... not automatically calibratable (f = {:4.1e} > fC)'.format(f))
        continue
    #
    pathFileOut = pathFolderImagesToAutoCalibrate + os.sep + fnImageToAutoCalibrate[0:fnImageToAutoCalibrate.rfind('.')] + 'cal.txt'
    fileout = open(pathFileOut, 'w')
    fileout.write('{:21.9f} {:}\n'.format(mainSets[0]['xc'], 'xc'))
    fileout.write('{:21.9f} {:}\n'.format(mainSets[0]['yc'], 'yc'))
    fileout.write('{:21.9f} {:}\n'.format(mainSets[0]['zc'], 'zc'))
    fileout.write('{:21.9f} {:}\n'.format(xN[0], 'ph'))
    fileout.write('{:21.9f} {:}\n'.format(xN[1], 'sg'))
    fileout.write('{:21.9f} {:}\n'.format(xN[2], 'ta'))
    fileout.write('{:21.9f} {:}\n'.format(mainSets[0]['k1a'], 'k1a'))
    fileout.write('{:21.9f} {:}\n'.format(mainSets[0]['k2a'], 'k2a'))
    fileout.write('{:21.9f} {:}\n'.format(mainSets[0]['p1a'], 'p1a'))
    fileout.write('{:21.9f} {:}\n'.format(mainSets[0]['p2a'], 'p2a'))
    fileout.write('{:21.9f} {:}\n'.format(mainSets[0]['sca'], 'sca'))
    fileout.write('{:21.9f} {:}\n'.format(mainSets[0]['sra'], 'sra'))
    fileout.write('{:21.9f} {:}\n'.format(mainSets[0]['oc'], 'oc'))
    fileout.write('{:21.9f} {:}\n'.format(mainSets[0]['or'], 'or'))
    fileout.write('{:21.0f} {:}\n'.format(mainSets[0]['nc'], 'nc'))
    fileout.write('{:21.0f} {:}\n'.format(mainSets[0]['nr'], 'nr'))
    fileout.write('{:21.9f} {:}\n'.format(f, 'f'))
    fileout.write('{:21.0f} {:}\n'.format(K, 'K'))
    fileout.close()
