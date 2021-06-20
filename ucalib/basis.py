'''
Created on 2021 by Gonzalo Simarro and Daniel Calvete
'''
#
#import ulises as uli
#
import cv2
import numpy as np
import os
from scipy import optimize
# own modules
from ucalib import ulises as uli
#
#pathFolderBasis = './basis' # USER DEFINED: ONLY THIS
#

def nonlinearCalibrationOfBasis(pathFolderBasis):
# --------------------------------------------------------------------------------------------------
# --- preliminaries --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
#
    selectedUnifiedVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'sca']
    dataBasic = uli.LoadDataBasic0({'selectedUnifiedVariablesKeys':selectedUnifiedVariablesKeys})
    assert np.abs(dataBasic['scalesDictionary']['sca'] - dataBasic['scalesDictionary']['sra']) < 1.e-11
    fnsImages = sorted([item for item in os.listdir(pathFolderBasis) if '.' in item and item[item.rfind('.')+1:] in ['jpeg', 'jpg', 'png']])
    fnsCodes = [item[0:item.rfind('.')] for item in fnsImages]
#
# --------------------------------------------------------------------------------------------------
# --- load files -----------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
#
    csForCodes, rsForCodes, xsForCodes, ysForCodes, zsForCodes, chsForCodes, rhsForCodes, z0ForCodes, thereIsHForCodes = [{} for item in range(9)]
    for posFnImage, fnImage in enumerate(fnsImages):
        #
        fnCode = fnsCodes[posFnImage]
        #
        # load ncsForCodes and nrsForCodes
        nr, nc = cv2.imread(pathFolderBasis + os.sep + fnImage).shape[0:2]
        if posFnImage == 0:
            kc, kr = uli.N2K(nc), uli.N2K(nr)
            nc0, nr0 = nc, nr
        else:
            if not (nc == nc0 and nr == nr0):
                print('*** check that all the images have the same size'); assert False
        #
        # load cdgs
        fnCdgTxt = fnCode + 'cdg.txt'
        if not os.path.exists(pathFolderBasis + os.sep + fnCdgTxt):
            print('*** error looking for file {:}'.format(fnCdgTxt)); assert False
        try:
            cs, rs, xs, ys, zs = [[] for item in range(5)]
            openedFile = open(pathFolderBasis + os.sep + fnCdgTxt, 'r')
            listOfLines = openedFile.readlines()
            openedFile.close()
            for line in listOfLines:
                elements = line.split()[0:5]
                cs.append(float(elements[0]))
                rs.append(float(elements[1]))
                xs.append(float(elements[2]))
                ys.append(float(elements[3]))
                zs.append(float(elements[4]))
            cs, rs, xs, ys, zs = [np.asarray(item) for item in [cs, rs, xs, ys, zs]]
            csForCodes[fnCode], rsForCodes[fnCode], xsForCodes[fnCode], ysForCodes[fnCode], zsForCodes[fnCode] = cs, rs, xs, ys, zs
        except:
            print('*** error reading cdg for {:}'.format(fnCode))
        #
        # load horizon points and z0
        fnCdhTxt, fnZmsTxt = fnCode + 'cdh.txt', fnCode + 'zms.txt'
        if any([os.path.exists(pathFolderBasis + os.sep + item) for item in [fnCdhTxt, fnZmsTxt]]):
            if not all([os.path.exists(pathFolderBasis + os.sep + item) for item in [fnCdhTxt, fnZmsTxt]]):
                print('*** error looking for cdh and zms files for {:}'.format(fnImage)); assert False
        try:
            if os.path.exists(pathFolderBasis + os.sep + fnCdhTxt):
                chs, rhs = [[] for item in range(2)]
                openedFile = open(pathFolderBasis + os.sep + fnCdhTxt, 'r')
                listOfLines = openedFile.readlines()
                openedFile.close()
                for line in listOfLines:
                    elements = line.split()[0:2]
                    chs.append(float(elements[0]))
                    rhs.append(float(elements[1]))
                chs, rhs = [np.asarray(item) for item in [chs, rhs]]
                openedFile = open(pathFolderBasis + os.sep + fnZmsTxt, 'r')
                listOfLines = openedFile.readlines()
                openedFile.close()
                for line in listOfLines[0:1]:
                    elements = line.split()[0:1]
                    z0 = float(elements[0])
                chsForCodes[fnCode], rhsForCodes[fnCode], z0ForCodes[fnCode] = chs, rhs, z0
                thereIsHForCodes[fnCode] = True
            else:
                chsForCodes[fnCode], rhsForCodes[fnCode], z0ForCodes[fnCode] = None, None, None
                thereIsHForCodes[fnCode] = False
        except:
            print('*** error reading cdh or zms for {:}'.format(fnCode)); assert False
    print('... files information loaded successfully!')
#
# --------------------------------------------------------------------------------------------------
# --- obtain one initial calibration ---------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
#
    errorTMin = 1.e+6
    for fnCode in fnsCodes:
        #
        optionsTMP = {'chs':None, 'rhs':None, 'z0':None, 'xc':None, 'yc':None, 'zc':None, 'selectedUnifiedVariablesKeys':selectedUnifiedVariablesKeys, 'MonteCarloNOfSeeds':10}
        mainSet, errorT = uli.NonlinearManualCalibrationFromGCPs(xsForCodes[fnCode], ysForCodes[fnCode], zsForCodes[fnCode], csForCodes[fnCode], rsForCodes[fnCode], nc, nr, optionsTMP)
        if errorT < errorTMin:
            xc, yc, zc, ph, sg, ta, k1a, sca = [mainSet[item] for item in ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'sca']]
            errorTMin = 1. * errorT
        if errorTMin < 2.:
            break
    phsForCodes, sgsForCodes, tasForCodes = [{fnCode:item for fnCode in fnsCodes} for item in [ph, sg, ta]]
    #
    # initialize scaledVariables0ForCodes (Code dependent: ph, sg, ta)
    scaledVariables0ForCodes = {}
    for fnCode in fnsCodes:
        scaledVariables0 = np.zeros(3)
        scaledVariables0[0] = phsForCodes[fnCode] / dataBasic['scalesDictionary']['ph']
        scaledVariables0[1] = sgsForCodes[fnCode] / dataBasic['scalesDictionary']['sg']
        scaledVariables0[2] = tasForCodes[fnCode] / dataBasic['scalesDictionary']['ta']
        scaledVariables0ForCodes[fnCode] = scaledVariables0
    #
    # initialize scaledVariables1 (Code independent: xc, yc, zc, k1a, sca)
    scaledVariables1 = np.zeros(5)
    scaledVariables1[0] = xc / dataBasic['scalesDictionary']['xc']
    scaledVariables1[1] = yc / dataBasic['scalesDictionary']['yc']
    scaledVariables1[2] = zc / dataBasic['scalesDictionary']['zc']
    scaledVariables1[3] = k1a / dataBasic['scalesDictionary']['k1a']
    scaledVariables1[4] = sca / dataBasic['scalesDictionary']['sca']
    #
    print('... initial calibration obtained!')
#
# --------------------------------------------------------------------------------------------------
# --- calibrations forcing constant camera position and intrinsic parameter ------------------------
# --------------------------------------------------------------------------------------------------
#
    # perform calibration
    counter, errorT0ForCodes = 0, {}
    while counter < 10000:
        #
        print('... counter = {:4}'.format(counter))
        #
        # recompute Code dependent (scaledVariables0: ph, sg, ta) given Code independent (scaledVariables1: xc, yc, zc, k1a, sca)
        for fnCode in fnsCodes:
            #
            theArgs = {}
            theArgs['dataBasic'], theArgs['scaledVariables1'] = dataBasic, scaledVariables1
            theArgs['nc'], theArgs['nr'], theArgs['kc'], theArgs['kr'] = nc, nr, kc, kr
            theArgs['xs'], theArgs['ys'], theArgs['zs'], theArgs['cs'], theArgs['rs'] = xsForCodes[fnCode], ysForCodes[fnCode], zsForCodes[fnCode], csForCodes[fnCode], rsForCodes[fnCode]
            theArgs['chs'], theArgs['rhs'], theArgs['z0'], theArgs['thereIsH'] = chsForCodes[fnCode], rhsForCodes[fnCode], z0ForCodes[fnCode], thereIsHForCodes[fnCode]
            #
            scaledVariables0 = scaledVariables0ForCodes[fnCode]
            errorT0 = uli.p190227FunctionToMinimize0(scaledVariables0, theArgs)
            #
            scaledVariables0 = optimize.minimize(uli.p190227FunctionToMinimize0, scaledVariables0, args = (theArgs)).x
            errorT0 = uli.p190227FunctionToMinimize0(scaledVariables0, theArgs)
            scaledVariables0ForCodes[fnCode], errorT0ForCodes[fnCode] = scaledVariables0, errorT0
            print('... {:}: errorT {:8.2f}'.format(fnCode, errorT0))
            #
        if (counter > 3 and np.sqrt(np.mean(np.asarray([errorT0ForCodes[fnCode] for fnCode in fnsCodes]) ** 2)) < 1.) or counter > 40:
            break
        #
        # recompute Code independent (scaledVariables1: xc, yc, zc, k1a, sca) given Code dependent (scaledVariables0: ph, sg, ta)
        #
        theArgs = {}
        theArgs['dataBasic'], theArgs['scaledVariables0ForCodes'] = dataBasic, scaledVariables0ForCodes
        theArgs['nc'], theArgs['nr'], theArgs['kc'], theArgs['kr'] = nc, nr, kc, kr
        theArgs['xsForCodes'], theArgs['ysForCodes'], theArgs['zsForCodes'], theArgs['csForCodes'], theArgs['rsForCodes'] = xsForCodes, ysForCodes, zsForCodes, csForCodes, rsForCodes
        theArgs['chsForCodes'], theArgs['rhsForCodes'], theArgs['z0ForCodes'], theArgs['thereIsHForCodes'] = chsForCodes, rhsForCodes, z0ForCodes, thereIsHForCodes
        #
        errorT1 = uli.p190227FunctionToMinimize1(scaledVariables1, theArgs)
        #
        scaledVariables1 = optimize.minimize(uli.p190227FunctionToMinimize1, scaledVariables1, args = (theArgs)).x
        errorT1 = uli.p190227FunctionToMinimize1(scaledVariables1, theArgs)
        #
        if counter == 0:
            scaledVariables1Previous, errorT1Previous = None, 1.e+12
        if errorT1 > errorT1Previous:
            scaledVariables1, errorT1 = 1. * scaledVariables1Previous, 1. * errorT1Previous
            break
        elif errorT1 > 0.9999 * errorT1Previous:
            break
        else:
            scaledVariables1Previous, errorT1Previous = scaledVariables1, errorT1
        #
        counter = counter + 1
#
# --------------------------------------------------------------------------------------------------
# --- write the results ----------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
#
    for fnCode in fnsCodes:
        #
        scaledUnifiedVariables = np.zeros(14)
        scaledUnifiedVariables[0] = scaledVariables1[0] # xc - constant
        scaledUnifiedVariables[1] = scaledVariables1[1] # yc - constant
        scaledUnifiedVariables[2] = scaledVariables1[2] # zc - constant
        scaledUnifiedVariables[3] = scaledVariables0ForCodes[fnCode][0] # ph
        scaledUnifiedVariables[4] = scaledVariables0ForCodes[fnCode][1] # sg
        scaledUnifiedVariables[5] = scaledVariables0ForCodes[fnCode][2] # ta
        scaledUnifiedVariables[6] = scaledVariables1[3] # k1a - constant
        scaledUnifiedVariables[7] = 0. # k2a
        scaledUnifiedVariables[8] = 0. # p1a
        scaledUnifiedVariables[9] = 0. # p2a
        scaledUnifiedVariables[10] = scaledVariables1[4] # sca - constant
        scaledUnifiedVariables[11] = scaledVariables1[4] # sra - constant
        scaledUnifiedVariables[12] = kc / dataBasic['scalesDictionary']['oc'] # oca
        scaledUnifiedVariables[13] = kr / dataBasic['scalesDictionary']['or'] # ora
        unifiedVariables = uli.VariablesScaling(dataBasic, scaledUnifiedVariables, 'unified', 'unscale')
        mainSet = uli.UnifiedVariables2MainSet(nc, nr, unifiedVariables)
        #
        pathFileOut = pathFolderBasis + os.sep + fnCode + 'cal.txt'
        fileout = open(pathFileOut, 'w')
        fileout.write('{:21.9f} {:}\n'.format(mainSet['xc'], 'xc'))
        fileout.write('{:21.9f} {:}\n'.format(mainSet['yc'], 'yc'))
        fileout.write('{:21.9f} {:}\n'.format(mainSet['zc'], 'zc'))
        fileout.write('{:21.9f} {:}\n'.format(mainSet['ph'], 'ph'))
        fileout.write('{:21.9f} {:}\n'.format(mainSet['sg'], 'sg'))
        fileout.write('{:21.9f} {:}\n'.format(mainSet['ta'], 'ta'))
        fileout.write('{:21.9f} {:}\n'.format(mainSet['k1a'], 'k1a'))
        fileout.write('{:21.9f} {:}\n'.format(mainSet['k2a'], 'k2a'))
        fileout.write('{:21.9f} {:}\n'.format(mainSet['p1a'], 'p1a'))
        fileout.write('{:21.9f} {:}\n'.format(mainSet['p2a'], 'p2a'))
        fileout.write('{:21.9f} {:}\n'.format(mainSet['sca'], 'sca'))
        fileout.write('{:21.9f} {:}\n'.format(mainSet['sra'], 'sra'))
        fileout.write('{:21.9f} {:}\n'.format(mainSet['oc'], 'oc'))
        fileout.write('{:21.9f} {:}\n'.format(mainSet['or'], 'or'))
        fileout.write('{:21.0f} {:}\n'.format(mainSet['nc'], 'nc'))
        fileout.write('{:21.0f} {:}\n'.format(mainSet['nr'], 'nr'))
        fileout.write('{:21.9f} {:}\n'.format(errorT0ForCodes[fnCode], 'errorT'))
        fileout.close()
