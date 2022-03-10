#
# Thu Mar 10 15:08:57 2022, extract from Ulises by Gonzalo Simarro
#
import cv2
import copy
import datetime
import numpy as np
import os
import random
from scipy import optimize
import time
#
def AB2Pa11(A, b): # *** 
    '''
    .- input A is a (2*nx11)-float-ndarray
    .- input b is a 2*n-float-ndarray
    .- output Pa11 is a 11-float-ndarray
    '''
    try:
        Pa11 = np.linalg.solve(np.dot(np.transpose(A), A), np.dot(np.transpose(A), b))
    except:
        Pa11 = None
    return Pa11
def AllVariables2MainSet(allVariables, nc, nr, options={}): # 202109141500 # *** 
    ''' comments:
    .- input allVariables is a 14-float-ndarray (xc, yc, zc, ph, sg, ta, k1a, k2a, p1a, p2a, sca, sra, oc, or)
    .- input nc and nr are integers or floats
    .- output mainSet is a dictionary
    '''
    keys, defaultValues = ['orderOfTheHorizonPolynomial', 'radiusOfEarth'], [5, 6.371e+6]
    options = CompleteADictionary(options, keys, defaultValues)
    def MainSet2HorizonLine(mainSet, options={}): # 202109141400
        ''' comments:
        .- input mainSet is a dictionary
        .- output horizonLine is a dictionary
        '''
        keys, defaultValues = ['z0'], 0.
        options = CompleteADictionary(options, keys, defaultValues)
        horizonLine = {key:mainSet[key] for key in ['nc', 'nr']}
        bp = np.sqrt(2. * max([1.e-2, mainSet['zc'] - options['z0']]) * mainSet['radiusOfEarth']) / np.sqrt(np.sum(mainSet['ef'][0:2] ** 2))
        px, py, pz, vx, vy, vz = mainSet['xc'] + bp * mainSet['efx'], mainSet['yc'] + bp * mainSet['efy'], -max([1.e-2, mainSet['zc'] - 2. * options['z0']]), -mainSet['efy'], +mainSet['efx'], 0.
        dc, cc = np.sum(mainSet['Pa'][0, 0:3] * np.asarray([vx, vy, vz])), np.sum(mainSet['Pa'][0, 0:3] * np.asarray([px, py, pz])) + mainSet['Pa'][0, 3]
        dr, cr = np.sum(mainSet['Pa'][1, 0:3] * np.asarray([vx, vy, vz])), np.sum(mainSet['Pa'][1, 0:3] * np.asarray([px, py, pz])) + mainSet['Pa'][1, 3] 
        dd, cd = np.sum(mainSet['Pa'][2, 0:3] * np.asarray([vx, vy, vz])), np.sum(mainSet['Pa'][2, 0:3] * np.asarray([px, py, pz])) + 1.
        ccUh1, crUh1, ccUh0 = dr * cd - dd * cr, dd * cc - dc * cd, dc * cr - dr * cc
        TMP = max([np.sqrt(ccUh1 ** 2 + crUh1 ** 2), 1.e-8])
        horizonLine['ccUh1'] = ccUh1 / TMP
        horizonLine['crUh1'] = crUh1 / TMP
        horizonLine['ccUh0'] = ccUh0 / TMP
        horizonLine['crUh1'] = ClipWithSign(horizonLine['crUh1'], 1.e-8, 1.e+8)
        cUhMin = -0.1 * mainSet['nc']
        cUhMax = +1.1 * mainSet['nc']
        cUhs = np.linspace(cUhMin, cUhMax, 31, endpoint=True)
        rUhs = CUh2RUh(horizonLine, cUhs)
        cDhs, rDhs = CURU2CDRD(mainSet, cUhs, rUhs) # explicit
        A = np.ones((len(cDhs), mainSet['orderOfTheHorizonPolynomial'] + 1))
        for n in range(1, mainSet['orderOfTheHorizonPolynomial'] + 1):
            A[:, n] = cDhs ** n
        b = rDhs
        try:
            horizonLine['ccDh'] = np.linalg.solve(np.dot(np.transpose(A), A), np.dot(np.transpose(A), b))
            if np.max(np.abs(b - np.dot(A, horizonLine['ccDh']))) > 5e-1: # IMP* WATCH OUT
                horizonLine['ccDh'] = np.zeros(mainSet['orderOfTheHorizonPolynomial'] + 1)
                horizonLine['ccDh'][0] = 1.e+2 # IMP* WATCH OUT
        except:
            horizonLine['ccDh'] = np.zeros(mainSet['orderOfTheHorizonPolynomial'] + 1)
            horizonLine['ccDh'][0] = 1.e+2 # IMP* WATCH OUT
        return horizonLine
    allVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or'] # WATCH OUT order matters
    mainSet = {'nc':nc, 'nr':nr, 'orderOfTheHorizonPolynomial':options['orderOfTheHorizonPolynomial'], 'radiusOfEarth':options['radiusOfEarth']}
    allVariablesDictionary = Array2Dictionary(allVariablesKeys, allVariables)
    allVariablesDictionary['sca'] = ClipWithSign(allVariablesDictionary['sca'], 1.e-8, 1.e+8)
    allVariablesDictionary['sra'] = ClipWithSign(allVariablesDictionary['sra'], 1.e-8, 1.e+8)
    allVariables = Dictionary2Array(allVariablesKeys, allVariablesDictionary)
    mainSet['allVariablesDictionary'] = allVariablesDictionary
    mainSet.update(allVariablesDictionary) # IMP* (absorb all the keys in allVariablesDictionary: 'xc', 'yc', ...)
    mainSet['allVariables'] = allVariables
    mainSet['pc'] = np.asarray([mainSet['xc'], mainSet['yc'], mainSet['zc']])
    R = EulerianAngles2R(mainSet['ph'], mainSet['sg'], mainSet['ta'])
    eu, ev, ef = R2UnitVectors(R)
    mainSet['R'] = R
    mainSet['eu'], (mainSet['eux'], mainSet['euy'], mainSet['euz']) = eu, eu
    mainSet['ev'], (mainSet['evx'], mainSet['evy'], mainSet['evz']) = ev, ev
    mainSet['ef'], (mainSet['efx'], mainSet['efy'], mainSet['efz']) = ef, ef
    P, (tu, tv, tf) = np.zeros((3, 4)), [-np.sum(mainSet['pc'] * mainSet[item]) for item in ['eu', 'ev', 'ef']]
    P[0, 0:3], P[0, 3] = mainSet['eu'] / mainSet['sca'] + mainSet['oc'] * mainSet['ef'], tu / mainSet['sca'] + mainSet['oc'] * tf
    P[1, 0:3], P[1, 3] = mainSet['ev'] / mainSet['sra'] + mainSet['or'] * mainSet['ef'], tv / mainSet['sra'] + mainSet['or'] * tf
    P[2, 0:3], P[2, 3] = mainSet['ef'], tf
    mainSet['Pa'] = P / P[2, 3]
    horizonLine = MainSet2HorizonLine(mainSet)
    mainSet['horizonLine'] = horizonLine
    mainSet.update(horizonLine) # IMP* (absorb all the keys in horizonLine: 'ccUh0', ...)
    return mainSet
def AllVariables2SubsetVariables(dataBasic, allVariables, subsetVariablesKeys, options={}): # 202109251523
    ''' comments:
    .- input dataBasic is a dictionary # UNUSED!
    .- input allVariables is a 14-float-ndarray
    .- input subsetVariablesKeys is a string-list
    .- output subsetVariables is a float-ndarray
    '''
    keys, defaultValues = ['possSubsetInAll'], None
    options = CompleteADictionary(options, keys, defaultValues)
    if options['possSubsetInAll'] is not None:
        subsetVariables = allVariables[options['possSubsetInAll']] # best choice
    else:
        allVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or'] # WATCH OUT order matters
        allVariablesDictionary = Array2Dictionary(allVariablesKeys, allVariables)
        subsetVariables = Dictionary2Array(subsetVariablesKeys, allVariablesDictionary)
    return subsetVariables
def ApplyHomographyHa01(Ha01, xs0, ys0): # 202110141303
    ''' comments:
    .- input Ha01 is a 3x3-float-ndarray (allows to transform from 0 to 1)
        .- Ha01[2,2] is not required to be 1
    .- input xs0 and ys0 are float-ndarrays of the same length
    .- output xs1 and ys1 are float-ndarrays of the same length as xs0 and ys0
    '''
    dens = Ha01[2, 0] * xs0 + Ha01[2, 1] * ys0 + Ha01[2, 2]
    if np.min(np.abs(dens)) > 1.e-6:
        xs1 = (Ha01[0, 0] * xs0 + Ha01[0, 1] * ys0 + Ha01[0, 2]) / dens
        ys1 = (Ha01[1, 0] * xs0 + Ha01[1, 1] * ys0 + Ha01[1, 2]) / dens
    else:
        xs1 = 1.e+11 * np.ones(xs0.shape)
        ys1 = 1.e+11 * np.ones(ys0.shape)
    return xs1, ys1
def AreImgMarginsOK(nc, nr, imgMargins): # 202109101200 # *** 
    ''' comments:
    .- input nc and nr are integers
    .- input imgMargins is a dictionary
    .- output areImgMarginsOK is a boolean
    '''
    imgMargins = CompleteImgMargins(imgMargins)
    condC = min([imgMargins['c0'], imgMargins['c1'], nc-1-(imgMargins['c0']+imgMargins['c1'])]) >= 0
    condR = min([imgMargins['r0'], imgMargins['r1'], nr-1-(imgMargins['r0']+imgMargins['r1'])]) >= 0
    areImgMarginsOK = condC and condR
    return areImgMarginsOK
def Array2Dictionary(keys, theArray): # 202109101200 # *** 
    ''' comments:
    .- input keys is a string-list
    .- input theArray is a ndarray of the same length of keys
    .- output theDictionary is a dictionary
    '''
    assert len(set(keys)) == len(keys) == len(theArray)
    theDictionary = {}
    for posKey, key in enumerate(keys):
        theDictionary[key] = theArray[posKey]
    return theDictionary
def Array2Matrix(A, nc, nr, options={}): # 202109141800 # *** 
    ''' comments:
    .- input A is a float-ndarray
    .- input nc and nr are integers
    .- output M is float-2d-ndarray
    '''
    keys, defaultValues = ['way'], ['byRows']
    options = CompleteADictionary(options, keys, defaultValues)
    assert options['way'] in ['byRows', 'byColumns'] and nc * nr == len(A)
    if options['way'] == 'byRows':
        M = np.reshape(A, (nr, nc))
    else:
        M = np.transpose(np.reshape(A, (nc, nr)))
    return M
def CDRD2CURU(mainSet, cDs, rDs): # can be expensive 202109101200 # *** 
    ''' comments:
    .- input mainSet is a dictionary (including at least 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc' and 'or')
    .- input cDs and rDs are float-ndarrays of the same length
    .- output cUs and rUs are float-ndarrays of the same length or None (if it does not succeed)
    '''
    uDas, vDas = CR2UaVa(mainSet, cDs, rDs)
    uUas, vUas = UDaVDa2UUaVUa(mainSet, uDas, vDas) # can be expensive
    if uUas is None or vUas is None:
        cUs, rUs = None, None
    else:
        cUs, rUs = UaVa2CR(mainSet, uUas, vUas)
    return cUs, rUs
def CDRD2CURUForParabolicSquaredDistortion(cDs, rDs, oca, ora, k1asa2): # *** 
    ''' comments:
    .- input cDs and rDs are float-ndarrays
    .- input oca and ora are floats
    .- input k1asa2 is a float (k1a * sa ** 2)
    .- output cUs and rUs are float-ndarrays
    '''
    if np.abs(k1asa2) < 1.e-14:
        cUs = 1. * cDs
        rUs = 1. * rDs
    else:
        dDs2 = (cDs - oca) ** 2 + (rDs - ora) ** 2
        xias = k1asa2 * dDs2
        xas = Xi2XForParabolicDistortion(xias)
        cUs = (cDs - oca) / (1. + xas) + oca
        rUs = (rDs - ora) / (1. + xas) + ora
    cDsR, rDsR = CURU2CDRDForParabolicSquaredDistortion(cUs, rUs, oca, ora, k1asa2)
    assert np.allclose(cDs, cDsR) and np.allclose(rDs, rDsR)
    return cUs, rUs
def CDRDZ2XY(mainSet, cDs, rDs, zs, options={}): # can be expensive 202109231442
    ''' comments:
    .- input main is a dictionary
    .- input cDs, rDs and zs are float-ndarrays of the same length
        .- zs can be a float if zs are all the same
    .- output xs and ys are float-ndarrays of the same length or None (if it does not succeed)
    .- output possGood is None (if it does not succeed or not options['returnGoodPositions'])
    '''
    keys, defaultValues = ['returnGoodPositions', 'imgMargins'], [False, {'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}]
    options = CompleteADictionary(options, keys, defaultValues)
    cUs, rUs = CDRD2CURU(mainSet, cDs, rDs) # potentially expensive
    if cUs is None or rUs is None:
        return None, None, None
    uUas, vUas = CR2UaVa(mainSet, cUs, rUs)
    if isinstance(zs, (np.ndarray)): # float-ndarray
        planes = {'pxs':np.zeros(zs.shape), 'pys':np.zeros(zs.shape), 'pzs':np.ones(zs.shape), 'pts': -zs}
    else: # float
        planes = {'pxs':0., 'pys':0., 'pzs':1., 'pts': -zs}
    optionsTMP = {'returnPositionsRightSideOfCamera':options['returnGoodPositions']}
    xs, ys, zs, possGood = UUaVUa2XYZ(mainSet, planes, uUas, vUas, options=optionsTMP)
    if options['returnGoodPositions']: # so far possGood are at the right side of the camera
        if len(possGood) > 0:
            nc, nr, cDsGood, rDsGood = mainSet['nc'], mainSet['nr'], cDs[possGood], rDs[possGood]
            optionsTMP = {'imgMargins':options['imgMargins']}
            possGoodInGood = CR2PositionsWithinImage(nc, nr, cDsGood, rDsGood, options=optionsTMP)
            possGood = [possGood[item] for item in possGoodInGood]
        else:
            assert possGood == [] # avoidable
    else: # possGood is None from UUaVUa2XYZ
        assert possGood is None # avoidable
    return xs, ys, possGood
def CDh2RDh(horizonLine, cDhs, options={}): # 202109141100 # *** 
    ''' comments:
    .- input horizonLine is a dictionary (including at least 'ccDh')
        .- the horizon line is rDhs = ccDh[0] + ccDh[1] * cDhs + ccDh[2] * cDhs ** 2 + ...
    .- input cDhs is a float-ndarray
    .- output rDhs is a float-ndarray of the same length as cDhs
    .- output possGood is an integer-list or None (if not options['returnGoodPositions'])
    '''
    keys, defaultValues = ['imgMargins', 'returnGoodPositions'], [{'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}, False]
    options = CompleteADictionary(options, keys, defaultValues)
    rDhs = horizonLine['ccDh'][0] * np.ones(cDhs.shape)
    for n in range(1, len(horizonLine['ccDh'])):
        rDhs = rDhs + horizonLine['ccDh'][n] * cDhs ** n
    if options['returnGoodPositions']:
        nc, nr, optionsTMP = horizonLine['nc'], horizonLine['nr'], {'imgMargins':options['imgMargins']}
        possGood = CR2PositionsWithinImage(nc, nr, cDhs, rDhs, options=optionsTMP)
    else:
        possGood = None
    return rDhs, possGood
class MinimizeStopper(object):
    def __init__(self, max_sec=60):
        self.max_sec = max_sec
        self.start = time.time()
    def __call__(self, xk=None):
        elapsed = time.time() - self.start
        if elapsed > self.max_sec:
            assert False
        else:
            pass
def CR2CRInteger(cs, rs): # 202109131000 # *** 
    ''' comments:
    .- input cs and rs are integer- or float-ndarrays
    .- output cs and rs are integer-ndarrays
    '''
    cs = np.round(cs).astype(int)
    rs = np.round(rs).astype(int)
    return cs, rs
def CR2CRIntegerAroundAndWeights(cs, rs): # 202109131400 # *** 
    ''' comments:
    .- input cs and rs are integer- or float-ndarrays of the same length
    .- output csIAround, rsIAround are len(cs)x4 integer-ndarrays 
    .- output wsAround is a len(cs)x4 float-ndarray
    '''
    csFloor, rsFloor = np.floor(cs).astype(int), np.floor(rs).astype(int)
    csDelta, rsDelta = cs - csFloor, rs - rsFloor
    (csIAround, rsIAround), wsAround = [np.zeros((len(cs), 4)).astype(int) for item in range(2)], np.zeros((len(cs), 4))
    csIAround[:, 0], rsIAround[:, 0], wsAround[:, 0] = csFloor + 0, rsFloor + 0, (1. - csDelta) * (1. - rsDelta)
    csIAround[:, 1], rsIAround[:, 1], wsAround[:, 1] = csFloor + 1, rsFloor + 0, (0. + csDelta) * (1. - rsDelta)
    csIAround[:, 2], rsIAround[:, 2], wsAround[:, 2] = csFloor + 0, rsFloor + 1, (1. - csDelta) * (0. + rsDelta)
    csIAround[:, 3], rsIAround[:, 3], wsAround[:, 3] = csFloor + 1, rsFloor + 1, (0. + csDelta) * (0. + rsDelta)
    possCs0, possCs1 = [np.where(np.abs(csDelta - item) < 1.e-8)[0] for item in [0, 1]]
    possRs0, possRs1 = [np.where(np.abs(rsDelta - item) < 1.e-8)[0] for item in [0, 1]]
    for pos in range(4): # all pos (corners) are given the same value
        if len(possCs0) > 0:
            csIAround[possCs0, pos] = csFloor[possCs0]
        if len(possCs1) > 0:
            csIAround[possCs1, pos] = csFloor[possCs1] + 1
        if len(possRs0) > 0:
            rsIAround[possRs0, pos] = rsFloor[possRs0]
        if len(possRs1) > 0:
            rsIAround[possRs1, pos] = rsFloor[possRs1] + 1
    return csIAround, rsIAround, wsAround
def CR2CRIntegerWithinImage(nc, nr, cs, rs, options={}): # 202109141700 # *** 
    ''' comments:
    .- input nc and nr are integers or floats
    .- input cs and rs are float-ndarrays
    .- output csIW and rsIW are integer-ndarrays
    '''
    keys, defaultValues = ['imgMargins'], [{'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}]
    options = CompleteADictionary(options, keys, defaultValues)
    imgMargins = CompleteImgMargins(options['imgMargins'])
    csI, rsI = CR2CRInteger(cs, rs)
    optionsTMP = {'imgMargins':imgMargins, 'rounding':False}
    possWithin = CR2PositionsWithinImage(nc, nr, csI, rsI, optionsTMP)
    csIW, rsIW = csI[possWithin], rsI[possWithin]
    return csIW, rsIW
def CR2PositionsWithinImage(nc, nr, cs, rs, options={}): # 202109131400 # *** 
    ''' comments:
    .- input nc and nr are integers
    .- input cs and rs are integer- or float-ndarrays
    .- output possWithin is an integer-list
    '''
    keys, defaultValues = ['imgMargins', 'rounding'], [{'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}, False]
    options = CompleteADictionary(options, keys, defaultValues)
    imgMargins = CompleteImgMargins(options['imgMargins'])
    assert AreImgMarginsOK(nc, nr, imgMargins)
    if options['rounding']:
        cs, rs = CR2CRInteger(cs, rs)
    cMin, cMax = imgMargins['c0'], nc-1-imgMargins['c1'] # recall that img[:, nc-1, :] is OK, but not img[:, nc, :]
    rMin, rMax = imgMargins['r0'], nr-1-imgMargins['r1'] # recall that img[nr-1, :, :] is OK, but not img[nr, :, :]
    possWithin = np.where((cs >= cMin) & (cs <= cMax) & (rs >= rMin) & (rs <= rMax))[0]
    return possWithin
def CR2UaVa(mainSet, cs, rs): # 202109101200 # *** 
    ''' comments:
    .- input mainSet is a dictionary (including at least 'sca', 'sra', 'oc' and 'or')
        .- mainSet['sca'] and mainSet['sra'] are non-zero, but allowed to be negative
    .- input cs and rs are floats or float-ndarrays
    .- output uas and vas are floats or float-ndarrays
    '''
    uas = (cs - mainSet['oc']) * mainSet['sca'] # WATCH OUT (sca?)
    vas = (rs - mainSet['or']) * mainSet['sra'] # WATCH OUT (sra?)
    return uas, vas
def CR2XY(dataPdfTxt, cs, rs, options={}): # 202109101200 # *** 
    ''' comments:
    .- input dataPdfTxt is a dictionary (including at least 'ppm', 'angle', 'xUpperLeft', 'yUpperLeft', 'nc' and 'nr')
    .- input cs and rs are float-arrays of the same length
    .- output xs and ys are float-arrays of the same length
    .- output possGood is an integer-list or None
    '''
    keys, defaultValues = ['dobleCheck', 'imgMargins', 'returnGoodPositions'], [True, {'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}, False]
    options = CompleteADictionary(options, keys, defaultValues)
    us = + cs / dataPdfTxt['ppm']
    vs = - rs / dataPdfTxt['ppm']
    xs = + np.cos(dataPdfTxt['angle']) * us - np.sin(dataPdfTxt['angle']) * vs + dataPdfTxt['xUpperLeft']
    ys = + np.sin(dataPdfTxt['angle']) * us + np.cos(dataPdfTxt['angle']) * vs + dataPdfTxt['yUpperLeft']
    if options['returnGoodPositions']:
        possGood = CR2PositionsWithinImage(dataPdfTxt['nc'], dataPdfTxt['nr'], cs, rs, options={'imgMargins':options['imgMargins']})
    else:
        possGood = None
    if options['dobleCheck']:
        csR, rsR = XY2CR(dataPdfTxt, xs, ys, options={'dobleCheck':False, 'imgMargins':options['imgMargins'], 'returnGoodPositions':False})[0:2]
        assert np.allclose(cs, csR) and np.allclose(rs, rsR)
    return xs, ys, possGood
def CRWithinImage2NormalizedLengthsAndAreas(nc, nr, cs, rs, options={}): # 1903081230 # *** 
	keys, defaultValues = ['imgMargins'], [None]
	options = CompleteADictionary(options, keys, defaultValues)
	imgMargins = CompleteImgMargins(options['imgMargins'])
	assert len(cs) == len(rs) and AreImgMarginsOK(nc, nr, imgMargins)
	nOfPixels = len(cs)
	positionsWithin = CR2PositionsWithinImage(nc, nr, cs, rs, {'imgMargins':imgMargins}) #!!
	assert len(positionsWithin) == nOfPixels
	cMinimum, cMaximum = imgMargins['c'], nc - 1 - imgMargins['c']
	rMinimum, rMaximum = imgMargins['r'], nr - 1 - imgMargins['r']
	totalLength = np.max([cMaximum - cMinimum, rMaximum - rMinimum])
	totalArea = float((cMaximum - cMinimum) * (rMaximum - rMinimum))
	lengths = np.zeros((nOfPixels, 4))
	lengths[:, 0] = cs - cMinimum
	lengths[:, 1] = cMaximum - cs
	lengths[:, 2] = rs - rMinimum
	lengths[:, 3] = rMaximum - rs
	areas = np.zeros((nOfPixels, 4))
	areas[:, 0] = (cs - cMinimum) * (rs - rMinimum)
	areas[:, 1] = (cs - cMinimum) * (rMaximum - rs)
	areas[:, 2] = (cMaximum - cs) * (rs - rMinimum)
	areas[:, 3] = (cMaximum - cs) * (rMaximum - rs)
	assert np.min(lengths) >= 0. and np.min(areas) >= 0.
	normalizedLengths = np.min(lengths, axis=1) / totalLength
	normalizedAreas = np.min(areas, axis=1) / totalArea
	assert len(normalizedLengths) == len(normalizedAreas) == nOfPixels
	return normalizedLengths, normalizedAreas
def CURU2B(cUs, rUs): # *** 
    poss0, poss1 = Poss0AndPoss1(len(cUs))    
    b = np.zeros(2 * len(cUs))
    b[poss0] = cUs
    b[poss1] = rUs
    return b
def CURU2CDRD(mainSet, cUs, rUs): # 202109101200 # *** 
    ''' comments:
    .- input mainSet is a dictionary
    .- input cUs and rUs are floats or float-ndarrays of the same length
    .- output cDs and rDs are floats or float-ndarrays of the same length
    '''
    uUas, vUas = CR2UaVa(mainSet, cUs, rUs)
    uDas, vDas = UUaVUa2UDaVDa(mainSet, uUas, vUas)
    cDs, rDs = UaVa2CR(mainSet, uDas, vDas)
    return cDs, rDs
def CURU2CDRDForParabolicSquaredDistortion(cUs, rUs, oca, ora, k1asa2): # *** 
    ''' comments:
    .- input cUs and rUs are float-ndarrays
    .- input oca and ora are floats
    .- input k1asa2 is a float (k1a * sa ** 2)
    .- output cUs and rUs are float-ndarrays
    '''
    dUs2 = (cUs - oca) ** 2 + (rUs - ora) ** 2
    cDs = (cUs - oca) * (1. + k1asa2 * dUs2) + oca
    rDs = (rUs - ora) * (1. + k1asa2 * dUs2) + ora
    return cDs, rDs
def CURUXYZ2A(cUs, rUs, xs, ys, zs): # 202201250813
    '''
    .- input cUs, rUs, xs, ys and zs are float-ndarrays of the same length
    .- output A is a (2*len(cUs)x11)-float-ndarray
    '''
    A0 = XYZ2A0(xs, ys, zs)
    A1 = CURUXYZ2A1(cUs, rUs, xs, ys, zs)
    A = np.concatenate((A0, A1), axis=1)
    return A
def CURUXYZ2A1(cUs, rUs, xs, ys, zs): # 202201250812
    '''
    .- input cUs, rUs, xs, ys and zs are float-ndarrays of the same length
    .- output A1 is a (2*len(cUs)x3)-float-ndarray
    '''
    poss0, poss1 = Poss0AndPoss1(len(cUs))
    A1 = np.zeros((2 * len(xs), 3))
    A1[poss0, 0], A1[poss0, 1], A1[poss0, 2] = -cUs*xs, -cUs*ys, -cUs*zs
    A1[poss1, 0], A1[poss1, 1], A1[poss1, 2] = -rUs*xs, -rUs*ys, -rUs*zs
    return A1
def CUh2RUh(horizonLine, cUhs): # 202109101200 # *** 
    ''' comments:
    .- input horizonLine is a dictionary (including at least 'ccUh1', 'crUh1' and 'ccUh0')
        .- the horizon line is 'ccUh1' * cUhs + 'crUh1' * rUhs + 'ccUh0' = 0, i.e., rUhs = - ('ccUh0' + 'ccUh1' * cUhs) / 'crUh1'
    .- input cUhs is a float-ndarray
    .- output rUhs is a float-ndarray
    '''
    crUh1 = ClipWithSign(horizonLine['crUh1'], 1.e-8, 1.e+8)
    rUhs = - (horizonLine['ccUh0'] + horizonLine['ccUh1'] * cUhs) / crUh1
    return rUhs
def ClipWithSign(xs, x0, x1): # 202109101200 # *** 
    ''' comments:
    .- input xs is a float of a float-ndarray
    .- input x0 and x1 are floats so that x1 >= x0 >= 0
    .- output xs is a float of a float-ndarray
        .- output xs is in [-x1, -x0] U [x0, x1] and retains the signs of input xs
    '''
    assert x1 >= x0 and x0 >= 0.
    signs = np.sign(xs)
    if isinstance(signs, (np.ndarray)): # array
        signs[signs == 0] = 1
    else: # float
        if signs == 0:
            signs = 1
    xs = signs * np.clip(np.abs(xs), x0, x1)
    return xs
def Cloud2Rectangle(xs, ys, options={}): # 202109161400
    ''' comments:
    .- input xs and ys are float-ndarrays
    .- output angle, x, y, H, W are floats
    '''
    def Cloud2RectangleAnalysisForAnAngle(angle, xs, ys, options={}): # 202109280948
        ''' comments:
        .- input angle is a float
        .- input xs and ys are float-ndarrays of the same length
        '''
        keys, defaultValues = ['margin'], [0.]
        options = CompleteADictionary(options, keys, defaultValues)
        lDs = - np.sin(angle) * xs + np.cos(angle) * ys # direction vector = (+cos, +sin): normalized!
        lPs = + np.cos(angle) * xs + np.sin(angle) * ys # direction vector = (-sin, +cos): normalized!
        edgeD = np.max(lDs) - np.min(lDs) + 2 * options['margin']
        edgeP = np.max(lPs) - np.min(lPs) + 2 * options['margin']
        lD0 = {'lx':-np.sin(angle), 'ly':np.cos(angle), 'lt':-(np.min(lDs)-options['margin'])}
        lD1 = {'lx':-np.sin(angle), 'ly':np.cos(angle), 'lt':-(np.max(lDs)+options['margin'])}
        lP0 = {'lx':+np.cos(angle), 'ly':np.sin(angle), 'lt':-(np.min(lPs)-options['margin'])}
        lP1 = {'lx':+np.cos(angle), 'ly':np.sin(angle), 'lt':-(np.max(lPs)+options['margin'])}
        xcs, ycs = [np.zeros(4) for item in range(2)]
        xcs[0], ycs[0] = IntersectionOfTwoLines(lD0, lP0, options={})[0:2] 
        xcs[1], ycs[1] = IntersectionOfTwoLines(lD1, lP0, options={})[0:2]
        xcs[2], ycs[2] = IntersectionOfTwoLines(lD1, lP1, options={})[0:2] 
        xcs[3], ycs[3] = IntersectionOfTwoLines(lD0, lP1, options={})[0:2]
        edge01, edge12 = edgeD, edgeP
        return xcs, ycs, edge01, edge12
    keys, defaultValues = ['margin'], [0.]
    options = CompleteADictionary(options, keys, defaultValues)
    angles, area = np.arange(-3.14, 3.15, 0.01), 1.e+11
    for posAngle, angle in enumerate(angles):
        xcs, ycs, edge01, edge12 = Cloud2RectangleAnalysisForAnAngle(angle, xs, ys, options={'margin':options['margin']})
        if edge01 * edge12 < area:
            area = edge01 * edge12
            angleO, xcsO, ycsO, edge01O, edge12O = [copy.deepcopy(item) for item in [angle, xcs, ycs, edge01, edge12]]
    angle, xcs, ycs, edge01, edge12 = [copy.deepcopy(item) for item in [angleO, xcsO, ycsO, edge01O, edge12O]]
    pos = np.argmin(np.sqrt((xcs - xs[0]) ** 2 + (ycs - ys[0]) ** 2))
    if pos in [0, 1]:
        x1, y1, x2, y2, x3, y3 = xcs[pos], ycs[pos], xcs[pos+1], ycs[pos+1], xcs[pos+2], ycs[pos+2]
    elif pos == 2:
        x1, y1, x2, y2, x3, y3 = xcs[pos], ycs[pos], xcs[pos+1], ycs[pos+1], xcs[0], ycs[0]
    elif pos == 3:
        x1, y1, x2, y2, x3, y3 = xcs[pos], ycs[pos], xcs[0], ycs[0], xcs[1], ycs[1]
    angle, x, y, W, H = np.angle((x2 - x1) + 1j * (y2 - y1)), x1, y1, np.sqrt((x2-x1)**2+(y2-y1)**2), np.sqrt((x3-x2)**2+(y3-y2)**2)
    return angle, x, y, H, W
def CompleteADictionary(theDictionary, keys, defaultValues): # 202109101200 # *** 
    ''' comments:
    .- input theDictionary is a dictionary
    .- input keys is a string-list
    .- input defaultValues is a list of the same length of keys or a single value (string, float, integer or None)
    .- output theDictionary is a dictionary that includes keys and defaultValues for the keys not in input theDictionary
    '''
    if set(keys) <= set(theDictionary.keys()): # no work to do
        pass
    else:
        if isinstance(defaultValues, (list)): # defaultValues is a list
            assert len(keys) == len(defaultValues)
            for posKey, key in enumerate(keys):
                if key not in theDictionary.keys(): # only assigns if there is no key
                    theDictionary[key] = defaultValues[posKey]
        else: # defaultValues is a single value
            for key in keys:
                if key not in theDictionary.keys(): # only assigns if there is no key
                    theDictionary[key] = defaultValues
    return theDictionary
def CompleteImgMargins(imgMargins): # 202109101200 # *** 
    ''' comments:
    .- input imgMargins is a dictionary or None
        .- if imgMargins['isComplete'], then it does nothing
        .- if imgMargins is None, then it is initialized to {'c':0, 'r':0}
        .- if imgMargins includes 'c', then generates 'c0' and 'c1' (if not included); otherwise, 'c0' and 'c1' must already be included
        .- if imgMargins includes 'r', then generates 'r0' and 'r1' (if not included); otherwise, 'r0' and 'r1' must already be included
        .- imgMargins['c*'] and imgMargins['r*'] are integers
    .- output imgMargins is a dictionary (including at least 'c0', 'c1', 'r0' and 'r1' and 'isComplete')
        .- imgMargins['c*'] and imgMargins['r*'] are integers
    '''
    if imgMargins is not None and 'isComplete' in imgMargins.keys() and imgMargins['isComplete']:
        return imgMargins
    if imgMargins is None:
        imgMargins = {'c':0, 'r':0}
    for letter in ['c', 'r']:
        try:
            assert int(imgMargins[letter]) == imgMargins[letter]
        except: # imgMargins[letter] is not an integer (it is None or it even does not exist)
            for number in ['0', '1']: # check that c0(r0) and c1(r1) are already in imgMargins
                assert int(imgMargins[letter+number]) == imgMargins[letter+number]
            continue # go to the next letter since letter+number already ok for this letter
        for number in ['0', '1']:
            try: 
                assert int(imgMargins[letter+number]) == imgMargins[letter+number]
            except:
                imgMargins[letter+number] = imgMargins[letter]
    imgMargins['isComplete'] = True
    return imgMargins
def CreatePlanview(planviewPrecomputations, imgs): # 202109151300 VA # *** 
    ''' comments:
    .- input planviewPrecomputations is a dictionary (including at least 'nc', 'nr' and 'cameras' and dictionaries for cameras)
    .- input imgs is a dictionary for cameras
    .- imgPlanview is a cv2.image or None (if there are no images in imgs)
    '''
    assert all([img.shape[2] == 3 for img in [imgs[camera] for camera in imgs.keys()]])
    cameras = [item for item in imgs.keys() if item in planviewPrecomputations['cameras']]
    if len(cameras) == 0:
        return None
    wsPlanview = np.zeros(planviewPrecomputations['nc'] * planviewPrecomputations['nr'])
    for camera in cameras:
        planviewPositions = planviewPrecomputations[camera]['planviewPositions']
        wsPlanview[planviewPositions] = wsPlanview[planviewPositions] + planviewPrecomputations[camera]['ws']
    imgPlanview = np.zeros((planviewPrecomputations['nr'], planviewPrecomputations['nc'], 3))
    for camera in cameras:
        planviewPositions = planviewPrecomputations[camera]['planviewPositions']
        csPlanview = np.round(planviewPrecomputations['cs'][planviewPositions]).astype(int) # nOfPlanviewPositions (CHECK round?)
        rsPlanview = np.round(planviewPrecomputations['rs'][planviewPositions]).astype(int) # nOfPlanviewPositions (CHECK round?)
        wsCamera = planviewPrecomputations[camera]['ws'] # nOfPlanviewPositions
        for corner in range(4):
            csIACamera = planviewPrecomputations[camera]['csIA'][:, corner] # nOfPlanviewPositions
            rsIACamera = planviewPrecomputations[camera]['rsIA'][:, corner] # nOfPlanviewPositions
            wsACamera = planviewPrecomputations[camera]['wsA1'][:, corner]  # nOfPlanviewPositions
            contribution = imgs[camera][rsIACamera, csIACamera, :] * np.outer(wsACamera * wsCamera / wsPlanview[planviewPositions], np.ones(3))
            imgPlanview[rsPlanview, csPlanview, :] = imgPlanview[rsPlanview, csPlanview, :] + contribution
    imgPlanview = imgPlanview.astype(np.uint8)
    return imgPlanview
def Dictionary2Array(keys, theDictionary): # 202109101200 # *** 
    ''' comments:
    .- input keys is a string-list
    .- input theDictionary is a dictionary
    .- output theArray is a ndarray
    '''
    assert set(keys) <= set(theDictionary.keys()) # avoidable
    theArray = np.zeros(len(keys))
    for posKey, key in enumerate(keys):
        theArray[posKey] = theDictionary[key]
    return theArray
def DisplayCRInImage(img, cs, rs, options={}): # 202109141700 # *** 
    ''' comments:
    .- input img is a cv2 image
    .- input cs and rs are integer- or float-ndarrays of the same length
        .- they are not required to be within the image
    .- output imgOut is a cv2 image
    '''
    keys, defaultValues = ['colors', 'imgMargins', 'size'], [[[0, 0, 0]], {'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}, 2]
    options = CompleteADictionary(options, keys, defaultValues)
    ''' comments:
    .- options['colors'] is a list of colors:
        .- if len(options['colors']) = 1, all the pixels have the same color
        .- if len(options['colors']) > 1, it must be len(options['colors']) > len(cs)
    '''
    imgOut = copy.deepcopy(img)
    nr, nc = img.shape[0:2]
    optionsTMP = {item:options[item] for item in ['imgMargins']}
    csIW, rsIW = CR2CRIntegerWithinImage(nc, nr, cs, rs, optionsTMP)
    if len(options['colors']) == 1:
        colors = [options['colors'][0] for item in range(len(csIW))]
    else: # we do not require
        assert len(options['colors']) >= len(csIW) == len(rsIW)
        colors = options['colors']
    if len(csIW) == len(rsIW) > 0:
        for pos in range(len(csIW)):
            cv2.circle(imgOut, (csIW[pos], rsIW[pos]), int(options['size']), colors[pos], -1)
    return imgOut
def ErrorC(xc, yc, zc, mainSet): # 202109231429
    ''' comments:
    .- input xc, yc, zc are floats
    .- input mainSet is a dictionary (including at least 'xc', 'yc' and 'zc')
    .- output errorC is a float
    .- last revisions without modifications: 20220125
    '''
    errorC = np.sqrt((mainSet['xc'] - xc) ** 2 + (mainSet['yc'] - yc) ** 2 + (mainSet['zc'] - zc) ** 2) # distance (in m)
    return errorC
def ErrorG(xs, ys, zs, cs, rs, mainSet): # 202109131100 # *** 
    ''' comments:
    .- input xs, ys, zs, cs and rs are float-ndarrays of the same length
        .- cs and rs are distorted pixel coordinates
    .- input mainSet is a dictionary
    .- output errorG is a float
    .- avoids the use of implicit functions
    .- last revisions without modifications: 20220125
    '''
    csR, rsR = XYZ2CDRD(mainSet, xs, ys, zs, options={})[0:2] #! IMP*
    errorG = np.sqrt(np.mean((csR - cs) ** 2 + (rsR - rs) ** 2)) # RMSE (in pixels)
    return errorG
def ErrorH(chs, rhs, horizonLine): # 202109131100 # *** 
    ''' comments:
    .- input chs and rhs are float-ndarrays of the same length
        .- chs and rhs are distorted pixel coordinates
    .- input horizonLine is a dictionary
    .- output errorH is a float
    .- last revisions without modifications: 20220125
    '''
    rhsR = CDh2RDh(horizonLine, chs, options={})[0] #! IMP*
    errorH = np.sqrt(np.mean((rhsR - rhs) ** 2)) # RMSE (in pixels)
    return errorH
def ErrorT(dataForCal, mainSet, options={}): # 202201250914
    ''' comments:
    .- input dataForCal is a dictionary (including at least 'cs', 'rs', 'xs', 'ys', 'zs' and 'aG')
        .- dataForCal keys for errorC: 'xc', 'yc', 'zc' and 'aC'
        .- dataForCal keys for errorH: 'chs', 'rhs' (that can be empty ndarrays) and 'aH'
    .- input mainSet is a dictionary
    .- output errorT is a float
    '''
    keys, defaultValues = ['verbose'], [False]
    options = CompleteADictionary(options, keys, defaultValues)
    keysTMP = ['xc', 'yc', 'zc', 'aC']
    if set(keysTMP) <= dataForCal.keys() and all([dataForCal[item] is not None for item in keysTMP]) and dataForCal['aC'] > 1.e-8: # account for errorC
        xc, yc, zc = [dataForCal[item] for item in ['xc', 'yc', 'zc']]
        errorC = ErrorC(xc, yc, zc, mainSet) / dataForCal['aC']
    else:
        errorC = 0.
    xs, ys, zs, cs, rs = [dataForCal[item] for item in ['xs', 'ys', 'zs', 'cs', 'rs']]
    errorG = ErrorG(xs, ys, zs, cs, rs, mainSet) / dataForCal['aG']
    keysTMP = ['chs', 'rhs', 'aH']
    if set(keysTMP) <= set(dataForCal.keys()) and all([dataForCal[item] is not None for item in keysTMP]) and len(dataForCal['chs']) == len(dataForCal['rhs']) > 0 and dataForCal['aH'] > 1.e-8:
        chs, rhs = [dataForCal[item] for item in ['chs', 'rhs']]
        errorH = ErrorH(chs, rhs, mainSet['horizonLine']) / dataForCal['aH']
    else:
        errorH = 0.
    errorT = errorC + errorG + errorH
    if options['verbose']:
        print('... errorT = {:4.3f} + {:4.3f} + {:4.3e} = {:4.3e}'.format(errorC, errorG, errorH, errorT))
    return errorT
def ErrorT2PerturbationFactorAndNOfSeeds(errorT): # MUTABLE 202201270857
    ''' comments:
    .- input errorT is a float
    .- output perturbationFactor is a float
    .- output nOfSeeds is an integer
    '''
    factorForNOfSeeds = 2. # 10.
    log10E = np.log10(max([errorT, 1.]))
    perturbationFactor, nOfSeeds = 0.1 + 0.4 * log10E, 1 * int(factorForNOfSeeds + factorForNOfSeeds * log10E + 2. * factorForNOfSeeds * log10E ** 2) + 2 # WATCH OUT (nOfSeeds could be doubled)
    return perturbationFactor, nOfSeeds
def EulerianAngles2R(ph, sg, ta): # 202109131100 # *** 
    ''' comments:
    .- input ph, sg and ta are floats
    .- output R is a orthonormal 3x3-float-ndarray positively oriented
    '''
    eu, ev, ef = EulerianAngles2UnitVectors(ph, sg, ta)
    R = UnitVectors2R(eu, ev, ef)
    return R
def EulerianAngles2UnitVectors(ph, sg, ta): # 202109231415
    ''' comments:
    .- input ph, sg and ta are floats
    .- output eu, ev and ef are 3-float-ndarrays which are orthonormal and positively oriented
    '''
    sph, cph = np.sin(ph), np.cos(ph)
    ssg, csg = np.sin(sg), np.cos(sg)
    sta, cta = np.sin(ta), np.cos(ta)
    eux = +csg * cph - ssg * sph * cta
    euy = -csg * sph - ssg * cph * cta
    euz = -ssg * sta
    eu = np.asarray([eux, euy, euz])
    evx = -ssg * cph - csg * sph * cta
    evy = +ssg * sph - csg * cph * cta
    evz = -csg * sta
    ev = np.asarray([evx, evy, evz])
    efx = +sph * sta
    efy = +cph * sta
    efz = -cta
    ef = np.asarray([efx, efy, efz])
    R = UnitVectors2R(eu, ev, ef)
    assert np.allclose(np.dot(R, np.transpose(R)), np.eye(3)) and np.allclose(np.linalg.det(R), 1.)
    return eu, ev, ef
def FindAFirstSeed(dataBasic, dataForCal, subsetVariablesKeys, subCsetVariablesDictionary, options={}): # 202109241556
    ''' comments
    .- input dataBasic is a dictionary
    .- input dataForCal is a dictionary (including at least 'nc', 'nr')
    .- input subsetVariablesKeys is a string-list
    .- input subCsetVariablesDictionary is a dictionary
    .- output mainSetSeed is a dictionary or None (if it does not succeed, i.e., if errorT >= ctt * imgDiagonal)
    .- output errorTSeed is a float os None (if it does not succeed, i.e., if errorT >= ctt * imgDiagonal)
    '''
    keys, defaultValues = ['counter', 'timedelta', 'xc', 'yc', 'zc'], [1000, datetime.timedelta(seconds=600), None, None, None]
    options = CompleteADictionary(options, keys, defaultValues)
    dataForCal['aC'] = 0. # the errorC is not used
    for key in ['xc', 'yc', 'zc']: # camera position
        if options[key] is None:
            dataForCal[key] = np.mean(dataForCal[key[0] + 's'])
            dataBasic['referenceRangesDictionary'][key] = 10. * np.std(dataForCal[key[0] + 's']) # WATCH OUT
            if key == 'zc': # special
                dataForCal[key] = dataForCal[key] + 50. # WATCH OUT
                dataBasic['referenceRangesDictionary'][key] = 150. # WATCH OUT
        else:
            dataForCal[key] = options[key] # and referenceRangesDictionary is not changed
    (nc, nr), optionsHL = (dataForCal['nc'], dataForCal['nr']), {key:dataBasic[key] for key in ['orderOfTheHorizonPolynomial', 'radiusOfEarth']}
    mainSetSeed, errorTSeed, counter, time0, imgDiagonal = None, 1.e+11, 0, datetime.datetime.now(), np.sqrt(nc ** 2 + nr ** 2)
    theArgs = {'dataBasic':dataBasic, 'dataForCal':dataForCal, 'subsetVariablesKeys':subsetVariablesKeys, 'subCsetVariablesDictionary':subCsetVariablesDictionary}
    while datetime.datetime.now() - time0 < options['timedelta'] and counter < options['counter']: # IMP*
        counter, scaledSubsetVariables = counter + 1, GenerateRandomScaledVariables(dataBasic, subsetVariablesKeys, options={key:dataForCal[key] for key in ['xc', 'yc', 'zc']})
        try: # IMP* to try, WATCH OUT: dataForCal must include 'aG' and, if 'chs' and 'rhs' are considered, 'aH'
            errorT = ScaledSubsetVariables2FTM(scaledSubsetVariables, theArgs)
        except: # IMP* not to inform
            continue
        if not IsVariablesOK(scaledSubsetVariables, subsetVariablesKeys):
            continue
        if errorT < 0.5 * imgDiagonal: # IMP* WATCH OUT
            try:
                scaledSubsetVariables = optimize.minimize(ScaledSubsetVariables2FTM, scaledSubsetVariables, args=(theArgs), callback=MinimizeStopper(10.)).x # IMP*
            except:
                pass
            if not IsVariablesOK(scaledSubsetVariables, subsetVariablesKeys):
                continue
            errorT = ScaledSubsetVariables2FTM(scaledSubsetVariables, theArgs)
            if errorT < errorTSeed:
                mainSet = ScaledSubsetVariables2MainSet(dataBasic, scaledSubsetVariables, subsetVariablesKeys, subCsetVariablesDictionary, nc, nr, options=optionsHL)
                assert np.isclose(errorT, ErrorT(dataForCal, mainSet, options={'verbose':False})) # avoidable
                mainSetSeed, errorTSeed = [copy.deepcopy(item) for item in [mainSet, errorT]]
            if errorT < 0.05 * imgDiagonal: # IMP* WATCH OUT
                break
    return mainSetSeed, errorTSeed
def FindGoodPositionsForHomographyHa01ViaRANSAC(xs0, ys0, xs1, ys1, parametersRANSAC): # 202110141316
    ''' comments:
    .- input xs0, ys0, xs1 and ys1 are float-ndarrays of the same length
    .- input parametersRANSAC is a dictionary (including at least 'p', 'e', 's' and 'errorC')
        .- p is the probability that all points are good (0.9999)
        .- e is the probability that one point is wrong (0.8)
        .- s is the number of points required by the model (2 for linear regressions, e.g.)
        .- errorC is the critical error for one point to be considered ok for a model
    .- output possGood is an integer-list or None (if it does not succeed)
    '''
    if len(xs0) == len(ys0) == len(xs1) == len(ys1) >= 4:
        N = int(np.log(1. - parametersRANSAC['p']) / np.log(1. - (1. - parametersRANSAC['e']) ** parametersRANSAC['s'])) + 1
        possGood = []
        for iN in range(N):
            poss4 = random.sample(range(0, len(xs0)), 4)
            Ha01 = FindHomographyHa01(xs0[poss4], ys0[poss4], xs1[poss4], ys1[poss4])
            if Ha01 is None:
                continue
            xs1R, ys1R = ApplyHomographyHa01(Ha01, xs0, ys0)
            errors = np.sqrt((xs1R - xs1) ** 2 + (ys1R - ys1) ** 2)
            possGoodH = np.where(errors < parametersRANSAC['errorC'])[0]
            if len(possGoodH) > len(possGood):
                possGood = copy.deepcopy(possGoodH)
        if len(possGood) == 0:
            possGood = None
    else:
        possGood = None
    return possGood
def FindHomographyHa01(xs0, ys0, xs1, ys1): # 202110141311 write it as vector? # *** 
    ''' comments:
    .- input xs0, ys0, xs1 and ys1 are float-ndarrays of the same length
    .- output Ha01 is a 3x3-float-ndarray or None (if it does not succeed)
        .- Ha01[2,2] is 1
        .- Ha01 allows to transform from 0 to 1
    '''
    if len(xs0) == len(ys0) == len(xs1) == len(ys1) >= 4:
        A, b = np.zeros((2 * len(xs0), 8)), np.zeros(2 * len(xs0))
        poss0, poss1 = Poss0AndPoss1InFind2DTransform(len(xs0))
        A[poss0, 0], A[poss0, 1], A[poss0, 2], A[poss0, 6], A[poss0, 7], b[poss0] = xs0, ys0, np.ones(xs0.shape), -xs0 * xs1, -ys0 * xs1, xs1
        A[poss1, 3], A[poss1, 4], A[poss1, 5], A[poss1, 6], A[poss1, 7], b[poss1] = xs0, ys0, np.ones(xs0.shape), -xs0 * ys1, -ys0 * ys1, ys1
        try:
            sol = np.linalg.solve(np.dot(np.transpose(A), A), np.dot(np.transpose(A), b))
            Ha01 = np.ones((3, 3)) # IMP* initialize with one
            Ha01[0, 0:3], Ha01[1, 0:3], Ha01[2, 0:2] = sol[0:3], sol[3:6], sol[6:8]
        except: # aligned points
            Ha01 = None
    else:
        Ha01 = None
    return Ha01
def FindHomographyHa01ViaRANSAC(xs0, ys0, xs1, ys1, parametersRANSAC): # 202110141321
    ''' comments:
    .- input xs0, ys0, xs1 and ys1 are float-ndarrays of the same length
    .- input parametersRANSAC is a dictionary (including at least 'p', 'e', 's' and 'errorC')
        .- p is the probability that all points are good (0.9999)
        .- e is the probability that one point is wrong (0.8)
        .- s is the number of points required by the model (2 for linear regressions, e.g.)
        .- errorC is the critical error for one point to be considered ok for a model
    .- output Ha is a 3x3-float-ndarray or None (if it does not succeed)
        .- Ha01[2,2] is 1
        .- Ha01 allows to transform from 0 to 1
    .- output possGood is an integer-list or None (if it does not succeed)
    '''
    possGood = FindGoodPositionsForHomographyHa01ViaRANSAC(xs0, ys0, xs1, ys1, parametersRANSAC)
    if possGood is None:
        Ha01 = None
    else:
        Ha01 = FindHomographyHa01(xs0[possGood], ys0[possGood], xs1[possGood], ys1[possGood])
    return Ha01, possGood
def GCPs2K1asa2(cDs, rDs, xs, ys, zs, oca, ora, k1asa2Min, k1asa2Max, options={}): # *** 
    ''' comments:
    .- input cDs, rDs, xs, ys and zs are is a float-ndarrays of the same length
    .- input oca, ora, k1asa2Min, k1asa2Max are floats
    .- output k1asa2 is a float
    '''
    keys, defaultValues = ['nOfK1asa2'], [1000]
    options = CompleteADictionary(options, keys, defaultValues)
    A0, k1asa2s, errors = XYZ2A0(xs, ys, zs), np.linspace(k1asa2Min, k1asa2Max, options['nOfK1asa2']), []
    for k1asa2 in k1asa2s:
        cUs, rUs = CDRD2CURUForParabolicSquaredDistortion(cDs, rDs, oca, ora, k1asa2)
        A, b = np.concatenate((A0, CURUXYZ2A1(cUs, rUs, xs, ys, zs)), axis=1), CURU2B(cUs, rUs)
        Pa11 = AB2Pa11(A, b)
        cUsR, rUsR = XYZPa112CURU(xs, ys, zs, Pa11)
        errors.append(np.sqrt(np.mean((cUsR - cUs) ** 2 + (rUsR - rUs) ** 2)))
    k1asa2 = k1asa2s[np.argmin(np.asarray(errors))]
    return k1asa2
def GenerateRandomScaledVariables(dataBasic, variablesKeys, options={}): # 202109241335
    ''' comments:
    .- input dataBasic is a dictionary
    .- input variablesKeys is a string-list
    .- output scaledVariables is a float-ndarray
    '''
    keys, defaultValues = ['xc', 'yc', 'zc'], None
    options = CompleteADictionary(options, keys, defaultValues)
    variablesDictionary = {}
    for key in variablesKeys:
        if key in ['xc', 'yc', 'zc']: # dataBasic['referenceValuesDictionary'][key] is None
            value0 = options[key]
        else:
            value0 = dataBasic['referenceValuesDictionary'][key]
        variablesDictionary[key] = value0 + Random(-1., +1.) * dataBasic['referenceRangesDictionary'][key] # IMP*
        if key == 'zc':
            variablesDictionary[key] = max([variablesDictionary[key], options['zc'] / 2.])
    variables = Dictionary2Array(variablesKeys, variablesDictionary)
    scaledVariables = VariablesScaling(dataBasic, variables, variablesKeys, 'scale')
    return scaledVariables
def IntersectionOfTwoLines(line0, line1, options={}): # 202002291043 # *** 
    ''' comments:
    .- input lines are not required to be normalized
    .- output case is in ['point', 'coincident', 'parallel']
    .- output xIntersection and yIntersection are None if the lines are parallel
    .- output xIntersection and yIntersection is the point closest to the origin if the lines are coincident
    '''
    keys, defaultValues = ['epsilon'], [1.e-11]
    options = CompleteADictionary(options, keys, defaultValues)
    line0 = NormalizeALine(line0)
    line1 = NormalizeALine(line1)
    determinantT = + line0['lx'] * line1['ly'] - line0['ly'] * line1['lx']
    determinantX = - line0['lt'] * line1['ly'] + line0['ly'] * line1['lt']
    determinantY = - line0['lx'] * line1['lt'] + line0['lt'] * line1['lx']
    if np.abs(determinantT) > options['epsilon']: # point
        xIntersection, yIntersection = determinantX / determinantT, determinantY / determinantT 
        case = 'point'
    elif np.abs(determinantX) < options['epsilon'] and np.abs(determinantY) < options['epsilon']: # coincident
        xIntersection, yIntersection = PointInALineClosestToAPoint(line0, 0., 0.)
        case = 'coincident'
    else: # parallel
        xIntersection, yIntersection = None, None
        case = 'parallel'
    return xIntersection, yIntersection, case
def IsStringAnInteger(theString): # 202109201600
    ''' comments:
    .- input theString is a string
    .- output isStringAnInteger is a boolean
    '''
    try:
        int(theString)
        isStringAnInteger = True
    except:
        isStringAnInteger = False
    return isStringAnInteger
def IsVariablesOK(variables, variablesKeys): # 202109241432
    ''' comments:
    .- input variables is a float-ndarray
    .- input variablesKeys is a string-list
    .- output isVariablesOK is a boolean
    '''
    variablesDictionary = Array2Dictionary(variablesKeys, variables)
    isVariablesOK = True
    for key in variablesKeys:
        if key in ['zc', 'sca', 'sra']:
            isVariablesOK = isVariablesOK and variablesDictionary[key] > 0
        elif key == 'sg':
            isVariablesOK = isVariablesOK and np.abs(variablesDictionary[key]) <= np.pi / 2.
        elif key == 'ta':
            isVariablesOK = isVariablesOK and 0 <= variablesDictionary[key] < np.pi
    return isVariablesOK
def LoadDataBasic0(options={}): # 202109271434
    ''' comments:
    .- output data is a dictionary (does not include station-dependent information)
    '''
    keys, defaultValues = ['nc', 'nr', 'selectedVariablesKeys'], [4000, 3000, ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'sca']]
    options = CompleteADictionary(options, keys, defaultValues)
    data = {'date0OfTheWorld':'19000101000000000', 'date1OfTheWorld':'40000101000000000'}
    data['selectedVariablesKeys'] = options['selectedVariablesKeys']
    data['allVariablesKeys'] = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']
    assert set(['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'sca']) <= set(data['selectedVariablesKeys']) and set(data['selectedVariablesKeys']) <= set(data['allVariablesKeys'])
    data['referenceValuesDictionary'], data['referenceRangesDictionary'], data['scalesDictionary'] = {}, {}, {}
    data['referenceValuesDictionary']['xc'], data['referenceRangesDictionary']['xc'], data['scalesDictionary']['xc'] = None, 1.0e+1, 1.0e+1
    data['referenceValuesDictionary']['yc'], data['referenceRangesDictionary']['yc'], data['scalesDictionary']['yc'] = None, 1.0e+1, 1.0e+1
    data['referenceValuesDictionary']['zc'], data['referenceRangesDictionary']['zc'], data['scalesDictionary']['zc'] = None, 1.0e+1, 1.0e+1
    data['referenceValuesDictionary']['ph'], data['referenceRangesDictionary']['ph'], data['scalesDictionary']['ph'] = 0.*np.pi/2., np.pi/1., 1.0e+0
    data['referenceValuesDictionary']['sg'], data['referenceRangesDictionary']['sg'], data['scalesDictionary']['sg'] = 0.*np.pi/2., np.pi/4., 1.0e+0
    data['referenceValuesDictionary']['ta'], data['referenceRangesDictionary']['ta'], data['scalesDictionary']['ta'] = 1.*np.pi/2., np.pi/2., 1.0e+0 # IMP*
    data['referenceValuesDictionary']['k1a'], data['referenceRangesDictionary']['k1a'], data['scalesDictionary']['k1a'] = 0.0e+0, 1.0e+0, 1.e-1
    data['referenceValuesDictionary']['k2a'], data['referenceRangesDictionary']['k2a'], data['scalesDictionary']['k2a'] = 0.0e+0, 1.0e+0, 1.e-0
    data['referenceValuesDictionary']['p1a'], data['referenceRangesDictionary']['p1a'], data['scalesDictionary']['p1a'] = 0.0e+0, 1.0e-2, 1.e-2
    data['referenceValuesDictionary']['p2a'], data['referenceRangesDictionary']['p2a'], data['scalesDictionary']['p2a'] = 0.0e+0, 1.0e-2, 1.e-2
    data['referenceValuesDictionary']['sca'], data['referenceRangesDictionary']['sca'], data['scalesDictionary']['sca'] = 1.0e-3, 1.0e-3, 1.e-4
    data['referenceValuesDictionary']['sra'], data['referenceRangesDictionary']['sra'], data['scalesDictionary']['sra'] = 1.0e-3, 1.0e-3, 1.e-4
    data['referenceValuesDictionary']['oc'], data['referenceRangesDictionary']['oc'], data['scalesDictionary']['oc'] = options['nc']/2., options['nc']/20., options['nc']/10.
    data['referenceValuesDictionary']['or'], data['referenceRangesDictionary']['or'], data['scalesDictionary']['or'] = options['nr']/2., options['nr']/20., options['nr']/10.
    data['orderOfTheHorizonPolynomial'], data['radiusOfEarth'] = 5, 6.371e+6 # IMP*
    return data
def LoadDataPdfTxt(options={}): # 202109101200 # *** 
    ''' comments:
    .- output data is a dictionary
    .- the number of pixels in each direction is nOfPixels = length * ppm + 1
    '''
    def Length2NOfPixels(length, ppm): # 202110191119
        ''' comments:
        .- input length is a float
        .- input ppm is a float
        .- output nOfPixels is an integer
        '''
        nOfPixels = int(np.round(length * ppm)) + 1 # IMP*
        return nOfPixels
    def NOfPixels2Length(nOfPixels, ppm): # 202110191119
        ''' comments:
        .- input nOfPixels is an integer
        .- input ppm is a float
        .- output length is a float
        '''
        length = (nOfPixels - 1.) / ppm # IMP*
        return length
    keys, defaultValues = ['dataBasic', 'pathFile', 'planview', 'rewrite'], [None, None, None, False]
    options = CompleteADictionary(options, keys, defaultValues)
    keys, defaultValues = ['xUpperLeft', 'yUpperLeft', 'angle', 'xYLengthInC', 'xYLengthInR', 'ppm', 'timedeltaTolerance'], [None, None, None, None, None, None, datetime.timedelta(hours = 1.)]
    options = CompleteADictionary(options, keys, defaultValues)
    if all([options[item] is not None for item in ['xUpperLeft', 'yUpperLeft', 'angle', 'xYLengthInC', 'xYLengthInR', 'ppm', 'timedeltaTolerance']]):
        data = {key:options[key] for key in ['xUpperLeft', 'yUpperLeft', 'angle', 'xYLengthInC', 'xYLengthInR', 'ppm', 'timedeltaTolerance']}
        pathFile, options['rewrite'] = None, False
    else:
        if options['pathFile'] is not None:
            pathFile = options['pathFile']
        else:
            pathFile = options['dataBasic']['pathPlanviews'] + os.sep + options['dataBasic']['station'] + options['dataBasic']['date0OfTheWorld'] + 'pdf' + options['planview'] + '.txt' # WATCH OUT
        data = {'pathFile':pathFile}
        if pathFile is not None:
            rawData = ReadRectangleFromTxt(pathFile, {'c1':1, 'valueType':'float'})
            data = {'timedeltaTolerance':datetime.timedelta(hours = rawData[6])}
            data['xUpperLeft'], data['yUpperLeft'], data['angle'] = rawData[0], rawData[1], rawData[2] * np.pi / 180. # 0 = E, 90 = N, IMP*
            data['xYLengthInC'], data['xYLengthInR'], data['ppm'] = rawData[3], rawData[4], rawData[5]
    data['nc'], data['nr'] = Length2NOfPixels(data['xYLengthInC'], data['ppm']), Length2NOfPixels(data['xYLengthInR'], data['ppm'])
    data['xYLengthInC'], data['xYLengthInR'] = NOfPixels2Length(data['nc'], data['ppm']), NOfPixels2Length(data['nr'], data['ppm'])
    assert np.min([data['nc'], data['nr']]) > 5
    data['csBasic'], data['rsBasic'] = np.arange(0., data['nc']), np.arange(0., data['nr'])
    data['nOfPixels'] = data['nc'] * data['nr']
    mcs, mrs = np.meshgrid(data['csBasic'], data['rsBasic'])
    assert mcs.shape == mrs.shape == (data['nr'], data['nc'])
    data['cs'], data['rs'] = Matrix2Array(mcs), Matrix2Array(mrs)
    data['xs'], data['ys'] = CR2XY(data, data['cs'], data['rs'])[0:2]
    xsU, ysU = CR2XY(data, data['csBasic'], data['rsBasic'][+0] * np.ones(data['csBasic'].shape))[0:2] #! up
    xsD, ysD = CR2XY(data, data['csBasic'], data['rsBasic'][-1] * np.ones(data['csBasic'].shape))[0:2] #! down
    xs0, ys0 = CR2XY(data, data['csBasic'][+0] * np.ones(data['rsBasic'].shape), data['rsBasic'])[0:2] #! left
    xs1, ys1 = CR2XY(data, data['csBasic'][-1] * np.ones(data['rsBasic'].shape), data['rsBasic'])[0:2] #! right
    data['polylineU'], data['polylineD'] = {'xs':xsU, 'ys':ysU}, {'xs':xsD, 'ys':ysD}
    data['polyline0'], data['polyline1'] = {'xs':xs0, 'ys':ys0}, {'xs':xs1, 'ys':ys1}
    csC, rsC = np.asarray([0, data['nc']-1, 0, data['nc']-1]), np.asarray([0, 0, data['nr']-1, data['nr']-1])
    xsC, ysC = CR2XY(data, csC, rsC)[0:2]
    data['csC'], data['rsC'], data['xsC'], data['ysC'] = csC, rsC, xsC, ysC
    if options['rewrite']:
        WriteDataPdfTxt(data)
    return data
def MakeFolder(pathFolder): # 202109131100 # *** 
    ''' comments:
    .- input pathFolder is a string
    .- creates a folder if pathFolder does not exist
    '''
    if not os.path.exists(pathFolder):
        os.makedirs(pathFolder)
    return None
def Matrix2Array(M, options={}): # 202109141800 # *** 
    ''' comments:
    .- input M is float-2d-ndarray
    .- output A is a float-ndarray
    '''
    keys, defaultValues = ['way'], ['byRows']
    options = CompleteADictionary(options, keys, defaultValues)
    ''' comments:
    .- if options['way'] is 'byRows', one row after another
    .- if options['way'] is 'bColumns', one column after another
    '''
    assert options['way'] in ['byRows', 'byColumns']
    if options['way'] == 'byRows':
        A = np.reshape(M, -1)
    else:
        A = np.reshape(np.transpose(M), -1)
    (nr, nc), optionsTMP = M.shape[0:2], {key:options[key] for key in ['way']}
    assert np.allclose(M, Array2Matrix(A, nc, nr, options=optionsTMP))
    return A
def N2K(n): # 202109131100 # *** 
    ''' comments:
    .- input n is an integer or float
    .- output k is a float
    '''
    k = (n - 1.) / 2.
    return k
def NForRANSAC(eRANSAC, pRANSAC, sRANSAC): # *** 
    ''' comments:
    .- input eRANSAC is a float (probability of a point being "bad")
    .- input pRANSAC is a float (goal probability of sRANSAC points being "good")
    .- input sRANSAC is an integer (number of points of the model)
    .- note that: (1 - e) ** s is the probability of a set of s points being good
    .- note that: 1 - (1 - e) ** s is the probability of a set of s points being bad (at least one is bad)
    .- note that: (1 - (1 - e) ** s) ** N is the probability of choosing N sets all being bad
    .- note that: 1 - (1 - (1 - e) ** s) ** N is the probability of choosing N sets where at least one set if good
    .- note that: from 1 - (1 - (1 - e) ** s) ** N = p -> 1 - p = (1 - (1 - e) ** s) ** N and we get the expression
    '''
    N = int(np.log(1. - pRANSAC) / np.log(1. - (1. - eRANSAC) ** sRANSAC)) + 1
    return N
def NonlinearManualCalibration(dataBasic, dataForCal, subsetVariablesKeys, subCsetVariablesDictionary, options={}): # 202109010000 # *** 
    ''' comments:
    .- input dataBasic is a dictionary
    .- input dataForCal is a dictionary (including at least 'nc', 'nr', 'cs', 'rs', 'xs', 'ys', 'zs', 'aG')
        .- optional relevant keys: 'mainSetSeeds', 'chs', 'rhs', 'aH', 'xc', 'yc', 'zc' and 'aC' (check the last 4)
    .- input subsetVariablesKeys is a string-list
    .- input subCsetVariablesDictionary is a dictionary
    .- output mainSet is a dictionary or None (if it does not succeed)
    .- output errorT is a float or None (if it does not succeed)
    '''
    keys, defaultValues = ['timedelta', 'verbose', 'xc', 'yc', 'zc'], [datetime.timedelta(seconds=100.), False, None, None, None]
    options = CompleteADictionary(options, keys, defaultValues)
    if len(dataForCal['xs']) < int((len(subsetVariablesKeys) + 1.) / 2.):
        return None, None
    imgDiagonal = np.sqrt(dataForCal['nc'] ** 2 + dataForCal['nr'] ** 2)
    if 'mainSetSeeds' in dataForCal.keys() and dataForCal['mainSetSeeds'] is not None and len(dataForCal['mainSetSeeds']) > 0:
        mainSetSeed0, errorTSeed0 = ReadAFirstSeed(dataBasic, dataForCal, subsetVariablesKeys, subCsetVariablesDictionary, dataForCal['mainSetSeeds'])
        if options['verbose']:
            print('... NonlinearManualCalibration: seed provided with errorT = {:9.3f}'.format(errorTSeed0))
    else:
        mainSetSeed0, errorTSeed0 = None, 1.e+11 # IMP* the same values as in ReadAFirstSeed
        if options['verbose']:
            print('... NonlinearManualCalibration: no seed provided')
    if mainSetSeed0 is not None and errorTSeed0 < 0.2 * imgDiagonal: # IMP* WATCH OUT
        mainSetSeed, errorTSeed = [copy.deepcopy(item) for item in [mainSetSeed0, errorTSeed0]]
    else:
        optionsTMP = {key:options[key] for key in ['timedelta', 'xc', 'yc', 'zc']}
        mainSetSeed, errorTSeed = FindAFirstSeed(dataBasic, dataForCal, subsetVariablesKeys, subCsetVariablesDictionary, options=optionsTMP)
        if options['verbose']:
            print('... NonlinearManualCalibration: seed obtained with errorT = {:9.3f}'.format(errorTSeed))
    subsetVariablesSeed = AllVariables2SubsetVariables(dataBasic, mainSetSeed['allVariables'], subsetVariablesKeys)
    scaledSubsetVariablesSeed = VariablesScaling(dataBasic, subsetVariablesSeed, subsetVariablesKeys, 'scale')
    perturbationFactor, nOfSeeds = ErrorT2PerturbationFactorAndNOfSeeds(errorTSeed)
    (nc, nr), optionsHL = (dataForCal['nc'], dataForCal['nr']), {key:dataBasic[key] for key in ['orderOfTheHorizonPolynomial', 'radiusOfEarth']}
    theArgs = {'dataBasic':dataBasic, 'dataForCal':dataForCal, 'subsetVariablesKeys':subsetVariablesKeys, 'subCsetVariablesDictionary':subCsetVariablesDictionary}
    mainSetO, errorTO, scaledSubsetVariablesO = [copy.deepcopy(item) for item in [mainSetSeed, errorTSeed, scaledSubsetVariablesSeed]]
    for iOfSeeds in range(nOfSeeds): # monteCarlo
        if iOfSeeds == 0: # IMP* to ensure that the read 
            perturbationFactorH = 0.
        else:
            perturbationFactorH = 1. * perturbationFactor
        scaledSubsetVariablesP = PerturbateScaledVariables(dataBasic, scaledSubsetVariablesO, subsetVariablesKeys, options={'perturbationFactor':perturbationFactorH})
        try: # IMP* to try
            errorTP = ScaledSubsetVariables2FTM(scaledSubsetVariablesP, theArgs)
        except: # IMP* not to inform
            continue
        if errorTP >= 1.0 * imgDiagonal: # IMP* WATCH OUT
            continue
        try:
            scaledSubsetVariablesP = optimize.minimize(ScaledSubsetVariables2FTM, scaledSubsetVariablesP, args=(theArgs), callback=MinimizeStopper(5.)).x # IMP*
            errorTP = ScaledSubsetVariables2FTM(scaledSubsetVariablesP, theArgs)
        except:
            continue
        if errorTP < errorTO:
            if not IsVariablesOK(scaledSubsetVariablesP, subsetVariablesKeys):
                continue
            mainSetP = ScaledSubsetVariables2MainSet(dataBasic, scaledSubsetVariablesP, subsetVariablesKeys, subCsetVariablesDictionary, nc, nr, options=optionsHL)
            assert np.isclose(errorTP, ErrorT(dataForCal, mainSetP, options={'verbose':False})) # avoidable
            xs, ys, zs = dataForCal['xs'], dataForCal['ys'], dataForCal['zs']
            if not (len(XYZ2PositionsRightSideOfCamera(mainSetP, xs, ys, zs)) == len(xs) == len(ys) == len(zs)):
                continue
            mainSetO, errorTO, scaledSubsetVariablesO = [copy.deepcopy(item) for item in [mainSetP, errorTP, scaledSubsetVariablesP]]
            perturbationFactor = ErrorT2PerturbationFactorAndNOfSeeds(errorTO)[0]
            if options['verbose']:
                print('... improvement in iteration {:3}, errorT = {:9.3f}'.format(iOfSeeds+1, errorTO))
        if errorTO < 0.1: # IMP* interesting for RANSAC WATCH OUT
            break
    return mainSetO, errorTO
def NonlinearManualCalibrationForcingUniqueSubCset(dataBasic, ncs, nrs, css, rss, xss, yss, zss, chss, rhss, subsetVariabless, subsetVariablesKeys, subCsetVariabless, subCsetVariablesKeys, options={}): # *** 
    ''' comments:
    .- input dataBasic is a dictionary
    .- input ncs and nrs are integer- or float-lists
    .- input css, rss, xss, yss, zss, chss, rhss, subsetVariabless and subCsetVariabless are float-ndarrays-lists (free calibrations are provided)
    .- input subsetVariablesKeys is a string-list
    .- input subCsetVariablesKeys is a string-list
    .- output mainSetsO is a dictionary
    .- output errorTsO is a float
    '''
    keys, defaultValues = ['aG', 'aH', 'nOfSeedsForC', 'orderOfTheHorizonPolynomial', 'radiusOfEarth'], [1., 1., 20, 5, 6.371e+6]
    options = CompleteADictionary(options, keys, defaultValues)
    subCsetVariables0 = np.average(np.asarray(subCsetVariabless), axis=0)
    assert subCsetVariables0.shape == subCsetVariabless[0].shape # avoidable
    ctrlContinue, ctrlFirst, mainSetsO, errorTsO = True, True, [{} for item in range(len(css))], np.zeros(len(css))
    while ctrlContinue:
        print('... iterating (to obtain unique parameters) ...')
        if ctrlFirst: 
            subCsetVariables = copy.deepcopy(subCsetVariables0)
            mainSets, errorTs = [{} for item in range(len(css))], np.zeros(len(css))
            optionsTMP = {key:options[key] for key in ['orderOfTheHorizonPolynomial', 'radiusOfEarth']}
            for pos in range(len(css)):
                subCsetVariablesDictionary = Array2Dictionary(subCsetVariablesKeys, subCsetVariables0) # move up
                allVariables = SubsetVariables2AllVariables(dataBasic, subsetVariabless[pos], subsetVariablesKeys, subCsetVariablesDictionary, options={'nc':ncs[pos], 'nr':nrs[pos]})
                mainSet = AllVariables2MainSet(allVariables, ncs[pos], nrs[pos], options=optionsTMP)
                dataForCal = {}
                dataForCal['cs'], dataForCal['rs'] = css[pos], rss[pos]
                dataForCal['xs'], dataForCal['ys'], dataForCal['zs'] = xss[pos], yss[pos], zss[pos]
                dataForCal['aG'] = options['aG']
                if all([item is not None and len(item) > 0 for item in [chss[pos], rhss[pos]]]) and options['aH'] > 1.e-8:
                    dataForCal['chs'], dataForCal['rhs'] = chss[pos], rhss[pos]
                    dataForCal['aH'] = options['aH']
                errorT = ErrorT(dataForCal, mainSet)
                mainSets[pos], errorTs[pos] = mainSet, errorT
            ctrlFirst = False
        optionsTMP = {key:options[key] for key in ['aG', 'aH', 'nOfSeedsForC', 'orderOfTheHorizonPolynomial', 'radiusOfEarth']}
        subCsetVariables, errorT = UpdateSubCsetVariables(dataBasic, ncs, nrs, css, rss, xss, yss, zss, chss, rhss, subsetVariabless, subsetVariablesKeys, subCsetVariables0, subCsetVariablesKeys, options=optionsTMP) # ojo aqui subCsetVariables0
        subCsetVariablesDictionary = Array2Dictionary(subCsetVariablesKeys, subCsetVariables)
        errorTsOld = copy.deepcopy(errorTs)
        for pos in range(len(css)): # IMPROVABLE
            dataForCal = {}
            dataForCal['nc'], dataForCal['nr'] = ncs[pos], nrs[pos]
            dataForCal['cs'], dataForCal['rs'] = css[pos], rss[pos]
            dataForCal['xs'], dataForCal['ys'], dataForCal['zs'] = xss[pos], yss[pos], zss[pos]
            dataForCal['aG'] = options['aG']
            if all([item is not None and len(item) > 0 for item in [chss[pos], rhss[pos]]]) and options['aH'] > 1.e-8:
                dataForCal['chs'], dataForCal['rhs'] = chss[pos], rhss[pos]
                dataForCal['aH'] = options['aH']
            dataForCal['mainSetSeeds'] = mainSets
            mainSet, errorT = NonlinearManualCalibration(dataBasic, dataForCal, subsetVariablesKeys, subCsetVariablesDictionary, options={}) 
            subsetVariabless[pos], mainSets[pos], errorTs[pos] = AllVariables2SubsetVariables(dataBasic, mainSet['allVariables'], subsetVariablesKeys, options={}), mainSet, errorT
        if max(errorTs) < max(errorTsOld) * 0.999:
            mainSetsO, errorTsO = mainSets, errorTs
        else:
            ctrlContinue = False
    return mainSetsO, errorTsO
def NormalizeALine(line, options={}): # 202109280946
    ''' comments:
    .- input line is a dictionary (including at least 'lx', 'ly' and 'lt')
        .- a line is so that line['lx'] * x + line['ly'] * y + line['lt'] = 0
        .- a normalized line is so that line['lx'] ** 2 + line['ly'] ** 2 = 1
    .- output line includes key 'isNormalized' (=True)
    .- output line maintains the orientation
    '''
    keys, defaultValues = ['forceToNormalize'], [False]
    options = CompleteADictionary(options, keys, defaultValues)
    ''' comments:
    .- if not options['forceToNormalize'] it normalizes only if necessary (ie, if not isNormalized)
    '''
    if options['forceToNormalize'] or 'isNormalized' not in line.keys() or not line['isNormalized']:
        lm = np.sqrt(line['lx'] ** 2 + line['ly'] ** 2) # > 0
        line = {item:line[item] / lm for item in ['lx', 'ly', 'lt']}
        line['isNormalized'] = True
    return line
def ORBKeypoints(img, options={}): # 202109221400
    ''' comments:
    .- input img is an image or its path
    .- output nc and nr are integers or Nones (if it does not succeed)
    .- output kps are ORB keypoints or None (if it does not succeed)
    .- output des are ORB descriptions or None (if it does not succeed)
    .- output ctrl is a boolean (False if it does not succeed)
    '''
    keys, defaultValues = ['mask', 'nOfFeatures'], [None, 5000]
    options = CompleteADictionary(options, keys, defaultValues)
    try:
        img = PathImgOrImg2Img(img)
        nr, nc = img.shape[0:2]
        orb = cv2.ORB_create(nfeatures=options['nOfFeatures'], scoreType=cv2.ORB_FAST_SCORE)
        if options['mask'] is not None:
            kps, des = orb.detectAndCompute(img, mask=options['mask'])
        else:
            kps, des = orb.detectAndCompute(img, None)
        assert len(kps) == len(des) > 0
        ctrl = True
    except:
        nc, nr, kps, des, ctrl = None, None, None, None, False
    return nc, nr, kps, des, ctrl
def ORBKeypointsForAllImagesInAFolder(pathFolder, options={}): # 202202011457
    ''' comments:
    .- input pathFolder is a string
    .- output fnsImages, ncs, nrs, kpss ands dess are lists
    '''
    keys, defaultValues = ['mask', 'nOfFeatures'], [None, 5000]
    options = CompleteADictionary(options, keys, defaultValues)
    pathsImages = [os.path.join(pathFolder, item) for item in os.listdir(pathFolder) if item[item.rfind('.')+1:] in ['jpg', 'jpeg', 'png']] # IMP*
    fnsImages, ncs, nrs, kpss, dess = ORBKeypointsForPathsImages(pathsImages, options={'mask':options['mask'], 'nOfFeatures':options['nOfFeatures']})
    return fnsImages, ncs, nrs, kpss, dess
def ORBKeypointsForPathsImages(pathsImages, options={}): # 202202011455
    ''' comments:
    .- input pathsImages is a list of strings
    .- output fnsImages, ncs, nrs, kpss and dess are lists (including Nones whe it does not succeed)
    '''
    keys, defaultValues = ['mask', 'nOfFeatures'], [None, 5000]
    options = CompleteADictionary(options, keys, defaultValues)
    fnsImages, ncs, nrs, kpss, dess = [[] for item in range(5)]
    for pathImage in pathsImages:
        fnsImages.append(os.path.split(pathImage)[1])
        optionsTMP = {key:options[key] for key in ['mask', 'nOfFeatures']}
        nc, nr, kps, des, ctrl = ORBKeypoints(pathImage, options=optionsTMP)
        ncs.append(nc); nrs.append(nr); kpss.append(kps); dess.append(des)
    return fnsImages, ncs, nrs, kpss, dess
def ORBMatches(kps1, des1, kps2, des2, options={}): # 202109131700 # *** 
    ''' comments:
    .- input kps1 and des1 are ORB keys and descriptions for image1 (see ORBKeypoints)
    .- input kps2 and des2 are ORB keys and descriptions for image2 (see ORBKeypoints)
    .- output cs1, rs1, cs2, rs2 and ers are float-ndarrays
    .- compares both ways, so that commutativity is ensured
    '''
    keys, defaultValues = ['erMaximum', 'nOfStd'], [20., 2.]
    options = CompleteADictionary(options, keys, defaultValues)
    cs1, rs1 = np.asarray([item.pt[0] for item in kps1]), np.asarray([item.pt[1] for item in kps1])
    cs2, rs2 = np.asarray([item.pt[0] for item in kps2]), np.asarray([item.pt[1] for item in kps2])
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches12 = sorted(bf.match(des1, des2), key = lambda x:x.distance)
    matches21 = sorted(bf.match(des2, des1), key = lambda x:x.distance)
    poss1 = [match.queryIdx for match in matches12] + [match.trainIdx for match in matches21]
    poss2 = [match.trainIdx for match in matches12] + [match.queryIdx for match in matches21]
    cs1, rs1, cs2, rs2 = cs1[poss1], rs1[poss1], cs2[poss2], rs2[poss2]
    ers = np.asarray([match.distance for match in matches12] + [match.distance for match in matches21])
    assert len(cs1) == len(rs1) == len(cs2) == len(rs2) == len(ers)
    cs1, rs1, cs2, rs2, ers = np.unique(np.asarray([cs1, rs1, cs2, rs2, ers]), axis=1)
    dps = np.sqrt((cs1 - cs2) ** 2 + (rs1 - rs2) ** 2)
    possGood = np.where((ers < options['erMaximum']) & (dps < np.mean(dps) + options['nOfStd'] * np.std(dps) + 1.e-8))[0]
    cs1, rs1, cs2, rs2, ers = [item[possGood] for item in [cs1, rs1, cs2, rs2, ers]]
    return cs1, rs1, cs2, rs2, ers
def PathImgOrImg2Img(img): # 202110041642
    ''' comments:
    .- input img is a cv2-image or a string
    .- output img is a cv2-image
    '''
    try:
        nr, nc = img.shape[0:2]
    except:
        img = cv2.imread(img)
        nr, nc = img.shape[0:2]
    img[nr-1, nc-1, 0]
    return img
def PerturbateScaledVariables(dataBasic, scaledVariables, variablesKeys, options={}): # 202109241340
    ''' comments:
    .- input dataBasic is a dictionary
    .- input scaledVariables is a float-ndarray
    .- input variablesKeys is a string-list
    .- output scaledVariables is a float-ndarray
    '''
    keys, defaultValues = ['perturbationFactor'], [1.]
    options = CompleteADictionary(options, keys, defaultValues)
    variables = VariablesScaling(dataBasic, scaledVariables, variablesKeys, 'unscale')
    variablesDictionary = Array2Dictionary(variablesKeys, variables)
    for key in variablesKeys:
        variablesDictionary[key] = variablesDictionary[key] + options['perturbationFactor'] * Random(-1., +1.) * dataBasic['scalesDictionary'][key] # IMP*
    variables = Dictionary2Array(variablesKeys, variablesDictionary)
    scaledVariables = VariablesScaling(dataBasic, variables, variablesKeys, 'scale')
    return scaledVariables
def PixelsErrorOfRotationalHomographyUsingUVUas(x, theArgs): # 190001010000 # *** 
    ''' comments:
    .- input x is a 3-float-ndarray (ph, sg and ta)
    .- input theArgs is a dictionary
    '''
    R1, uUas0, vUas0, uUas1, vUas1, sca = [theArgs[item] for item in ['R1', 'uUas0', 'vUas0', 'uUas1', 'vUas1', 'sca']]
    R0 = EulerianAngles2R(x[0], x[1], x[2])
    H01 = np.dot(R1, np.transpose(R0)) # 0 is unknown, 1 is known
    uUas1R, vUas1R = ApplyHomographyHa01(H01, uUas0, vUas0)
    f = np.sqrt(np.mean((uUas1R - uUas1) ** 2 + (vUas1R - vUas1) ** 2)) / sca
    return f
def PlanviewPrecomputations(mainSets, dataPdfTxt, z): # 202109101200 # *** 
    ''' comments:
    .- input mainSets is a dictionary of dictionaries (the keys are the cameras)
    .- input dataPdfTxt is a dictionary
    '''
    assert all([len(item) == 2 and IsStringAnInteger(item) for item in mainSets.keys()])
    planviewPrecomputations = {key:dataPdfTxt[key] for key in ['nc', 'nr', 'cs', 'rs']}
    planviewPrecomputations['cameras'] = []
    for camera in sorted(mainSets.keys()):
        ncCamera, nrCamera, mainSet = mainSets[camera]['nc'], mainSets[camera]['nr'], mainSets[camera]
        xs, ys, zs = dataPdfTxt['xs'], dataPdfTxt['ys'], z * np.ones(dataPdfTxt['nOfPixels'])
        optionsTMP = {'imgMargins':{'c0':2, 'c1':2, 'r0':2, 'r1':2, 'isComplete':True}, 'returnGoodPositions':True}
        csCamera, rsCamera, planviewPositionsInCamera = XYZ2CDRD(mainSet, xs, ys, zs, options=optionsTMP)
        nOfplanviewPositionsInCamera = len(planviewPositionsInCamera)
        if nOfplanviewPositionsInCamera == 0:
            continue
        csCamera, rsCamera = csCamera[planviewPositionsInCamera], rsCamera[planviewPositionsInCamera]
        planviewPrecomputations['cameras'].append(camera)
        planviewPrecomputations[camera] = {}
        planviewPrecomputations[camera]['planviewPositions'] = planviewPositionsInCamera  # nOfplanviewPositionsInCamera
        aux0s = CRWithinImage2NormalizedLengthsAndAreas(ncCamera, nrCamera, csCamera, rsCamera)[0]
        planviewPrecomputations[camera]['ws'] = aux0s # nOfplanviewPositionsInCamera x 1
        csCameraIA, rsCameraIA, wsCameraA = CR2CRIntegerAroundAndWeights(csCamera, rsCamera)
        planviewPrecomputations[camera]['csIA'] = csCameraIA # nOfplanviewPositionsInCamera x 4
        planviewPrecomputations[camera]['rsIA'] = rsCameraIA # nOfplanviewPositionsInCamera x 4
        planviewPrecomputations[camera]['wsA1'] = wsCameraA # nOfplanviewPositionsInCamera x 4
    return planviewPrecomputations
def PlotMainSet(img, mainSet, cs, rs, xs, ys, zs, chs, rhs, pathImgOut): # 202111171719
    ''' comments:
    .- input img is a cv2-image or a string
    .- input mainSet is a dictionary (including at least 'horizonLine')
    .- input cs, rs, xs, ys and zs are float-ndarrays of the same length
    .- input chs and rhs are float-ndarrays or None
    .- input pathImgOut is a string
    '''
    img = PathImgOrImg2Img(img)
    nr, nc = img.shape[0:2]
    img = DisplayCRInImage(img, cs, rs, options={'colors':[[0, 0, 0]], 'size':np.sqrt(nc*nr)/200})
    csR, rsR = XYZ2CDRD(mainSet, xs, ys, zs, options={})[0:2] #!
    img = DisplayCRInImage(img, csR, rsR, options={'colors':[[0, 255, 255]], 'size':np.sqrt(nc*nr)/400})
    chsR, rhsR = np.arange(0, nc, 1), CDh2RDh(mainSet['horizonLine'], np.arange(0, nc, 1), options={})[0] #!
    img = DisplayCRInImage(img, chsR, rhsR, options={'colors':[[0, 255, 255]], 'size':1})
    if chs is not None and rhs is not None:
        img = DisplayCRInImage(img, chs, rhs, options={'colors':[[0, 0, 0]], 'size':np.sqrt(nc*nr)/200})
    cv2.imwrite(pathImgOut, img)
    return None
def PointInALineClosestToAPoint(line, x, y): # 202002291043 # *** 
    ''' comments:
    .- input line required keys: 'lx', 'ly', 'lt'
    .- input line is not required to be normalized
    .- input x and y can be scalars or arrays (of the same length)
    .- output xClosest and yClosest are scalars or arrays
    '''
    line = NormalizeALine(line)
    xClosest = line['ly'] * (line['ly'] * x - line['lx'] * y) - line['lx'] * line['lt']
    yClosest = line['lx'] * (line['lx'] * y - line['ly'] * x) - line['ly'] * line['lt']
    return xClosest, yClosest
def Poss0AndPoss1(n): # 202201250804
    '''
    .- input n is an integer
    .- output poss0 and poss1 are n-integer-list
    '''
    poss0 = [2*pos+0 for pos in range(n)]
    poss1 = [2*pos+1 for pos in range(n)]
    return poss0, poss1
def Poss0AndPoss1InFind2DTransform(n): # *** 
    poss0 = [2*pos+0 for pos in range(n)]
    poss1 = [2*pos+1 for pos in range(n)]
    return poss0, poss1
def R2UnitVectors(R): # 202109131100 # *** 
    ''' comments:
    .- input R is a 3x3-float-ndarray
        .- the rows of R are eu, ev and ef
    .- output eu, ev and ef are 3-float-ndarrays
    '''
    assert R.shape == (3, 3)
    eu, ev, ef = R[0, :], R[1, :], R[2, :]
    return eu, ev, ef
def RANSACForGCPs(cDs, rDs, xs, ys, zs, oca, ora, eRANSAC, pRANSAC, ecRANSAC, NForRANSACMax, options={}): # *** 
    if len(cDs) < 6:
        return None, None
    keys, defaultValues = ['nOfK1asa2'], [1000]
    options = CompleteADictionary(options, keys, defaultValues)
    dD2Max = np.max((cDs - oca) ** 2 + (rDs - ora) ** 2)
    k1asa2 = 0.
    possGood = RANSACForGCPsAndK1asa2(cDs, rDs, xs, ys, zs, oca, ora, k1asa2, eRANSAC, pRANSAC, 3. * ecRANSAC, NForRANSACMax) # WATCH OUT
    cDsSel, rDsSel, xsSel, ysSel, zsSel = [item[possGood] for item in [cDs, rDs, xs, ys, zs]]
    k1asa2Min, k1asa2Max = -4./(27.*dD2Max)+1.e-11, 4./(27.*dD2Max) # WATCH OUT
    k1asa2 = GCPs2K1asa2(cDsSel, rDsSel, xsSel, ysSel, zsSel, oca, ora, k1asa2Min, k1asa2Max, options={'nOfK1asa2':options['nOfK1asa2']})
    possGood = RANSACForGCPsAndK1asa2(cDs, rDs, xs, ys, zs, oca, ora, k1asa2, eRANSAC, pRANSAC, ecRANSAC, NForRANSACMax)
    cDsSel, rDsSel, xsSel, ysSel, zsSel = [item[possGood] for item in [cDs, rDs, xs, ys, zs]] # departing from the original points
    k1asa2Min, k1asa2Max = -4./(27.*dD2Max)+1.e-11, 4./(27.*dD2Max) # WATCH OUT
    k1asa2 = GCPs2K1asa2(cDsSel, rDsSel, xsSel, ysSel, zsSel, oca, ora, k1asa2Min, k1asa2Max, options={'nOfK1asa2':options['nOfK1asa2']})
    return possGood, k1asa2
def RANSACForGCPsAndK1asa2(cDs, rDs, xs, ys, zs, oca, ora, k1asa2, eRANSAC, pRANSAC, ecRANSAC, NForRANSACMax): # *** 
    cUs, rUs = CDRD2CURUForParabolicSquaredDistortion(cDs, rDs, oca, ora, k1asa2)
    AAll, bAll = CURUXYZ2A(cUs, rUs, xs, ys, zs), CURU2B(cUs, rUs)
    sRANSAC = 6 #  (to obtain 12 equations >= 11 unkowns)
    N, possGood = min(NForRANSACMax, NForRANSAC(eRANSAC, pRANSAC, sRANSAC)), []
    for iN in range(N):
        possH = random.sample(range(0, len(cUs)), sRANSAC)
        poss01 = [2*item for item in possH] + [2*item+1 for item in possH] # who cares about the order (both A and b suffer the same)
        A, b = AAll[poss01, :], bAll[poss01]
        try:
            Pa11 = AB2Pa11(A, b)
            cUsR, rUsR = XYZPa112CURU(xs, ys, zs, Pa11) # all positions
            errors = np.sqrt((cUsR - cUs) ** 2 + (rUsR - rUs) ** 2)
        except:
            continue
        possGoodH = np.where(errors <= ecRANSAC)[0]
        if len(possGoodH) > len(possGood):
            possGood = copy.deepcopy(possGoodH)
    return possGood
def Random(value0, value1, options={}): # 202109131700 # *** 
    ''' comments:
    .- input value0 is a float
    .- input value1 is a float
    .- output randomValues is a float or a float-ndarray
    '''
    keys, defaultValues = ['shape'], None
    options = CompleteADictionary(options, keys, defaultValues)
    ''' comments:
    .- options['shape'] is an integer, a shape or None (if the output is to be a float)
    '''
    if options['shape'] is None:
        randomValues = value0 + (value1 - value0) * np.random.random()
    else:
        randomValues = value0 + (value1 - value0) * np.random.random(options['shape'])
    return randomValues
def ReadAFirstSeed(dataBasic, dataForCal, subsetVariablesKeys, subCsetVariablesDictionary, mainSetSeeds): # 202109241556
    ''' comments:
    .- input dataBasic is a dictionary (including at least 'selectedVariablesKeys')
    .- input dataForCal is a dictionary
    .- input mainSetSeeds is a list of dictionaries
    .- output mainSetSeed is a dictionary or None
    .- output errorTSeed is a float
    '''
    mainSetSeedO, errorTSeedO = None, 1.e+11
    for mainSetSeed in mainSetSeeds:
        subsetVariables = AllVariables2SubsetVariables(dataBasic, mainSetSeed['allVariables'], subsetVariablesKeys)
        allVariables = SubsetVariables2AllVariables(dataBasic, subsetVariables, subsetVariablesKeys, subCsetVariablesDictionary, options={'nc':mainSetSeed['nc'], 'nr':mainSetSeed['nr']})
        if not np.allclose(allVariables, mainSetSeed['allVariables']):
            optionsTMP = {key:mainSetSeed[key] for key in ['orderOfTheHorizonPolynomial', 'radiusOfEarth']}
            mainSetSeed = AllVariables2MainSet(allVariables, mainSetSeed['nc'], mainSetSeed['nr'], options=optionsTMP)
        errorTSeed = ErrorT(dataForCal, mainSetSeed, options={})
        if errorTSeed < errorTSeedO:
            mainSetSeedO, errorTSeedO = [copy.deepcopy(item) for item in [mainSetSeed, errorTSeed]]
    mainSetSeed, errorTSeed = [copy.deepcopy(item) for item in [mainSetSeedO, errorTSeedO]]
    return mainSetSeed, errorTSeed
def ReadCalTxt(pathCalTxt): # 202110131422
    ''' comments:
    .- input pathCalTxt is a string
    .- output allVariables is a 14-float-ndarray
    .- output nc and nr are integers
    .- output errorT is a float
    '''
    rawData = np.asarray(ReadRectangleFromTxt(pathCalTxt, {'c1':1, 'valueType':'float'}))
    allVariables, nc, nr, errorT = rawData[0:14], int(np.round(rawData[14])), int(np.round(rawData[15])), rawData[16]
    return allVariables, nc, nr, errorT
def ReadCdgTxt(pathCdgTxt, options={}): # 202110051016
    ''' comments:
    .- input pathCdgTxt is a string
    .- output cs, rs, xs, ys and zs are float-ndarrays (that can be empty)
    .- output codes is a string-list or None
    '''
    keys, defaultValues = ['readCodes', 'readOnlyGood'], [False, True]
    options = CompleteADictionary(options, keys, defaultValues)
    rawData = np.asarray(ReadRectangleFromTxt(pathCdgTxt, {'c1':5, 'valueType':'float'}))
    if len(rawData) == 0: # exception required
        cs, rs, xs, ys, zs = [np.asarray([]) for item in range(5)]
    else:
        cs, rs, xs, ys, zs = [rawData[:, item] for item in range(5)]
        if options['readOnlyGood']: # disregards negative pixels
            possGood = np.where((cs >= 0.) & (rs >= 0.))[0]
            cs, rs, xs, ys, zs = [item[possGood] for item in [cs, rs, xs, ys, zs]]
    if options['readCodes']:
        codes = ReadRectangleFromTxt(pathCdgTxt, {'c0':5, 'c1':6, 'valueType':'str'}) # can be []
        if len(codes) > 0 and options['readOnlyGood']:
            codes = [codes[pos] for pos in possGood]
    else:
        codes = None
    return cs, rs, xs, ys, zs, codes
def ReadCdhTxt(pathCdhTxt, options={}): # 202110051054
    ''' comments:
    .- input pathCdhTxt is a string
    .- output chs and rhs are float-ndarrays (that can be empty)
    '''
    keys, defaultValues = ['readOnlyGood'], [True]
    options = CompleteADictionary(options, keys, defaultValues)
    rawData = np.asarray(ReadRectangleFromTxt(pathCdhTxt, {'c1':2, 'valueType':'float'}))
    if len(rawData) == 0: # exception required
        chs, rhs = [np.asarray([]) for item in range(2)]
    else:
        chs, rhs = [rawData[:, item] for item in range(2)]
        if options['readOnlyGood']: # disregards negative pixels
            possGood = np.where((chs >= 0.) & (rhs >= 0.))[0]
            chs, rhs = [item[possGood] for item in [chs, rhs]]
    return chs, rhs
def ReadRectangleFromTxt(pathFile, options={}): # 202109141200 # *** 
    assert os.path.isfile(pathFile)
    keys, defaultValues = ['c0', 'c1', 'r0', 'r1', 'valueType', 'nullLine'], [0, 0, 0, 0, 'str', None]
    options = CompleteADictionary(options, keys, defaultValues)
    openedFile = open(pathFile, 'r')
    listOfLines = openedFile.readlines()
    if options['nullLine'] is not None:
        listOfLines = [item for item in listOfLines if item[0] != options['nullLine']]
    if not (options['r0'] == 0 and options['r1'] == 0): # if r0 == r1 == 0 it loads all the rows
        listOfLines = [listOfLines[item] for item in range(options['r0'], options['r1'])]
    for posOfLine in range(len(listOfLines)-1, -1, -1):
        if listOfLines[posOfLine] == '\n':
            print('... line {:5} is empty'.format(posOfLine))
            del listOfLines[posOfLine]
    stringsInLines = [item.split() for item in listOfLines]
    rectangle = stringsInLines
    if not (options['c0'] == options['c1'] == 0): # if c0 == c1 == 0 it loads all the columns
        rectangle = [item[options['c0']:options['c1']] for item in rectangle]
    if options['valueType'] == 'str':
        pass
    elif options['valueType'] == 'float':
        rectangle = [[float(item) for item in line] for line in rectangle]
    elif options['valueType'] == 'int':
        rectangle = [[int(item) for item in line] for line in rectangle]
    else:
        assert False
    if options['c1'] - options['c0'] == 1: # one column
        rectangle = [item[0] for item in rectangle]
    if options['r1'] - options['r0'] == 1: # one row
        rectangle = rectangle[0]
    return rectangle
def ScaledSubCsetVariables2FTM(scaledSubCsetVariables, theArgs): # 202109271811
    ''' comments:
    .- input scaledSubCsetVariables is a float-ndarray
    .- input theArgs is a dictionary
    .- output errorT is a float (errorC is not included)
    '''
    dataBasic, ncs, nrs, css, rss, xss, yss, zss, aG, chss, rhss, aH, subsetVariabless, subsetVariablesKeys, subCsetVariablesKeys = [theArgs[item] for item in ['dataBasic', 'ncs', 'nrs', 'css', 'rss', 'xss', 'yss', 'zss', 'aG', 'chss', 'rhss', 'aH', 'subsetVariabless', 'subsetVariablesKeys', 'subCsetVariablesKeys']]
    subCsetVariables = VariablesScaling(dataBasic, scaledSubCsetVariables, subCsetVariablesKeys, 'unscale')
    subCsetVariablesDictionary = Array2Dictionary(subCsetVariablesKeys, subCsetVariables)
    errorT, optionsMainSet = 0., {key:theArgs[key] for key in ['orderOfTheHorizonPolynomial', 'radiusOfEarth']}
    for pos in range(len(css)):
        allVariables = SubsetVariables2AllVariables(dataBasic, subsetVariabless[pos], subsetVariablesKeys, subCsetVariablesDictionary, options={'nc':ncs[pos], 'nr':nrs[pos]})
        mainSet = AllVariables2MainSet(allVariables, ncs[pos], nrs[pos], options=optionsMainSet)
        dataForCal = {}
        dataForCal['cs'], dataForCal['rs'] = css[pos], rss[pos]
        dataForCal['xs'], dataForCal['ys'], dataForCal['zs'] = xss[pos], yss[pos], zss[pos]
        dataForCal['aG'] = aG
        if all([item is not None and len(item) > 0 for item in [chss[pos], rhss[pos]]]) and aH > 1.e-8:
            dataForCal['chs'], dataForCal['rhs'] = chss[pos], rhss[pos]
            dataForCal['aH'] = aH
        errorT = errorT + ErrorT(dataForCal, mainSet, options={'verbose':False})
    return errorT
def ScaledSubsetVariables2ErrorT(dataBasic, dataForCal, scaledSubsetVariables, subsetVariablesKeys, subCsetVariablesDictionary, options={}): # 202109241454
    ''' comments:
    .- input dataBasic is a dictionary
    .- input dataForCal is a dictionary (including at least 'nc' and 'nr')
    .- input scaledSubsetVariables is a float-ndarray
    .- input subsetVariablesKeys is a string-list
    .- input subCsetVariablesDictionary is a dictionary
    .- output errorT is a float
    '''
    keys, defaultValues = ['orderOfTheHorizonPolynomial', 'radiusOfEarth'], [5, 6.371e+6]
    options = CompleteADictionary(options, keys, defaultValues)
    (nc, nr), optionsTMP = (dataForCal['nc'], dataForCal['nr']), {key:options[key] for key in ['orderOfTheHorizonPolynomial', 'radiusOfEarth']}
    mainSet = ScaledSubsetVariables2MainSet(dataBasic, scaledSubsetVariables, subsetVariablesKeys, subCsetVariablesDictionary, nc, nr, options=optionsTMP)
    errorT = ErrorT(dataForCal, mainSet, options={'verbose':False})
    return errorT
def ScaledSubsetVariables2FTM(scaledSubsetVariables, theArgs): # IMP* FTM = Function To Minimize 202109241454
    ''' comments:
    .- input scaledVariables is a float-ndarray
    .- input theArgs is a dictionary (including at least 'dataBasic', 'dataForCal', 'subsetVariablesKeys' and 'subCsetVariablesDictionary'
    .- output errorT is a float
    '''
    dataBasic, dataForCal, subsetVariablesKeys, subCsetVariablesDictionary = [theArgs[key] for key in ['dataBasic', 'dataForCal', 'subsetVariablesKeys', 'subCsetVariablesDictionary']]
    optionsTMP = {key:dataBasic[key] for key in ['orderOfTheHorizonPolynomial', 'radiusOfEarth']}
    errorT = ScaledSubsetVariables2ErrorT(dataBasic, dataForCal, scaledSubsetVariables, subsetVariablesKeys, subCsetVariablesDictionary, options=optionsTMP)
    return errorT
def ScaledSubsetVariables2MainSet(dataBasic, scaledSubsetVariables, subsetVariablesKeys, subCsetVariablesDictionary, nc, nr, options={}): # *** 
    ''' comments:
    .- input dataBasic is a dictionary (including at least 'selectedVariablesKeys')
    .- input scaledSubsetVariables is a float-ndarray
    .- input subsetVariablesKeys is a string-list
    .- input subCsetVariablesDictionary is a dictionary
    .- input nc and nr are integers or floats
    .- output mainSet is a dictionary
    '''
    keys, defaultValues = ['orderOfTheHorizonPolynomial', 'radiusOfEarth'], [5, 6.371e+6]
    options = CompleteADictionary(options, keys, defaultValues)
    optionsTMP = {'nc':nc, 'nr':nr}
    subsetVariables = VariablesScaling(dataBasic, scaledSubsetVariables, subsetVariablesKeys, 'unscale')
    allVariables = SubsetVariables2AllVariables(dataBasic, subsetVariables, subsetVariablesKeys, subCsetVariablesDictionary, options=optionsTMP)
    optionsTMP = {key:options[key] for key in ['orderOfTheHorizonPolynomial', 'radiusOfEarth']}
    mainSet = AllVariables2MainSet(allVariables, nc, nr, options=optionsTMP)
    return mainSet
def SelectPixelsInGrid(nOfBands, nc, nr, cs, rs, es, options={}): # *** 
    ''' comments:
    .- input nOfBands is an integer
    .- input nc and nr are integers or floats
    .- input cs and rs are integers- or floats-ndarrays of the same length
    .- input es is a float-ndarrays of the same length as cs and rs
    .- output possSelected, bandCsSelected and bandRsSelected are integer-list or Nones (if it does not succeed)
    '''
    keys, defaultValues = ['nOfCBands', 'nOfRBands'], None
    options = CompleteADictionary(options, keys, defaultValues)
    if options['nOfCBands'] is None or options['nOfRBands'] is None:
        nOfCBands, nOfRBands = nOfBands, nOfBands
    else:
        nOfCBands, nOfRBands = options['nOfCBands'], options['nOfRBands'] # nOfBands is actually ignored
    if len(cs) == 0:
        return None, None, None
    bandCs = (cs * nOfCBands / nc).astype(int) # cs=0 -> bandCs=0; cs=nc-1 -> bandCs=int((nc-1)*nOfBands/nc) = nOfBands-1
    bandRs = (rs * nOfRBands / nr).astype(int) # rs=0 -> bandRs=0; rs=nr-1 -> bandRs=int((nr-1)*nOfBands/nr) = nOfBands-1
    bandGs = bandCs * 1 * (nOfRBands + 1) + bandRs # global counter
    bandGsUnique = np.asarray(list(set(list(bandGs))))
    possSelected, bandCsSelected, bandRsSelected = [[] for item in range(3)]
    for pos, bandGUnique in enumerate(bandGsUnique):
        possOfBandGUnique = np.where(bandGs == bandGUnique)[0] # list of global positions
        if len(possOfBandGUnique) == 1:
            posOfBandGUnique = possOfBandGUnique[0] # global position
        else:
            posOfBandGUnique = possOfBandGUnique[np.argmin(es[possOfBandGUnique])] # global position
        possSelected.append(posOfBandGUnique)
        bandCsSelected.append(bandCs[posOfBandGUnique])
        bandRsSelected.append(bandRs[posOfBandGUnique])
    return possSelected, bandCsSelected, bandRsSelected
def SelectedVariables2AllVariables(selectedVariables, selectedVariablesKeys, options={}): # 202201101351
    ''' comments:
    .- input selectedVariablesKeys is a string-list
    .- input selectedVariables a float-ndarray
    .- output allVariables is a float-ndarray
        .- if not in selectedVariables, then k1a, k2a, p1a and p2a are set to 0
        .- if not in selectedVariables, then sra is set to sca
        .- if not in selectedVariables, then oc and or are respectively set to kc and kr
    '''
    allVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or'] # WATCH OUT order matters
    keys, defaultValues = ['nc', 'nr'], None
    options = CompleteADictionary(options, keys, defaultValues)
    allVariablesDictionary = Array2Dictionary(selectedVariablesKeys, selectedVariables)
    for key in [item for item in allVariablesKeys if item not in selectedVariablesKeys]: # IMP*
        if key in ['k1a', 'k2a', 'p1a', 'p2a']:
            allVariablesDictionary[key] = 0.
        elif key == 'sra':
            allVariablesDictionary[key] = allVariablesDictionary['sca']
        elif key == 'oc':
            allVariablesDictionary[key] = N2K(options['nc']) # kc
        elif key == 'or':
            allVariablesDictionary[key] = N2K(options['nr']) # kr
        else:
            assert False
    allVariables = Dictionary2Array(allVariablesKeys, allVariablesDictionary)
    return allVariables
def SubsetVariables2AllVariables(dataBasic, subsetVariables, subsetVariablesKeys, subCsetVariablesDictionary, options={}): # 202109251454
    ''' comments:
    .- input dataBasic is a dictionary (including at least 'selectedVariablesKeys')
    .- input subsetVariables is a float-ndarray
    .- input subsetVariablesKeys is a string-list
    .- input subCsetVariablesDictionary is a dictionary
    .- output allVariables is a float-ndarray
    '''
    keys, defaultValues = ['nc', 'nr'], None
    options = CompleteADictionary(options, keys, defaultValues)
    subsetVariablesDictionary = Array2Dictionary(subsetVariablesKeys, subsetVariables)
    selectedVariablesDictionary = {**subsetVariablesDictionary, **subCsetVariablesDictionary} # IMP*
    selectedVariables = Dictionary2Array(dataBasic['selectedVariablesKeys'], selectedVariablesDictionary)
    optionsTMP = {key:options[key] for key in ['nc', 'nr']}
    allVariables = SelectedVariables2AllVariables(selectedVariables, dataBasic['selectedVariablesKeys'], options=optionsTMP)
    return allVariables
def UDaVDa2UUaVUa(mainSet, uDas, vDas): # uD* and vD* -> uU* and vU* can be expensive 202109131500 # *** 
    ''' comments:
    .- input mainSet is a dictionary (including at least 'k1a', 'k2a', 'p1a', 'p2a')
    .- input uDas and vDas are float-ndarrays of the same length
    .- output uUas and vUas are float-ndarrays of the same length or Nones (if it does not succeed)
    .- the funcion is implicit unless k2a = p1a = p2a = 0
    '''
    def DeltaAndError(mainSet, uDas, vDas, uUas, vUas): # 202109131500
        uDasN, vDasN = UUaVUa2UDaVDa(mainSet, uUas, vUas)
        fxs, fys = uDasN - uDas, vDasN - vDas # errors
        error = np.max([np.max(np.abs(fxs)), np.max(np.abs(fys))])
        aux1s = uUas ** 2 + vUas ** 2
        aux1suUa = 2. * uUas
        aux1svUa = 2. * vUas
        aux2s = 1. + mainSet['k1a'] * aux1s + mainSet['k2a'] * aux1s ** 2
        aux2suUa = mainSet['k1a'] * aux1suUa + mainSet['k2a'] * 2. * aux1s * aux1suUa
        aux2svUa = mainSet['k1a'] * aux1svUa + mainSet['k2a'] * 2. * aux1s * aux1svUa
        aux3suUa = 2. * vUas
        aux3svUa = 2. * uUas
        aux4suUa = aux1suUa + 4. * uUas
        aux4svUa = aux1svUa
        aux5suUa = aux1suUa
        aux5svUa = aux1svUa + 4. * vUas
        JuUasuUa = aux2s + uUas * aux2suUa + mainSet['p2a'] * aux4suUa + mainSet['p1a'] * aux3suUa
        JuUasvUa = uUas * aux2svUa + mainSet['p2a'] * aux4svUa + mainSet['p1a'] * aux3svUa
        JvUasuUa = vUas * aux2suUa + mainSet['p1a'] * aux5suUa + mainSet['p2a'] * aux3suUa
        JvUasvUa = aux2s + vUas * aux2svUa + mainSet['p1a'] * aux5svUa + mainSet['p2a'] * aux3svUa
        determinants = JuUasuUa * JvUasvUa - JuUasvUa * JvUasuUa
        determinants = ClipWithSign(determinants, 1.e-8, 1. / 1.e-8)
        JinvuUasuUa = + JvUasvUa / determinants
        JinvvUasvUa = + JuUasuUa / determinants
        JinvuUasvUa = - JuUasvUa / determinants
        JinvvUasuUa = - JvUasuUa / determinants
        duUas = - JinvuUasuUa * fxs - JinvuUasvUa * fys
        dvUas = - JinvvUasuUa * fxs - JinvvUasvUa * fys
        return duUas, dvUas, error
    possZero = np.where(np.sqrt(uDas ** 2 + vDas ** 2) < 1.e-11)[0]
    if len(possZero) > 0:
        uDas[possZero], vDas[possZero] = [0.1 * np.ones(len(possZero)) for item in range(2)] # give another value (0.1) to proceed
    if np.allclose([mainSet['k2a'], mainSet['p1a'], mainSet['p2a']], [0., 0., 0.]): # explicit
        if np.allclose(mainSet['k1a'], 0.):
            uUas, vUas = uDas * 1., vDas * 1.
        else: # Cardano's solution
            aux0s = np.sqrt(uDas ** 2 + vDas ** 2)
            p, qs = 1. / mainSet['k1a'], - aux0s / mainSet['k1a']
            Ds = qs ** 2 + 4. / 27. * p ** 3 # discriminant
            aux1s = np.zeros(Ds.shape)
            pos0, posP, posN = np.where(Ds == 0.)[0], np.where(Ds > 0.)[0], np.where(Ds < 0.)[0]
            assert len(posP) + len(posN) + len(pos0) == len(Ds)
            if len(posP) > 0.:
                auxTMP = (-qs[posP] + np.sqrt(Ds[posP])) / 2.
                us = np.sign(auxTMP) * np.abs(auxTMP) ** (1./3.)
                auxTMP = (-qs[posP] - np.sqrt(Ds[posP])) / 2.
                vs = np.sign(auxTMP) * np.abs(auxTMP) ** (1./3.)
                aux1s[posP] = us + vs
            if len(pos0) > 0.:
                aux1s[pos0] = -3. * qs[pos0] / 2. / p # the second solution
            if len(posN) > 0.:
                auxTMP = np.arccos(-qs[posN] * 0.5 * np.sqrt(27. / (-p ** 3))) / 3.
                aux1s[posN] = 2. * np.sqrt(-p / 3.) * np.cos(auxTMP + 2. * 2 * np.pi / 3.) # the third solution (k = 2)
            uUas, vUas = uDas * aux1s / aux0s, vDas * aux1s / aux0s
        converged = True
    else: # implicit (Newton using DeltaAndError)
        uUas, vUas, converged, speed, counter, error = 1. * uDas, 1. * vDas, False, 1., 0, 1.e+11 # initialize undistorted with distorted
        while not converged and counter <= 20:
            duUas, dvUas, errorN = DeltaAndError(mainSet, uDas, vDas, uUas, vUas)
            if errorN > 2. * error:
                break
            uUas, vUas, error = uUas + speed * duUas, vUas + speed * dvUas, 1. * errorN
            converged, counter = error <= 1.e-11, counter + 1
    if not converged:
        uUas, vUas = None, None
    else:
        if len(possZero) > 0:
            uDas[possZero], vDas[possZero] = [np.zeros(len(possZero)) for item in range(2)]
            uUas[possZero], vUas[possZero] = [np.zeros(len(possZero)) for item in range(2)]
        uDasR, vDasR = UUaVUa2UDaVDa(mainSet, uUas, vUas)
        assert max([np.max(np.abs(uDasR - uDas)), np.max(np.abs(vDasR - vDas))]) < 5. * 1.e-11
    return uUas, vUas
def UUaVUa2UDaVDa(mainSet, uUas, vUas): # uU* and vU* -> uD* and vD* 202109131500 # *** 
    ''' comments:
    .- input mainSet is a dictionary (including at least 'k1a', 'k2a', 'p1a' and 'p2a')
    .- input uUas and vUas are floats or float-ndarrays of the same length
    .- output uDas and vDas are floats or float-ndarrays of the same length as uUas and vUas
    '''
    aux1s = uUas ** 2 + vUas ** 2
    aux2s = 1. + mainSet['k1a'] * aux1s + mainSet['k2a'] * aux1s ** 2
    aux3s = 2. * uUas * vUas
    aux4s = aux1s + 2. * uUas ** 2
    aux5s = aux1s + 2. * vUas ** 2
    uDas = uUas * aux2s + mainSet['p2a'] * aux4s + mainSet['p1a'] * aux3s
    vDas = vUas * aux2s + mainSet['p1a'] * aux5s + mainSet['p2a'] * aux3s
    return uDas, vDas
def UUaVUa2XYZ(mainSet, planes, uUas, vUas, options={}): # can be expensive 202109141800 # *** 
    ''' comments:
    .- input mainSet is a dictionary (including at least 'eu', 'ev', 'ef', 'pc')
    .- input planes is a dictionary (including at least 'pxs', 'pys', 'pzs' and 'pts')
        .- input planes['pxs'/'pys'/'pzs'/'pts'] is a float or a float-ndarray of the same length as uUas and vUas
    .- input uUas and vUas are float-ndarrays of the same length
    .- output xs, ys, zs are float-ndarrays of the same length as uUas and vUas
    .- output possRightSideOfCamera is a list of integers or None (if not options['returnPositionsRightSideOfCamera'])
    '''
    keys, defaultValues = ['returnPositionsRightSideOfCamera'], [False]
    options = CompleteADictionary(options, keys, defaultValues)
    A11s = mainSet['ef'][0] * uUas - mainSet['eu'][0]
    A12s = mainSet['ef'][1] * uUas - mainSet['eu'][1]
    A13s = mainSet['ef'][2] * uUas - mainSet['eu'][2]
    bb1s = uUas * np.sum(mainSet['pc'] * mainSet['ef']) - np.sum(mainSet['pc'] * mainSet['eu'])
    A21s = mainSet['ef'][0] * vUas - mainSet['ev'][0]
    A22s = mainSet['ef'][1] * vUas - mainSet['ev'][1]
    A23s = mainSet['ef'][2] * vUas - mainSet['ev'][2]
    bb2s = vUas * np.sum(mainSet['pc'] * mainSet['ef']) - np.sum(mainSet['pc'] * mainSet['ev'])
    A31s = + planes['pxs'] # float or float-ndarray
    A32s = + planes['pys'] # float or float-ndarray
    A33s = + planes['pzs'] # float or float-ndarray
    bb3s = - planes['pts'] # float or float-ndarray
    auxs = A11s * (A22s * A33s - A23s * A32s) + A12s * (A23s * A31s - A21s * A33s) + A13s * (A21s * A32s - A22s * A31s)
    auxs = ClipWithSign(auxs, 1.e-8, 1. / 1.e-8)
    xs = (bb1s * (A22s * A33s - A23s * A32s) + A12s * (A23s * bb3s - bb2s * A33s) + A13s * (bb2s * A32s - A22s * bb3s)) / auxs
    ys = (A11s * (bb2s * A33s - A23s * bb3s) + bb1s * (A23s * A31s - A21s * A33s) + A13s * (A21s * bb3s - bb2s * A31s)) / auxs
    zs = (A11s * (A22s * bb3s - bb2s * A32s) + A12s * (bb2s * A31s - A21s * bb3s) + bb1s * (A21s * A32s - A22s * A31s)) / auxs
    poss = np.where(np.max(np.asarray([np.abs(xs), np.abs(ys), np.abs(zs)]), axis=0) < 1. / 1.e-8)[0]
    if isinstance(planes['pxs'], (np.ndarray)):
        auxs = planes['pxs'][poss] * xs[poss] + planes['pys'][poss] * ys[poss] + planes['pzs'][poss] * zs[poss] + planes['pts'][poss]
    else:
        auxs = planes['pxs'] * xs[poss] + planes['pys'] * ys[poss] + planes['pzs'] * zs[poss] + planes['pts']
    assert np.allclose(auxs, np.zeros(len(poss)))
    poss = np.where(np.max(np.asarray([np.abs(xs), np.abs(ys), np.abs(zs)]), axis=0) < 1. / 1.e-8)[0]
    uUasR, vUasR = XYZ2UUaVUa(mainSet, xs[poss], ys[poss], zs[poss], options={})[0:2]
    assert (np.allclose(uUasR, uUas[poss]) and np.allclose(vUasR, vUas[poss]))
    if options['returnPositionsRightSideOfCamera']:
        possRightSideOfCamera = XYZ2PositionsRightSideOfCamera(mainSet, xs, ys, zs)
    else:
        possRightSideOfCamera = None
    return xs, ys, zs, possRightSideOfCamera
def UaVa2CR(mainSet, uas, vas): # 202109101200 # *** 
    ''' comments:
    .- input mainSet is a dictionary (including at least 'sca', 'sra', 'oc' and 'or')
        .- mainSet['sca'] and mainSet['sra'] are non-zero, but allowed to be negative
    .- input uas and vas are floats or float-ndarrays
    .- output cs and rs are floats or float-ndarrays
    '''
    cs = uas / mainSet['sca'] + mainSet['oc'] # WATCH OUT (sca?)
    rs = vas / mainSet['sra'] + mainSet['or'] # WATCH OUT (sra?)
    return cs, rs
def UnitVectors2R(eu, ev, ef): # 202109231416
    ''' comments:
    .- input eu, ev and ef are 3-float-ndarrays
    .- output R is a 3x3-float-ndarray
        .- the rows of R are eu, ev and ef
    '''
    R = np.asarray([eu, ev, ef])
    euR, evR, efR = R2UnitVectors(R)
    assert np.allclose(euR, eu) and np.allclose(evR, ev) and np.allclose(efR, ef)
    return R
def UpdateSubCsetVariables(dataBasic, ncs, nrs, css, rss, xss, yss, zss, chss, rhss, subsetVariabless, subsetVariablesKeys, subCsetVariablesSeed, subCsetVariablesKeys, options={}): # 202109271810
    ''' comments:
    .- input dataBasic is a dictionary
    .- input ncs and nrs are integer- or float-lists
    .- input css, rss, xss, yss, zss, chss, rhss and subsetVariabless are float-ndarrays-lists
    .- input subsetVariablesKeys is a string-list
    .- input subCsetVariablesSeed is a float-ndarray
    .- input subCsetVariablesKeys is a string-list
    .- output subCsetVariables is a float-ndarray
    .- output errorT is a float
    '''
    keys, defaultValues = ['aG', 'aH', 'nOfSeedsForC', 'orderOfTheHorizonPolynomial', 'radiusOfEarth'], [1., 1., 20, 5, 6.371e+6]
    options = CompleteADictionary(options, keys, defaultValues)
    theArgs = {'dataBasic':dataBasic, 'ncs':ncs, 'nrs':nrs, 'css':css, 'rss':rss, 'xss':xss, 'yss':yss, 'zss':zss, 'aG':options['aG'], 'chss':chss, 'rhss':rhss, 'aH':options['aH'], 'subsetVariabless':subsetVariabless, 'subsetVariablesKeys':subsetVariablesKeys, 'subCsetVariablesKeys':subCsetVariablesKeys, 'orderOfTheHorizonPolynomial':options['orderOfTheHorizonPolynomial'], 'radiusOfEarth':options['radiusOfEarth']}
    scaledSubCsetVariablesSeed = VariablesScaling(dataBasic, subCsetVariablesSeed, subCsetVariablesKeys, 'scale')
    errorTSeed = ScaledSubCsetVariables2FTM(scaledSubCsetVariablesSeed, theArgs)
    scaledSubCsetVariablesO, errorTO = [copy.deepcopy(item) for item in [scaledSubCsetVariablesSeed, errorTSeed]]
    for iOfMonteCarlo in range(options['nOfSeedsForC']):
        scaledSubCsetVariablesP = scaledSubCsetVariablesO * (0.9 + 0.2 * np.random.random(len(scaledSubCsetVariablesO))) # WATCH OUT
        scaledSubCsetVariablesP = optimize.minimize(ScaledSubCsetVariables2FTM, scaledSubCsetVariablesP, args = (theArgs)).x
        errorTP = ScaledSubCsetVariables2FTM(scaledSubCsetVariablesP, theArgs)
        if errorTP < errorTO * 0.999:
            scaledSubCsetVariablesO, errorTO = [copy.deepcopy(item) for item in [scaledSubCsetVariablesP, errorTP]]
    scaledSubCsetVariables, errorT = [copy.deepcopy(item) for item in [scaledSubCsetVariablesO, errorTO]]
    subCsetVariables = VariablesScaling(dataBasic, scaledSubCsetVariables, subCsetVariablesKeys, 'unscale')
    return subCsetVariables, errorT
def VariablesScaling(dataBasic, variables, variablesKeys, direction): # 202109241439
    ''' comments:
    .- input dataBasic is a dictionary (including at least 'scalesDictionary')
    .- input variables is a float-ndarray
    .- input variablesKeys is a string-list
    .- input direction is a string ('scale' or 'unscale')
    .- output variables is a float-ndarray of the same length of input variables
    '''
    scales = Dictionary2Array(variablesKeys, dataBasic['scalesDictionary'])
    if direction == 'scale':
        variables = variables / scales
    elif direction == 'unscale':
        variables = variables * scales
    else:
        assert False
    return variables
def WriteCalTxt(pathCalTxt, allVariables, nc, nr, errorT): # 202110131423
    ''' comments:
    .- input pathCalTxt is a string
    .- input allVariables is a 14-float-ndarray
    .- input nc and nr are integers or floats
    .- input errorT is a float
    '''
    allVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or'] # WATCH OUT order matters
    MakeFolder(pathCalTxt[0:pathCalTxt.rfind(os.sep)])
    fileout = open(pathCalTxt, 'w')
    for pos in range(len(allVariables)):
        if allVariablesKeys[pos] in ['ph', 'sg', 'ta']:
            allVariables[pos] = np.angle(np.exp(1j * allVariables[pos]))
        fileout.write('{:21.9f} \t {:}\n'.format(allVariables[pos], allVariablesKeys[pos]))
    fileout.write('{:21.0f} \t {:}\n'.format(np.round(nc), 'nc'))
    fileout.write('{:21.0f} \t {:}\n'.format(np.round(nr), 'nr'))
    fileout.write('{:21.9f} \t {:}\n'.format(errorT, 'errorT'))
    fileout.close()
    return None
def WriteDataPdfTxt(data): # 202109141700 # *** 
    ''' comments:
    .- input data is a dictionary (including at least 'pathFile', 'xUpperLeft', 'yUpperLeft', 'angle', 'xYLengthInC', 'xYLengthInR', 'ppm' and 'timedeltaTolerance')
    '''
    fileout = open(data['pathFile'], 'w')
    fileout.write('{:20.8f} real world x for upper left corner, in meters \n'.format(data['xUpperLeft']))
    fileout.write('{:20.8f} real world y for upper left corner, in meters \n'.format(data['yUpperLeft']))
    fileout.write('{:20.8f} angle, in degrees: 0 = E, 90 = N \n'.format(data['angle'] * 180. / np.pi))
    fileout.write('{:20.8f} real world length column-wise, in meters \n'.format(data['xYLengthInC']))
    fileout.write('{:20.8f} real world length row-wise, in meters \n'.format(data['xYLengthInR']))
    fileout.write('{:20.8f} pixels per meter \n'.format(data['ppm']))
    fileout.write('{:20.8f} time delta tolerance, in hours \n'.format(data['timedeltaTolerance'].total_seconds() / 3600.))
    fileout.close()
    return None
def XY2CR(dataPdfTxt, xs, ys, options={}): # 202109101200 # *** 
    ''' comments:
    .- input dataPdfTxt is a dictionary (including at least 'ppm', 'angle', 'xUpperLeft', 'yUpperLeft', 'nc' and 'nr')
    .- input xs and ys are float-arrays of the same length
    .- output cs and rs are float-arrays of the same length
    .- output possGood is an integer-list or None
    '''
    keys, defaultValues = ['dobleCheck', 'imgMargins', 'returnGoodPositions'], [True, {'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}, False]
    options = CompleteADictionary(options, keys, defaultValues)
    xps = xs - dataPdfTxt['xUpperLeft']
    yps = ys - dataPdfTxt['yUpperLeft']
    cs = + (+ np.cos(dataPdfTxt['angle']) * xps + np.sin(dataPdfTxt['angle']) * yps) * dataPdfTxt['ppm']
    rs = - (- np.sin(dataPdfTxt['angle']) * xps + np.cos(dataPdfTxt['angle']) * yps) * dataPdfTxt['ppm']
    if options['returnGoodPositions']:
        possGood = CR2PositionsWithinImage(dataPdfTxt['nc'], dataPdfTxt['nr'], cs, rs, options={'imgMargins':options['imgMargins']})
    else:
        possGood = None
    if options['dobleCheck']:
        xsR, ysR = CR2XY(dataPdfTxt, cs, rs, options={'dobleCheck':False, 'imgMargins':options['imgMargins'], 'returnGoodPositions':False})[0:2]
        assert all([np.allclose(xs, xsR), np.allclose(ys, ysR)])
    return cs, rs, possGood
def XYZ2A0(xs, ys, zs): # 202201250808
    '''
    .- input xs, ys and zs are float-ndarrays of the same length
    .- output A0 is a (2*len(xs)x8)-float-ndarray
    '''
    poss0, poss1 = Poss0AndPoss1(len(xs))
    A0 = np.zeros((2 * len(xs), 8))
    A0[poss0, 0], A0[poss0, 1], A0[poss0, 2], A0[poss0, 3] = xs, ys, zs, np.ones(xs.shape)
    A0[poss1, 4], A0[poss1, 5], A0[poss1, 6], A0[poss1, 7] = xs, ys, zs, np.ones(xs.shape)
    return A0
def XYZ2CDRD(mainSet, xs, ys, zs, options={}): # 202109131600 # *** 
    ''' comments:
    .- input mainSet is a dictionary (including at least 'nc' and 'nr')
    .- input xs, ys and zs are float-ndarrays of the same length
    .- output cDs and rDs are float-ndarrays of the same length as xs, ys and zs
    .- output possGood is a list of integers or None (if not options['returnGoodPositions'])
    '''
    keys, defaultValues = ['imgMargins', 'returnGoodPositions'], [{'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}, False]
    options = CompleteADictionary(options, keys, defaultValues)
    optionsTMP = {'returnPositionsRightSideOfCamera':options['returnGoodPositions']}
    uUas, vUas, possGood = XYZ2UUaVUa(mainSet, xs, ys, zs, options=optionsTMP)
    uDas, vDas = UUaVUa2UDaVDa(mainSet, uUas, vUas)
    cDs, rDs = UaVa2CR(mainSet, uDas, vDas)
    if options['returnGoodPositions']: # so far possGood are at the right side of the camera
        if len(possGood) > 0:
            nc, nr, cDsGood, rDsGood = mainSet['nc'], mainSet['nr'], cDs[possGood], rDs[possGood]
            optionsTMP = {'imgMargins':options['imgMargins']}
            possGoodInGood = CR2PositionsWithinImage(nc, nr, cDsGood, rDsGood, options=optionsTMP)
            possGood = [possGood[item] for item in possGoodInGood]
        if len(possGood) > 0:
            xsGood, ysGood, zsGood, csGood, rsGood = [item[possGood] for item in [xs, ys, zs, cDs, rDs]]
            xsGoodR, ysGoodR = CDRDZ2XY(mainSet, csGood, rsGood, zsGood, options={})[0:2] # all, not good positions
            distances = np.sqrt((xsGood - xsGoodR) ** 2 + (ysGood - ysGoodR) ** 2)
            possGoodInGood = np.where(distances < 1.e-5)[0]
            possGood = [possGood[item] for item in possGoodInGood]
    else:
        assert possGood is None # from XYZ2UUaVUa
    return cDs, rDs, possGood
def XYZ2PositionsRightSideOfCamera(mainSet, xs, ys, zs): # 202109231412
    '''
    .- input mainSet is a dictionary (including 'xc', 'yc', 'zc' and 'ef')
    .- input xs, ys and zs are float-ndarrays of the same length
    .- output possRightSideOfCamera is a integer-list
    '''
    xas, yas, zas = xs - mainSet['xc'], ys - mainSet['yc'], zs - mainSet['zc']
    possRightSideOfCamera = np.where(xas * mainSet['ef'][0] + yas * mainSet['ef'][1] + zas * mainSet['ef'][2] > 0)[0]
    return possRightSideOfCamera
def XYZ2UUaVUa(mainSet, xs, ys, zs, options={}): # 202109231411
    ''' comments:
    .- input mainSet is a dictionary (including at least 'xc', 'yc', 'zc', 'eu', 'ev' and 'ef')
    .- input xs, ys and zs are float-ndarrays of the same length
    .- output uUas and vUas are float-ndarrays of the same length as xs, ys and zs
    .- output possRightSideOfCamera is a list of integers or None (if not options['returnPositionsRightSideOfCamera'])
    '''
    keys, defaultValues = ['returnPositionsRightSideOfCamera'], [False]
    options = CompleteADictionary(options, keys, defaultValues)
    xas, yas, zas = xs - mainSet['xc'], ys - mainSet['yc'], zs - mainSet['zc']
    nus = xas * mainSet['eu'][0] + yas * mainSet['eu'][1] + zas * mainSet['eu'][2]
    nvs = xas * mainSet['ev'][0] + yas * mainSet['ev'][1] + zas * mainSet['ev'][2]
    dns = xas * mainSet['ef'][0] + yas * mainSet['ef'][1] + zas * mainSet['ef'][2]
    dns = ClipWithSign(dns, 1.e-8, 1. / 1.e-8)
    uUas = nus / dns
    vUas = nvs / dns
    if options['returnPositionsRightSideOfCamera']:
        possRightSideOfCamera = XYZ2PositionsRightSideOfCamera(mainSet, xs, ys, zs)
    else:
        possRightSideOfCamera = None
    return uUas, vUas, possRightSideOfCamera
def XYZPa112CURU(xs, ys, zs, Pa11): # *** 
    ''''
    .- input xs, ys and zs are float-ndarrays of the same length
    .- input Pa11 is a 11-float-ndarray
    .- output cUs and rUs are float-ndarrays of the same length as xs
    '''
    dens = Pa11[8] * xs + Pa11[9] * ys + Pa11[10] * zs + 1.
    cUs = (Pa11[0] * xs + Pa11[1] * ys + Pa11[2] * zs + Pa11[3]) / dens
    rUs = (Pa11[4] * xs + Pa11[5] * ys + Pa11[6] * zs + Pa11[7]) / dens
    return cUs, rUs
def Xi2XForParabolicDistortion(xis): # *** 
    ''' comments:
    .- input xis is a float-ndarray
    .- output xs is a float-ndarray
        .- it is solved: xis = xs ** 3 + 2 * xs ** 2 + xs
    '''
    p, qs, Deltas = -1. /3., -(xis + 2. / 27.), (xis + 4. / 27.) * xis
    if np.max(Deltas) < 0: # for xis in (-4/27, 0)
        n3s = (qs + 1j * np.sqrt(np.abs(Deltas))) / 2.
        ns = np.abs(n3s) ** (1. / 3.) * np.exp(1j * (np.abs(np.angle(n3s)) + 2. * np.pi * 1.) / 3.) # we ensure theta > 0; + 2 pi j for j = 0, 1, 2
    elif np.min(Deltas) >= 0: # for xis not in (-4/27, 0)
        auxs = (qs + np.sqrt(Deltas)) / 2.
        ns = np.sign(auxs) * (np.abs(auxs) ** (1. / 3.))
    else:
        possN, possP, ns = np.where(Deltas < 0)[0], np.where(Deltas >= 0)[0], np.zeros(xis.shape) + 1j * np.zeros(xis.shape)
        n3sN = (qs[possN] + 1j * np.sqrt(np.abs(Deltas[possN]))) / 2.
        ns[possN] = np.abs(n3sN) ** (1. / 3.) * np.exp(1j * (np.abs(np.angle(n3sN)) + 2. * np.pi * 1.) / 3.) # we ensure theta > 0; + 2 pi j for j = 0, 1, 2
        auxs = (qs[possP] + np.sqrt(Deltas[possP])) / 2.
        ns[possP] = np.sign(auxs) * (np.abs(auxs) ** (1. / 3.))
    xs = np.real(p / (3. * ns) - ns - 2. / 3.)
    assert np.allclose(xs ** 3 + 2 * xs ** 2 + xs, xis) # avoidable
    return xs
