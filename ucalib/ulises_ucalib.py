#
# Thu Nov 10 15:14:11 2022, extract from Ulises by Gonzalo Simarro and Daniel Calvete
#
import copy
import cv2
import datetime
import numpy as np
import os
import random
from scipy import optimize
import time
#
def AB2Pa11(A, b): # 202207061243 (last read 2022-07-06)
    '''
    .- input A is a (2*nx11)-float-ndarray
    .- input b is a 2*n-float-ndarray
    .- output Pa11 is a 11-float-ndarray
    '''
    Pa11 = np.linalg.solve(np.dot(np.transpose(A), A), np.dot(np.transpose(A), b))
    return Pa11
def AllVariables2MainSet(allVariables, nc, nr, options={}): # 202109141500 (last read 2022-07-06)
    ''' comments:
    .- input allVariables is a 14-float-ndarray (xc, yc, zc, ph, sg, ta, k1a, k2a, p1a, p2a, sca, sra, oc, or)
    .- input nc and nr are integers or floats
    .- output mainSet is a dictionary
    '''
    keys, defaultValues = ['orderOfHorizonPoly', 'radiusOfEarth'], [5, 6.371e+6]
    options = CompleteADictionary(options, keys, defaultValues)
    allVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or'] # IMP* order matters
    mainSet = {'nc':nc, 'nr':nr, 'orderOfHorizonPoly':options['orderOfHorizonPoly'], 'radiusOfEarth':options['radiusOfEarth']}
    allVariablesDict = Array2Dictionary(allVariablesKeys, allVariables)
    allVariablesDict['sca'] = ClipWithSign(allVariablesDict['sca'], 1.e-8, 1.e+8)
    allVariablesDict['sra'] = ClipWithSign(allVariablesDict['sra'], 1.e-8, 1.e+8)
    allVariables = Dictionary2Array(allVariablesKeys, allVariablesDict)
    mainSet['allVariablesDict'] = allVariablesDict
    mainSet.update(allVariablesDict) # IMP* (absorb in mainSet all the keys which are in allVariablesDict)
    mainSet['allVariables'] = allVariables
    mainSet['pc'] = np.asarray([mainSet['xc'], mainSet['yc'], mainSet['zc']])
    R = EulerianAngles2R(mainSet['ph'], mainSet['sg'], mainSet['ta'])
    eu, ev, ef = R2UnitVectors(R)
    mainSet['R'] = R
    mainSet['eu'], (mainSet['eux'], mainSet['euy'], mainSet['euz']) = eu, eu
    mainSet['ev'], (mainSet['evx'], mainSet['evy'], mainSet['evz']) = ev, ev
    mainSet['ef'], (mainSet['efx'], mainSet['efy'], mainSet['efz']) = ef, ef
    Pa = MainSet2Pa(mainSet)
    mainSet['Pa'], mainSet['Pa11'] = Pa, Pa2Pa11(Pa)
    horizonLine = MainSet2HorizonLine(mainSet)
    mainSet['horizonLine'] = horizonLine
    mainSet.update(horizonLine) # IMP* (absorb in mainSet all the keys which are in horizonLine)
    return mainSet
def AllVariables2SubsetVariables(dataBasic, allVariables, subsetVariablesKeys, options={}): # 202109251523 (last read 2022-11-08)
    ''' comments:
    .- input dataBasic is a dictionary # WATCH OUT: UNUSED!!!
    .- input allVariables is a 14-float-ndarray
    .- input subsetVariablesKeys is a string-list (subset of allVariablesKeys below)
    .- output subsetVariables is a float-ndarray
    '''
    keys, defaultValues = ['possSubsetInAll'], None
    options = CompleteADictionary(options, keys, defaultValues)
    if options['possSubsetInAll'] is not None:
        subsetVariables = allVariables[options['possSubsetInAll']]
    else:
        allVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or'] # IMP* order matters
        allVariablesDict = Array2Dictionary(allVariablesKeys, allVariables)
        subsetVariables = Dictionary2Array(subsetVariablesKeys, allVariablesDict)
    return subsetVariables
def ApplyAffineA01(A01, xs0, ys0): # 202111241134 (last read 2022-11-10)
    ''' comments:
    .- input A01 is a 6-float-ndarray (allows to transform from 0 to 1)
    .- input xs0 and ys0 are float-ndarrays of the same length
    .- output xs1 and ys1 are float-ndarrays of the same length of xs0 and ys0
    '''
    xs1 = A01[0] * xs0 + A01[1] * ys0 + A01[2]
    ys1 = A01[3] * xs0 + A01[4] * ys0 + A01[5]
    return xs1, ys1
def ApplyHomographyHa01(Ha01, xs0, ys0): # 202110141303 (last read 2022-11-09)
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
def AreImgMarginsOK(nc, nr, imgMargins): # 202109101200 (last read 2022-07-06)
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
def Array2Dictionary(keys, theArray): # 202206291320 (last read 2022-06-29)
    ''' comments:
    .- input keys is a string-list
    .- input theArray is a list or ndarray of the same length of keys
    .- output theDictionary is a dictionary
    '''
    theDictionary = {keys[posKey]:theArray[posKey] for posKey in range(len(keys))}
    return theDictionary
def CDRD2CURU(mainSet, cDs, rDs): # 202109101200 (last read 2022-07-06) potentially expensive
    ''' comments:
    .- input mainSet is a dictionary (including at least 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc' and 'or')
    .- input cDs and rDs are float-ndarrays of the same length
    .- output cUs and rUs are float-ndarrays of the same length of cDs and rDs or Nones (if it does not succeed)
    '''
    uDas, vDas = CR2UaVa(mainSet, cDs, rDs)
    uUas, vUas = UDaVDa2UUaVUa(mainSet, uDas, vDas) # potentially expensive
    if uUas is None or vUas is None:
        cUs, rUs = None, None
    else:
        cUs, rUs = UaVa2CR(mainSet, uUas, vUas)
    return cUs, rUs
def CDRD2CURUForParabolicSquaredDistortion(cDs, rDs, oca, ora, k1asa2): # 202207061214 (last read 2022-07-06)
    ''' comments:
    .- input cDs and rDs are floats or float-ndarrays of the same length
    .- input oca and ora are floats
    .- input k1asa2 is a float (k1a * sa ** 2)
    .- output cUs and rUs are floats or float-ndarrays of the same length of cDs and rDs
    '''
    if np.abs(k1asa2) < 1.e-11:
        cUs = 1. * cDs
        rUs = 1. * rDs
    else:
        dDs2 = (cDs - oca) ** 2 + (rDs - ora) ** 2
        xias = k1asa2 * dDs2 # xi_*
        xas = Xi2XForParabolicDistortion(xias) # x_*
        cUs = (cDs - oca) / (1. + xas) + oca
        rUs = (rDs - ora) / (1. + xas) + ora
    cDsR, rDsR = CURU2CDRDForParabolicSquaredDistortion(cUs, rUs, oca, ora, k1asa2)
    assert max([np.max(np.abs(cDsR - cDs)), np.max(np.abs(rDsR - rDs))]) < 5. * 1.e-11
    return cUs, rUs
def CDRDZ2XY(mainSet, cDs, rDs, zs, options={}): # 202109231442 (last read 2022-07-12) potentially expensive
    ''' comments:
    .- input mainSet is a dictionary (including at least 'nc' and 'nr')
    .- input cDs, rDs and zs are float-ndarrays of the same length (zs can be just a float)
    .- output xs and ys are float-ndarrays of the same length or Nones (if it does not succeed)
    .- output possGood is a integer-list or None (if it does not succeed or not options['returnGoodPositions'])
    '''
    keys, defaultValues = ['returnGoodPositions', 'imgMargins'], [False, {'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}]
    options = CompleteADictionary(options, keys, defaultValues)
    cUs, rUs = CDRD2CURU(mainSet, cDs, rDs) # potentially expensive
    if cUs is None or rUs is None:
        return None, None, None
    uUas, vUas = CR2UaVa(mainSet, cUs, rUs)
    if isinstance(zs, np.ndarray): # float-ndarray
        planes = {'pxs':np.zeros(zs.shape), 'pys':np.zeros(zs.shape), 'pzs':np.ones(zs.shape), 'pts':-zs}
    else: # float
        planes = {'pxs':0., 'pys':0., 'pzs':1., 'pts':-zs}
    xs, ys, zsR, possGood = UUaVUa2XYZ(mainSet, planes, uUas, vUas, options={'returnPositionsRightSideOfCamera':options['returnGoodPositions']})
    if isinstance(zs, np.ndarray): # float-ndarray
        assert np.allclose(zsR, zs)
    else: # float
        assert np.allclose(zsR, zs*np.ones(len(xs)))
    if options['returnGoodPositions']:
        if len(possGood) > 0:
            nc, nr, cDsGood, rDsGood = mainSet['nc'], mainSet['nr'], cDs[possGood], rDs[possGood]
            possGoodInGood = CR2PositionsWithinImage(nc, nr, cDsGood, rDsGood, options={'imgMargins':options['imgMargins']})
            possGood = [possGood[item] for item in possGoodInGood]
    else: # possGood is None from UUaVUa2XYZ above
        assert possGood is None
    return xs, ys, possGood
def CDh2RDh(horizonLine, cDhs, options={}): # 202109141100 (last read 2022-07-06)
    ''' comments:
    .- input horizonLine is a dictionary (including at least 'ccDh')
        .- the horizon line is rDhs = ccDh[0] + ccDh[1] * cDhs + ccDh[2] * cDhs ** 2 + ...
    .- input cDhs is a float-ndarray
    .- output rDhs is a float-ndarray of the same length of cDhs
    .- output possGood is an integer-list or None (if not options['returnGoodPositions'])
    '''
    keys, defaultValues = ['imgMargins', 'returnGoodPositions'], [{'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}, False]
    options = CompleteADictionary(options, keys, defaultValues)
    rDhs = horizonLine['ccDh'][0] * np.ones(cDhs.shape)
    for n in range(1, len(horizonLine['ccDh'])):
        rDhs = rDhs + horizonLine['ccDh'][n] * cDhs ** n
    if options['returnGoodPositions']:
        possGood = CR2PositionsWithinImage(horizonLine['nc'], horizonLine['nr'], cDhs, rDhs, options={'imgMargins':options['imgMargins']})
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
def CR2CRInteger(cs, rs): # 202109131000 (last read 2022-07-06)
    ''' comments:
    .- input cs and rs are integer- or float-ndarrays
    .- output cs and rs are integer-ndarrays
    '''
    cs = np.round(cs).astype(int)
    rs = np.round(rs).astype(int)
    return cs, rs
def CR2CRIntegerAroundAndWeights(cs, rs): # 202109131400 (last read 2022-11-09)
    ''' comments:
    .- input cs and rs are integer- or float-ndarrays of the same length
    .- output csIAround, rsIAround are len(cs)x4-integer-ndarrays 
    .- output wsAround is a len(cs)x4-float-ndarray
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
def CR2CRIntegerWithinImage(nc, nr, cs, rs, options={}): # 202109141700 (last read 2022-07-12)
    ''' comments:
    .- input nc and nr are integers or floats
    .- input cs and rs are float-ndarrays
    .- output csIW and rsIW are integer-ndarrays
    '''
    keys, defaultValues = ['imgMargins'], [{'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}]
    options = CompleteADictionary(options, keys, defaultValues)
    possWithin = CR2PositionsWithinImage(nc, nr, cs, rs, options={'imgMargins':options['imgMargins'], 'rounding':True}) # IMP*
    csIW, rsIW = cs[possWithin].astype(int), rs[possWithin].astype(int)
    return csIW, rsIW
def CR2PositionsWithinImage(nc, nr, cs, rs, options={}): # 202109131400 (last read 2022-07-06)
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
def CR2UaVa(mainSet, cs, rs): # 202109101200 (last read 2022-07-06)
    ''' comments:
    .- input mainSet is a dictionary (including at least 'sca', 'sra', 'oc' and 'or')
        .- mainSet['sca'] and mainSet['sra'] are non-zero, but allowed to be negative
    .- input cs and rs are floats or float-ndarrays of the same length
    .- output uas and vas are floats or float-ndarrays of the same length of cs and rs
    '''
    uas = (cs - mainSet['oc']) * mainSet['sca']
    vas = (rs - mainSet['or']) * mainSet['sra']
    return uas, vas
def CRWithinImage2NormalizedLengthsAndAreas(nc, nr, cs, rs, options={}): # 202211101352 (last read 2022-11-10)
    ''' comments:
    .- input nc and anr are integers
    .- input cs and rs are float-ndarrays of the same length
    .- output normalizedLengths and normalizedAreas are float-ndarrays of the same length
    '''
    keys, defaultValues = ['imgMargins'], [None]
    options = CompleteADictionary(options, keys, defaultValues)
    imgMargins = CompleteImgMargins(options['imgMargins'])
    assert len(CR2PositionsWithinImage(nc, nr, cs, rs, {'imgMargins':imgMargins})) == len(cs)
    cMin, cMax = imgMargins['c0'], nc - 1 - imgMargins['c1']
    rMin, rMax = imgMargins['r0'], nr - 1 - imgMargins['r1']
    totalLength = np.max([cMax - cMin, rMax - rMin])
    totalArea = float((cMax - cMin) * (rMax - rMin))
    lengths = np.zeros((len(cs), 4))
    lengths[:, 0] = cs - cMin
    lengths[:, 1] = cMax - cs
    lengths[:, 2] = rs - rMin
    lengths[:, 3] = rMax - rs
    areas = np.zeros((len(cs), 4))
    areas[:, 0] = (cs - cMin) * (rs - rMin)
    areas[:, 1] = (cs - cMin) * (rMax - rs)
    areas[:, 2] = (cMax - cs) * (rs - rMin)
    areas[:, 3] = (cMax - cs) * (rMax - rs)
    assert np.min(lengths) >= 0. and np.min(areas) >= 0.
    normalizedLengths = np.min(lengths, axis=1) / totalLength
    normalizedAreas = np.min(areas, axis=1) / totalArea
    assert len(normalizedLengths) == len(normalizedAreas) == len(cs)
    return normalizedLengths, normalizedAreas
def CURU2B(cUs, rUs): # 202207061240 (last read 2022-07-06)
    poss0, poss1 = Poss0AndPoss1(len(cUs))    
    b = np.zeros(2 * len(cUs))
    b[poss0] = cUs
    b[poss1] = rUs
    return b
def CURU2CDRD(mainSet, cUs, rUs): # 202109101200 (last read 2022-07-06)
    ''' comments:
    .- input mainSet is a dictionary
    .- input cUs and rUs are floats or float-ndarrays of the same length
    .- output cDs and rDs are floats or float-ndarrays of the same length of cUs and rUs
    '''
    uUas, vUas = CR2UaVa(mainSet, cUs, rUs)
    uDas, vDas = UUaVUa2UDaVDa(mainSet, uUas, vUas)
    cDs, rDs = UaVa2CR(mainSet, uDas, vDas)
    return cDs, rDs
def CURU2CDRDForParabolicSquaredDistortion(cUs, rUs, oca, ora, k1asa2): # 202206261205 (last read 2022-07-06)
    ''' comments:
    .- input cUs and rUs are floats or float-ndarrays of the same length
    .- input oca and ora are floats
    .- input k1asa2 is a float (k1a * sa ** 2)
    .- output cDs and rDs are floats or float-ndarrays of the same length of cUs and rUs
    '''
    dUs2 = (cUs - oca) ** 2 + (rUs - ora) ** 2
    xas = k1asa2 * dUs2 # x_*
    cDs = (cUs - oca) * (1. + xas) + oca
    rDs = (rUs - ora) * (1. + xas) + ora
    return cDs, rDs
def CURUXYZ2A(cUs, rUs, xs, ys, zs): # 202201250813 (last read 2022-07-06)
    '''
    .- input cUs, rUs, xs, ys and zs are float-ndarrays of the same length
    .- output A is a (2*len(cUs)x11)-float-ndarray
    '''
    A = np.concatenate((XYZ2A0(xs, ys, zs), CURUXYZ2A1(cUs, rUs, xs, ys, zs)), axis=1)
    assert A.shape[1] == 11
    return A
def CURUXYZ2A1(cUs, rUs, xs, ys, zs): # 202201250812 (last read 2022-07-06)
    '''
    .- input cUs, rUs, xs, ys and zs are float-ndarrays of the same length
    .- output A1 is a (2*len(cUs)x3)-float-ndarray
    '''
    poss0, poss1 = Poss0AndPoss1(len(cUs))
    A1 = np.zeros((2 * len(xs), 3))
    A1[poss0, 0], A1[poss0, 1], A1[poss0, 2] = -cUs * xs, -cUs * ys, -cUs * zs
    A1[poss1, 0], A1[poss1, 1], A1[poss1, 2] = -rUs * xs, -rUs * ys, -rUs * zs
    return A1
def CUh2RUh(horizonLine, cUhs): # 202109101200 (last read 2022-07-06)
    ''' comments:
    .- input horizonLine is a dictionary (including at least 'ccUh1', 'crUh1' and 'ccUh0')
        .- the horizonLine is 'ccUh1' * cUhs + 'crUh1' * rUhs + 'ccUh0' = 0, i.e., rUhs = - ('ccUh1' * cUhs + 'ccUh0') / 'crUh1'
    .- input cUhs is a float-ndarray
    .- output rUhs is a float-ndarray
    '''
    crUh1 = ClipWithSign(horizonLine['crUh1'], 1.e-8, 1.e+8)
    rUhs = - (horizonLine['ccUh1'] * cUhs + horizonLine['ccUh0']) / crUh1
    return rUhs
def ClipWithSign(xs, x0, x1): # 202109101200 (last read 2022-07-06)
    ''' comments:
    .- input xs is a float of a float-ndarray
    .- input x0 and x1 are floats so that 0 <= x0 <= x1
    .- output xs is a float of a float-ndarray
        .- output xs is in [-x1, -x0] U [x0, x1] and retains the signs of input xs
    '''
    assert x1 >= x0 >= 0.
    signs = np.sign(xs)
    if isinstance(signs, np.ndarray): # ndarray
        signs[signs == 0] = 1
    elif signs == 0: # float and 0
        signs = 1
    xs = signs * np.clip(np.abs(xs), x0, x1)
    return xs
def CloudOfPoints2Rectangle(xs, ys, options={}): # 202110281047 (last read 2022-07-02, checked graphically with auxiliar code)
    ''' comments:
    .- input xs and ys are float-ndarrays of the same length
    .- output xcs and ycs are 4-float-ndarrays (4 corners, oriented clockwise, the first corner is the closest to the first point of the cloud)
    .- output area is a float
    '''
    keys, defaultValues = ['margin'], [0.]
    options = CompleteADictionary(options, keys, defaultValues)
    xcs, ycs, area = None, None, 1.e+11
    for angleH in np.linspace(0, np.pi / 2., 1000):
        xcsH, ycsH, areaH = CloudOfPoints2RectangleAnalysisForAnAngle(angleH, xs, ys, options={'margin':options['margin']})
        if areaH < area:
            xcs, ycs, area = [copy.deepcopy(item) for item in [xcsH, ycsH, areaH]]
    pos0 = np.argmin(np.sqrt((xcs - xs[0]) ** 2 + (ycs - ys[0]) ** 2)) # the first corner is the closest to the first point (they are oriented clockwise)
    xcs = np.asarray([xcs[pos0], xcs[(pos0+1)%4], xcs[(pos0+2)%4], xcs[(pos0+3)%4]])
    ycs = np.asarray([ycs[pos0], ycs[(pos0+1)%4], ycs[(pos0+2)%4], ycs[(pos0+3)%4]])
    return xcs, ycs, area
def CloudOfPoints2RectangleAnalysisForAnAngle(angle, xs, ys, options={}): # 202110280959 (last read 2022-07-02, checked graphically with auxiliar code)
    ''' comments:
    .- input angle is a float (0 = East; pi/2 = North)
    .- input xs and ys are float-ndarrays of the same length
    .- output xcs and ycs are 4-float-ndarrays (4 corners, oriented clockwise)
    .- output area is a float
    '''
    keys, defaultValues = ['margin'], [0.]
    options = CompleteADictionary(options, keys, defaultValues)
    lDs = - np.sin(angle) * xs + np.cos(angle) * ys # signed-distances to D-line dir = (+cos, +sin) through origin (0, 0)
    lPs = + np.cos(angle) * xs + np.sin(angle) * ys # signed-distances to P-line dir = (+sin, -cos) through origin (0, 0)
    area = (np.max(lDs) - np.min(lDs) + 2 * options['margin']) * (np.max(lPs) - np.min(lPs) + 2 * options['margin'])
    lD0 = {'lx':-np.sin(angle), 'ly':+np.cos(angle), 'lt':-(np.min(lDs)-options['margin'])}
    lD1 = {'lx':-np.sin(angle), 'ly':+np.cos(angle), 'lt':-(np.max(lDs)+options['margin'])}
    lP0 = {'lx':+np.cos(angle), 'ly':+np.sin(angle), 'lt':-(np.min(lPs)-options['margin'])}
    lP1 = {'lx':+np.cos(angle), 'ly':+np.sin(angle), 'lt':-(np.max(lPs)+options['margin'])}
    xcs, ycs = [np.zeros(4) for item in range(2)]
    xcs[0], ycs[0] = IntersectionOfTwoLines(lD0, lP0, options={})[0:2] 
    xcs[1], ycs[1] = IntersectionOfTwoLines(lP0, lD1, options={})[0:2]
    xcs[2], ycs[2] = IntersectionOfTwoLines(lD1, lP1, options={})[0:2]
    xcs[3], ycs[3] = IntersectionOfTwoLines(lP1, lD0, options={})[0:2]
    return xcs, ycs, area
def CompleteADictionary(theDictionary, keys, defaultValues): # 202109101200 (last read 2022-06-29)
    ''' comments:
    .- input theDictionary is a dictionary
    .- input keys is a string-list
    .- input defaultValues is a list of the same length of keys or a single value
    .- output theDictionary is a dictionary that includes keys and defaultValues for the keys not in input theDictionary
    '''
    if set(keys) <= set(theDictionary.keys()):
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
def CompleteImgMargins(imgMargins): # 202109101200 (last read 2022-07-05)
    ''' comments:
    .- input imgMargins is a dictionary or None
        .- if imgMargins['isComplete'], then it does nothing
        .- if imgMargins is None, then it is initialized to {'c':0, 'r':0}
        .- if imgMargins includes 'c', then generates 'c0' and 'c1' (if not included); otherwise, 'c0' and 'c1' must already be included
        .- if imgMargins includes 'r', then generates 'r0' and 'r1' (if not included); otherwise, 'r0' and 'r1' must already be included
        .- imgMargins['c*'] and imgMargins['r*'] are integers
    .- output imgMargins is a dictionary (including at least 'c0', 'c1', 'r0' and 'r1' and 'isComplete'; not necessarily 'c' and 'r')
        .- imgMargins['c*'] and imgMargins['r*'] are integers
    '''
    if imgMargins is not None and 'isComplete' in imgMargins.keys() and imgMargins['isComplete']:
        return imgMargins
    if imgMargins is None:
        imgMargins = {'c':0, 'r':0}
    for letter in ['c', 'r']:
        try:
            assert int(imgMargins[letter]) == imgMargins[letter]
        except:
            assert all([int(imgMargins[letter+number]) == imgMargins[letter+number] for number in ['0', '1']])
            continue # go to the next letter
        for number in ['0', '1']:
            try:
                assert int(imgMargins[letter+number]) == imgMargins[letter+number]
            except:
                imgMargins[letter+number] = imgMargins[letter]
    imgMargins['isComplete'] = True
    return imgMargins
def CreatePlanview(plwPC, imgs): # 202109151300 (last read 2022-11-10)
    ''' comments:
    .- input plwPC is a dictionary (including at least 'nc', 'nr' and 'cameras' and dictionaries for cameras)
    .- input imgs is a dictionary for cameras
    .- imgPlw is a cv2.image or None (if there are no images in imgs)
    '''
    assert all([img.shape[2] == 3 for img in [imgs[camera] for camera in imgs.keys()]])
    cameras = [item for item in imgs.keys() if item in plwPC['cameras']]
    if len(cameras) == 0:
        return None
    wsPlanview = np.zeros(plwPC['nc'] * plwPC['nr'])
    for camera in cameras:
        plwPoss = plwPC[camera]['plwPoss']
        wsPlanview[plwPoss] = wsPlanview[plwPoss] + plwPC[camera]['ws']
    imgPlw = np.zeros((plwPC['nr'], plwPC['nc'], 3))
    for camera in cameras:
        plwPoss = plwPC[camera]['plwPoss']
        csPlw = np.round(plwPC['cs'][plwPoss]).astype(int) # nOfPlanviewPositions (no need for round actually)
        rsPlw = np.round(plwPC['rs'][plwPoss]).astype(int) # nOfPlanviewPositions (no need for round actually)
        wsCamera = plwPC[camera]['ws'] # nOfPlanviewPositions
        for corner in range(4):
            csIACamera = plwPC[camera]['csIA'][:, corner] # nOfPlanviewPositions
            rsIACamera = plwPC[camera]['rsIA'][:, corner] # nOfPlanviewPositions
            wsACamera = plwPC[camera]['wsA1'][:, corner]  # nOfPlanviewPositions
            contribution = imgs[camera][rsIACamera, csIACamera, :] * np.outer(wsACamera * wsCamera / wsPlanview[plwPoss], np.ones(3))
            imgPlw[rsPlw, csPlw, :] = imgPlw[rsPlw, csPlw, :] + contribution
    imgPlw = imgPlw.astype(np.uint8)
    return imgPlw
def Dictionary2Array(keys, theDictionary): # 202206291320 (last read 2022-06-29)
    ''' comments:
    .- input keys is a string-list
    .- input theDictionary is a dictionary
    .- output theArray is a ndarray
    '''
    theArray = np.asarray([theDictionary[key] for key in keys])
    return theArray
def DisplayCRInImage(img, cs, rs, options={}): # 202109141700 (last read 2022-07-12)
    ''' comments:
    .- input img is a cv2-image
    .- input cs and rs are integer- or float-ndarrays of the same length (not necessarily within the image)
    .- output img is a cv2-image
    '''
    keys, defaultValues = ['colors', 'imgMargins', 'size'], [[[0, 0, 0]], {'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}, None]
    options = CompleteADictionary(options, keys, defaultValues)
    csIW, rsIW = CR2CRIntegerWithinImage(img.shape[1], img.shape[0], cs, rs, {'imgMargins':options['imgMargins']})
    if len(csIW) == len(rsIW) == 0:
        return img
    if len(options['colors']) == 1:
        colors = [options['colors'][0] for item in range(len(csIW))]
    else:
        assert len(options['colors']) >= len(csIW) == len(rsIW)
        colors = options['colors']
    if options['size'] is not None:
        size = int(options['size'])
    else:
        size = int(np.sqrt(img.shape[0]*img.shape[1])/150) + 1
    for pos in range(len(csIW)):
        img = cv2.circle(img, (csIW[pos], rsIW[pos]), size, colors[pos], -1)
    return img
def DistanceFromAPointToAPoint(x0, y0, x1, y1): # 202206201445 (last read 2022-06-20)
    ''' comments:
    .- input x0, y0, x1 and y1 are floats or float-ndarrays of the same length
    .- output distance is a float or a float-ndarray
    '''
    distance = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    return distance
def ErrorC(xc, yc, zc, mainSet): # 202109231429 (last read 2022-07-06)
    ''' comments:
    .- input xc, yc, zc are floats
    .- input mainSet is a dictionary (including at least 'xc', 'yc' and 'zc')
    .- output errorC is a float
    '''
    errorC = np.sqrt((mainSet['xc'] - xc) ** 2 + (mainSet['yc'] - yc) ** 2 + (mainSet['zc'] - zc) ** 2)
    return errorC
def ErrorG(xs, ys, zs, cs, rs, mainSet): # 202109131100 (last read 2022-07-06)
    ''' comments:
    .- input xs, ys, zs, cs and rs are float-ndarrays of the same length
        .- cs and rs are distorted pixel coordinates
    .- input mainSet is a dictionary
    .- output errorG is a float
    .- uses explicit functions only
    '''
    csR, rsR = XYZ2CDRD(mainSet, xs, ys, zs, options={})[0:2] #! IMP* (considers all positions; returnGoodPositions = False)
    errorG = np.sqrt(np.mean((csR - cs) ** 2 + (rsR - rs) ** 2)) # RMSE (in pixels)
    return errorG
def ErrorH(chs, rhs, horizonLine): # 202109131100 (last read 2022-07-06)
    ''' comments:
    .- input chs and rhs are float-ndarrays of the same length
        .- chs and rhs are distorted pixel coordinates
    .- input horizonLine is a dictionary
    .- output errorH is a float
    .- last revisions without modifications: 20220125
    '''
    rhsR = CDh2RDh(horizonLine, chs, options={})[0] #! IMP* (considers all positions; returnGoodPositions = False)
    errorH = np.sqrt(np.mean((rhsR - rhs) ** 2)) # RMSE (in pixels)
    return errorH
def ErrorT(dataForCal, mainSet, options={}): # 202201250914 (last read 2022-11-09)
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
    if set(keysTMP) <= set(dataForCal.keys()) and all([dataForCal[item] is not None for item in keysTMP]) and dataForCal['aC'] > 1.e-8: # account for errorC
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
def ErrorT2PerturbationFactorAndNOfSeeds(errorT): # 202201270857 (last read 2022-11-09) mutable
    ''' comments:
    .- input errorT is a float
    .- output perturbationFactor is a float
    .- output nOfSeeds is an integer
    '''
    baseForN, log10E = 2., np.log10(max([errorT, 1.])) # 10. for baseForN
    perturbationFactor, nOfSeeds = 0.1 + 0.4 * log10E, 1 * int(baseForN + baseForN * log10E + 2. * baseForN * log10E ** 2) + 2 # IMP* (apply 'x2' nOfSeeds?)
    return perturbationFactor, nOfSeeds
def EulerianAngles2R(ph, sg, ta): # 202109131100 (last read 2022-06-29)
    ''' comments:
    .- input ph, sg and ta are floats
    .- output R is a orthonormal 3x3-float-ndarray positively oriented
    '''
    eu, ev, ef = EulerianAngles2UnitVectors(ph, sg, ta)
    R = UnitVectors2R(eu, ev, ef)
    return R
def EulerianAngles2UnitVectors(ph, sg, ta): # 202109231415 (last read 2022-06-29)
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
    return eu, ev, ef
def FindAFirstSeed(dataBasic, dataForCal, subsetVariablesKeys, subCsetVariablesDict, options={}): # 202211091102 (last read 2022-11-09)
    ''' comments
    .- input dataBasic is a dictionary
    .- input dataForCal is a dictionary (including at least 'nc', 'nr')
    .- input subsetVariablesKeys is a string-list (keys of the selected variables that are to optimize)
    .- input subCsetVariablesDict is a dictionary (keys and values of the selected variables that are given)
    .- output mainSetSeed is a dictionary or None (if it does not succeed, i.e., if errorT >= ctt * imgDiagonal)
    .- output errorTSeed is a float os None (if it does not succeed, i.e., if errorT >= ctt * imgDiagonal)
    '''
    keys, defaultValues = ['counterMax', 'timedeltaMax', 'xc', 'yc', 'zc'], [1000, datetime.timedelta(seconds=600), None, None, None]
    options = CompleteADictionary(options, keys, defaultValues)
    assert 'aG' in dataForCal.keys() and (('chs' not in dataForCal.keys()) or all([item in dataForCal.keys() for item in ['aH', 'chs', 'rhs']]))
    dataForCal['aC'] = 0. # IMP*
    for key in ['xc', 'yc', 'zc']: # camera position
        if key in subCsetVariablesDict.keys():
            dataForCal[key] = subCsetVariablesDict[key] # IMP* 
            dataBasic['refRangesDict'][key] = 0. # IMP*
        elif key not in subCsetVariablesDict.keys() and options[key] is not None:
            dataForCal[key] = options[key]
        else:
            if key in ['xc', 'yc']:
                dataForCal[key] = np.mean(dataForCal[key[0] + 's']) # IMP*
                dataBasic['refRangesDict'][key] = 10. * np.std(dataForCal[key[0] + 's']) # IMP*
            elif key == 'zc':
                dataForCal[key] = np.mean(dataForCal[key[0] + 's']) + 50. # IMP*
                dataBasic['refRangesDict'][key] = 150. # IMP*
    (nc, nr), optionsMainSet = (dataForCal['nc'], dataForCal['nr']), {key:dataBasic[key] for key in ['orderOfHorizonPoly', 'radiusOfEarth']}
    mainSetSeed, errorTSeed, counter, time0, imgDiagonal = None, 1.e+11, 0, datetime.datetime.now(), np.sqrt(nc ** 2 + nr ** 2)
    theArgs = {'dataBasic':dataBasic, 'dataForCal':dataForCal, 'subsetVariablesKeys':subsetVariablesKeys, 'subCsetVariablesDict':subCsetVariablesDict}
    while datetime.datetime.now() - time0 < options['timedeltaMax'] and counter < options['counterMax']: # IMP*
        scaledSubsetVariables = GenerateRandomScaledVariables(dataBasic, subsetVariablesKeys, options={key:dataForCal[key] for key in ['xc', 'yc', 'zc']})
        try: # IMP* to try, WATCH OUT: dataForCal in theArgs must include 'aG', and 'aH' (if 'chs' and 'rhs' are considered)
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
                mainSet = ScaledSubsetVariables2MainSet(dataBasic, scaledSubsetVariables, subsetVariablesKeys, subCsetVariablesDict, nc, nr, options=optionsMainSet)
                assert np.isclose(errorT, ErrorT(dataForCal, mainSet, options={'verbose':False})) # avoidable
                mainSetSeed, errorTSeed = [copy.deepcopy(item) for item in [mainSet, errorT]]
            if errorT < 0.05 * imgDiagonal: # IMP* WATCH OUT
                break
        counter = counter + 1
    return mainSetSeed, errorTSeed
def FindAffineA01(xs0, ys0, xs1, ys1): # 202111241134 (last read 2022-11-10)
    ''' comments:
    .- input xs0, ys0, xs1 and ys1 are float-ndarrays of the same length (>= 3)
    .- output A01 is a 6-float-ndarray or None (if it does not succeed)
    '''
    assert len(xs0) == len(ys0) == len(xs1) == len(ys1) >= 3
    A, b = np.zeros((2 * len(xs0), 6)), np.zeros(2 * len(xs0))
    poss0, poss1 = Poss0AndPoss1InFind2DTransform(len(xs0))
    A[poss0, 0], A[poss0, 1], A[poss0, 2], b[poss0] = xs0, ys0, np.ones(xs0.shape), xs1
    A[poss1, 3], A[poss1, 4], A[poss1, 5], b[poss1] = xs0, ys0, np.ones(xs0.shape), ys1
    try:
        A01 = np.linalg.solve(np.dot(np.transpose(A), A), np.dot(np.transpose(A), b))
    except: # aligned points
        A01 = None
    return A01
def FindGoodPositionsForHomographyHa01ViaRANSAC(xs0, ys0, xs1, ys1, parRANSAC): # 202110141316 (last read 2022-11-09)
    ''' comments:
    .- input xs0, ys0, xs1 and ys1 are float-ndarrays of the same length
    .- input parRANSAC is a dictionary (including at least 'p', 'e', 's' and 'errorC'; see NForRANSAC)
    .- output possGood is an integer-list or None (if it does not succeed)
    '''
    if len(xs0) == len(ys0) == len(xs1) == len(ys1) >= 4:
        N = NForRANSAC(parRANSAC['e'], parRANSAC['p'], parRANSAC['s'])
        possGood = []
        for iN in range(N):
            poss4 = random.sample(range(0, len(xs0)), 4)
            Ha01 = FindHomographyHa01(xs0[poss4], ys0[poss4], xs1[poss4], ys1[poss4])
            if Ha01 is None:
                continue
            xs1R, ys1R = ApplyHomographyHa01(Ha01, xs0, ys0)
            errors = np.sqrt((xs1R - xs1) ** 2 + (ys1R - ys1) ** 2)
            possGoodH = np.where(errors < parRANSAC['errorC'])[0]
            if len(possGoodH) > len(possGood):
                possGood = copy.deepcopy(possGoodH)
        if len(possGood) == 0:
            possGood = None
    else:
        possGood = None
    return possGood
def FindHomographyHa01(xs0, ys0, xs1, ys1): # 202110141311 (last read 2022-11-09)
    ''' comments:
    .- input xs0, ys0, xs1 and ys1 are float-ndarrays of the same length (>= 4)
    .- output Ha01 is a 3x3-float-ndarray or None (if it does not succeed)
        .- Ha01[2,2] is 1
        .- Ha01 allows to transform from 0 to 1
    '''
    assert len(xs0) == len(ys0) == len(xs1) == len(ys1) >= 4
    A, b = np.zeros((2 * len(xs0), 8)), np.zeros(2 * len(xs0))
    poss0, poss1 = Poss0AndPoss1InFind2DTransform(len(xs0))
    A[poss0, 0], A[poss0, 1], A[poss0, 2], A[poss0, 6], A[poss0, 7], b[poss0] = xs0, ys0, np.ones(xs0.shape), -xs0 * xs1, -ys0 * xs1, xs1
    A[poss1, 3], A[poss1, 4], A[poss1, 5], A[poss1, 6], A[poss1, 7], b[poss1] = xs0, ys0, np.ones(xs0.shape), -xs0 * ys1, -ys0 * ys1, ys1
    try:
        sol = np.linalg.solve(np.dot(np.transpose(A), A), np.dot(np.transpose(A), b))
        Ha01 = np.ones((3, 3)) # IMP* initialize with ones, for H[2, 2]
        Ha01[0, 0:3], Ha01[1, 0:3], Ha01[2, 0:2] = sol[0:3], sol[3:6], sol[6:8]
    except: # aligned points
        Ha01 = None
    return Ha01
def FindHomographyHa01ViaRANSAC(xs0, ys0, xs1, ys1, parRANSAC): # 202208110920 (last read 2022-11-09)
    ''' comments:
    .- input xs0, ys0, xs1 and ys1 are float-ndarrays of the same length
    .- input parRANSAC is a dictionary (including at least 'p', 'e', 's' and 'errorC'; see NForRANSAC)
    .- output Ha is a 3x3-float-ndarray or None (if it does not succeed)
        .- Ha01[2,2] is 1
        .- Ha01 allows to transform from 0 to 1
    .- output possGood is an integer-list or None (if it does not succeed)
    '''
    possGood = FindGoodPositionsForHomographyHa01ViaRANSAC(xs0, ys0, xs1, ys1, parRANSAC)
    if possGood is None or len(possGood) < 4:
        Ha01 = None
    else:
        Ha01 = FindHomographyHa01(xs0[possGood], ys0[possGood], xs1[possGood], ys1[possGood])
    return Ha01, possGood
def GCPs2K1asa2(cDs, rDs, xs, ys, zs, oca, ora, k1asa2Min, k1asa2Max, options={}): # 202211091458 (last read 2022-11-09 except details)
    ''' comments:
    .- input cDs, rDs, xs, ys and zs are float-ndarrays of the same length
    .- input oca, ora, k1asa2Min, k1asa2Max are floats
    .- output k1asa2 is a float
    '''
    keys, defaultValues = ['nOfK1asa2'], [1000]
    options = CompleteADictionary(options, keys, defaultValues)
    A0, k1asa2s, errors = XYZ2A0(xs, ys, zs), np.linspace(k1asa2Min, k1asa2Max, options['nOfK1asa2']), np.empty(options['nOfK1asa2'])
    for posK1asa2, k1asa2 in enumerate(k1asa2s):
        cUs, rUs = CDRD2CURUForParabolicSquaredDistortion(cDs, rDs, oca, ora, k1asa2)
        A, b = np.concatenate((A0, CURUXYZ2A1(cUs, rUs, xs, ys, zs)), axis=1), CURU2B(cUs, rUs)
        Pa11 = AB2Pa11(A, b)
        cUsR, rUsR = XYZPa112CURU(xs, ys, zs, Pa11)
        errors[posK1asa2] = np.sqrt(np.mean((cUsR - cUs) ** 2 + (rUsR - rUs) ** 2))
    k1asa2 = k1asa2s[np.argmin(errors)]
    return k1asa2
def GenerateRandomScaledVariables(dataBasic, variablesKeys, options={}): # 202109241335 (last read 2022-07-06)
    ''' comments:
    .- input dataBasic is a dictionary
    .- input variablesKeys is a string-list
    .- output scaledVariables is a float-ndarray of the same length of variablesKeys
    '''
    keys, defaultValues = ['xc', 'yc', 'zc'], None
    options = CompleteADictionary(options, keys, defaultValues)
    ''' comments:
    .- if 'xc'/'yc'/'zc' is in variablesKeys, it has to be in options too
    '''
    variablesDict = {}
    for key in variablesKeys:
        if key in ['xc', 'yc', 'zc']: # dataBasic['refValuesDict'][key] is None
            value0 = options[key]
        else:
            value0 = dataBasic['refValuesDict'][key]
        variablesDict[key] = value0 + Random(-1., +1.) * dataBasic['refRangesDict'][key] # IMP*
        if key == 'zc':
            variablesDict[key] = max([variablesDict[key], options['zc'] / 2.])
    variables = Dictionary2Array(variablesKeys, variablesDict)
    scaledVariables = VariablesScaling(dataBasic, variables, variablesKeys, 'scale')
    return scaledVariables
def IntersectionOfTwoLines(line0, line1, options={}): # 202205260841 (last read 2022-07-02)
    ''' comments:
    .- input line0 and line1 are dictionaries (including at least 'lx', 'ly', 'lt')
    .- output xI and yI are floats or None (if the lines are parallel)
        .- output xI and yI is the point closest to the origin if the lines are coincident
    .- output case is a string ('point', 'coincident' or 'parallel')
    '''
    keys, defaultValues = ['epsilon'], [1.e-11]
    options = CompleteADictionary(options, keys, defaultValues)
    line0, line1 = [NormalizeALine(item) for item in [line0, line1]]
    detT = + line0['lx'] * line1['ly'] - line0['ly'] * line1['lx']
    detX = - line0['lt'] * line1['ly'] + line0['ly'] * line1['lt']
    detY = - line0['lx'] * line1['lt'] + line0['lt'] * line1['lx']
    if np.abs(detT) > options['epsilon']: # point
        xI, yI, case = detX / detT, detY / detT, 'point'
    elif max([np.abs(detX), np.abs(detY)]) < options['epsilon']: # coincident
        (xI, yI), case = PointInALineClosestToAPoint(line0, 0., 0.), 'coincident'
    else: # parallel
        xI, yI, case = None, None, 'parallel'
    return xI, yI, case
def IsVariablesOK(variables, variablesKeys): # 202109241432 (last read 2022-07-01)
    ''' comments:
    .- input variables is a float-ndarray
    .- input variablesKeys is a string-list of the same length of variables
    .- output isVariablesOK is a boolean
    '''
    variablesDict = Array2Dictionary(variablesKeys, variables)
    isVariablesOK = True
    for key in variablesKeys:
        if key in ['zc', 'sca', 'sra']:
            isVariablesOK = isVariablesOK and variablesDict[key] > 0.
        elif key == 'sg':
            isVariablesOK = isVariablesOK and np.abs(variablesDict[key]) <= np.pi / 2.
        elif key == 'ta':
            isVariablesOK = isVariablesOK and 0 <= variablesDict[key] <= np.pi
    return isVariablesOK
def LoadDataBasic0(options={}): # 202109271434 (last read 2022-11-10)
    ''' comments:
    .- output data is a dictionary (does not include station-dependent information)
    '''
    keys, defaultValues = ['nc', 'nr', 'selectedVariablesKeys'], [4000, 3000, ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'sca']] # IMP*
    options = CompleteADictionary(options, keys, defaultValues)
    data = {'date0OfTheWorld':'19000101000000000', 'date1OfTheWorld':'40000101000000000'}
    data['selectedVariablesKeys'] = options['selectedVariablesKeys']
    data['allVariablesKeys'] = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or'] # IMP*
    assert set(['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'sca']) <= set(data['selectedVariablesKeys']) and set(data['selectedVariablesKeys']) <= set(data['allVariablesKeys'])
    data['refValuesDict'], data['refRangesDict'], data['scalesDict'] = {}, {}, {}
    data['refValuesDict']['xc'], data['refRangesDict']['xc'], data['scalesDict']['xc'] = None, 1.0e+1, 1.0e+1
    data['refValuesDict']['yc'], data['refRangesDict']['yc'], data['scalesDict']['yc'] = None, 1.0e+1, 1.0e+1
    data['refValuesDict']['zc'], data['refRangesDict']['zc'], data['scalesDict']['zc'] = None, 1.0e+1, 1.0e+1
    data['refValuesDict']['ph'], data['refRangesDict']['ph'], data['scalesDict']['ph'] = 0.*np.pi/2., np.pi/1., 1.0e+0
    data['refValuesDict']['sg'], data['refRangesDict']['sg'], data['scalesDict']['sg'] = 0.*np.pi/2., np.pi/4., 1.0e+0
    data['refValuesDict']['ta'], data['refRangesDict']['ta'], data['scalesDict']['ta'] = 1.*np.pi/2., np.pi/2., 1.0e+0 # IMP*
    data['refValuesDict']['k1a'], data['refRangesDict']['k1a'], data['scalesDict']['k1a'] = 0.0e+0, 1.0e+0, 1.e-1
    data['refValuesDict']['k2a'], data['refRangesDict']['k2a'], data['scalesDict']['k2a'] = 0.0e+0, 1.0e+0, 1.e-0
    data['refValuesDict']['p1a'], data['refRangesDict']['p1a'], data['scalesDict']['p1a'] = 0.0e+0, 1.0e-2, 1.e-2
    data['refValuesDict']['p2a'], data['refRangesDict']['p2a'], data['scalesDict']['p2a'] = 0.0e+0, 1.0e-2, 1.e-2
    data['refValuesDict']['sca'], data['refRangesDict']['sca'], data['scalesDict']['sca'] = 1.0e-3, 1.0e-3, 1.e-4
    data['refValuesDict']['sra'], data['refRangesDict']['sra'], data['scalesDict']['sra'] = 1.0e-3, 1.0e-3, 1.e-4
    data['refValuesDict']['oc'], data['refRangesDict']['oc'], data['scalesDict']['oc'] = options['nc']/2., options['nc']/20., options['nc']/10.
    data['refValuesDict']['or'], data['refRangesDict']['or'], data['scalesDict']['or'] = options['nr']/2., options['nr']/20., options['nr']/10.
    data['orderOfHorizonPoly'], data['radiusOfEarth'] = 5, 6.371e+6 # IMP*
    return data
def LoadDataPdfTxt(options={}): # 202211101032 (last read 2022-11-10)
    ''' comments:
    .- output data is a dictionary
    .- the number of pixels in each direction is nOfPixels = length * ppm + 1
    '''
    keys, defaultValues = ['dataBasic', 'pathFile', 'planview', 'rewrite'], [None, None, None, True]
    options = CompleteADictionary(options, keys, defaultValues)
    keys, defaultValues = ['xUL', 'yUL', 'angle', 'xyLengthInC', 'xyLengthInR', 'ppm', 'timedeltaTol'], [None, None, None, None, None, None, datetime.timedelta(hours = 1.)]
    options = CompleteADictionary(options, keys, defaultValues)
    if all([options[item] is not None for item in ['xUL', 'yUL', 'angle', 'xyLengthInC', 'xyLengthInR', 'ppm', 'timedeltaTol']]):
        data = {key:options[key] for key in ['xUL', 'yUL', 'angle', 'xyLengthInC', 'xyLengthInR', 'ppm', 'timedeltaTol']}
        pathFile, options['rewrite'] = None, False # IMP* (no place where to rewrite)
    else:
        if options['pathFile'] is not None:
            pathFile = options['pathFile']
        else:
            dataBasic = options['dataBasic']
            pathFile = os.path.join(dataBasic['pathPlanviews'], dataBasic['station'], '{:}pdf{:}.txt'.format(dataBasic['date0OfTheWorld'], options['planview'])) # WATCH OUT
        xUL, yUL, angle, xyLengthInC, xyLengthInR, ppm, timedeltaTolInHours = ReadPdfTxt(pathFile)
        data = {'pathFile':pathFile, 'xUL':xUL, 'yUL':yUL, 'angle':angle, 'xyLengthInC':xyLengthInC, 'xyLengthInR':xyLengthInR, 'ppm':ppm, 'timedeltaTol':datetime.timedelta(hours = timedeltaTolInHours)}
    data['nc'], data['nr'] = [int(np.round(data[item] * data['ppm'])) + 1 for item in ['xyLengthInC', 'xyLengthInR']]
    data['xyLengthInC'], data['xyLengthInR'] = [(data[item] - 1.) / data['ppm'] for item in ['nc', 'nr']]
    assert all([data[item] > 5 for item in ['nc', 'nr']])
    (nc, nr), (csBasic, rsBasic) = [data[item] for item in ['nc', 'nr']], [np.arange(0., data[item]) for item in ['nc', 'nr']]
    csC, rsC = np.asarray([0, nc-1, nc-1, 0]), np.asarray([0, 0, nr-1, nr-1])
    xsC, ysC = PlanCR2XY(csC, rsC, data['xUL'], data['yUL'], data['angle'], data['ppm'], options={})[0:2]
    data['csC'], data['rsC'], data['xsC'], data['ysC'] = csC, rsC, xsC, ysC
    ACR2XY, AXY2CR = FindAffineA01(csC, rsC, xsC, ysC), FindAffineA01(xsC, ysC, csC, rsC)
    data['ACR2XY'], data['AXY2CR'] = ACR2XY, AXY2CR
    data['csBasic'], data['rsBasic'], data['nOfPixels'] = csBasic, rsBasic, nc * nr
    mcs, mrs = np.meshgrid(csBasic, rsBasic)
    data['cs'], data['rs'] = [np.reshape(item, -1) for item in [mcs, mrs]]
    data['xs'], data['ys'] = PlanCR2XY(data['cs'], data['rs'], data['xUL'], data['yUL'], data['angle'], data['ppm'], options={})[0:2]
    xsU, ysU = PlanCR2XY(csBasic, rsBasic[+0] * np.ones(nc), None, None, None, None, options={'ACR2XY':data['ACR2XY']})[0:2] # up
    xsD, ysD = PlanCR2XY(csBasic, rsBasic[-1] * np.ones(nc), None, None, None, None, options={'ACR2XY':data['ACR2XY']})[0:2] # down
    xs0, ys0 = PlanCR2XY(csBasic[+0] * np.ones(nr), rsBasic, None, None, None, None, options={'ACR2XY':data['ACR2XY']})[0:2] # left
    xs1, ys1 = PlanCR2XY(csBasic[-1] * np.ones(nr), rsBasic, None, None, None, None, options={'ACR2XY':data['ACR2XY']})[0:2] # right
    data['polylineU'], data['polylineD'] = {'xs':xsU, 'ys':ysU}, {'xs':xsD, 'ys':ysD}
    data['polyline0'], data['polyline1'] = {'xs':xs0, 'ys':ys0}, {'xs':xs1, 'ys':ys1}
    if options['rewrite']:
        WriteDataPdfTxt(data)
    return data
def MainSet2HorizonLine(mainSet): # 202109141400 (last read 2022-07-06)
    ''' comments:
    .- input mainSet is a dictionary (including at least 'nc', 'nr, 'zc', 'radiusOfEarth', 'ef', 'xc', 'efx', 'yc', 'efy', 'Pa', 'orderOfHorizonPoly')
    .- output horizonLine is a dictionary
    '''
    z0 = 0.
    horizonLine = {key:mainSet[key] for key in ['nc', 'nr']}
    bp = np.sqrt(2. * max([1.e-2, mainSet['zc'] - z0]) * mainSet['radiusOfEarth']) / np.sqrt(np.sum(mainSet['ef'][0:2] ** 2))
    px, py, pz, vx, vy, vz = mainSet['xc'] + bp * mainSet['efx'], mainSet['yc'] + bp * mainSet['efy'], -max([1.e-2, mainSet['zc']-2.*z0]), -mainSet['efy'], +mainSet['efx'], 0.
    dc, cc = np.sum(mainSet['Pa'][0, 0:3] * np.asarray([vx, vy, vz])), np.sum(mainSet['Pa'][0, 0:3] * np.asarray([px, py, pz])) + mainSet['Pa'][0, 3]
    dr, cr = np.sum(mainSet['Pa'][1, 0:3] * np.asarray([vx, vy, vz])), np.sum(mainSet['Pa'][1, 0:3] * np.asarray([px, py, pz])) + mainSet['Pa'][1, 3]
    dd, cd = np.sum(mainSet['Pa'][2, 0:3] * np.asarray([vx, vy, vz])), np.sum(mainSet['Pa'][2, 0:3] * np.asarray([px, py, pz])) + 1.
    ccUh1, crUh1, ccUh0 = dr * cd - dd * cr, dd * cc - dc * cd, dc * cr - dr * cc
    TMP = max([np.sqrt(ccUh1 ** 2 + crUh1 ** 2), 1.e-8])
    horizonLine['ccUh1'], horizonLine['crUh1'], horizonLine['ccUh0'] = [item / TMP for item in [ccUh1, crUh1, ccUh0]]
    horizonLine['crUh1'] = ClipWithSign(horizonLine['crUh1'], 1.e-8, 1.e+8)
    cUhs = np.linspace(-0.1 * mainSet['nc'], +1.1 * mainSet['nc'], 31, endpoint=True)
    rUhs = CUh2RUh(horizonLine, cUhs)
    cDhs, rDhs = CURU2CDRD(mainSet, cUhs, rUhs) # explicit
    A = np.ones((len(cDhs), mainSet['orderOfHorizonPoly'] + 1))
    for n in range(1, mainSet['orderOfHorizonPoly'] + 1):
        A[:, n] = cDhs ** n
    b = rDhs
    try:
        horizonLine['ccDh'] = np.linalg.solve(np.dot(np.transpose(A), A), np.dot(np.transpose(A), b))
        if np.max(np.abs(b - np.dot(A, horizonLine['ccDh']))) > 5e-1: # IMP* WATCH OUT
            horizonLine['ccDh'] = np.zeros(mainSet['orderOfHorizonPoly'] + 1)
            horizonLine['ccDh'][0] = 1.e+2 # IMP* WATCH OUT
    except:
        horizonLine['ccDh'] = np.zeros(mainSet['orderOfHorizonPoly'] + 1)
        horizonLine['ccDh'][0] = 1.e+2 # IMP* WATCH OUT
    return horizonLine
def MainSet2Pa(mainSet): # 202207061434 (last read 2022-07-06)
    ''' comments:
    .- input mainSet is a dictionary (including at least 'pc', 'eu', 'ev', 'ef', 'sca', 'sra', 'oc' and 'or'
    '''
    tu, tv, tf = [-np.sum(mainSet['pc'] * mainSet[item]) for item in ['eu', 'ev', 'ef']]
    P = np.zeros((3, 4))
    P[0, 0:3], P[0, 3] = mainSet['eu'] / mainSet['sca'] + mainSet['oc'] * mainSet['ef'], tu / mainSet['sca'] + mainSet['oc'] * tf
    P[1, 0:3], P[1, 3] = mainSet['ev'] / mainSet['sra'] + mainSet['or'] * mainSet['ef'], tv / mainSet['sra'] + mainSet['or'] * tf
    P[2, 0:3], P[2, 3] = mainSet['ef'], tf
    Pa = P / P[2, 3]
    return Pa
def MakeFolder(pathFolder): # 202109131100 (last read 2022-06-29)
    ''' comments:
    .- input pathFolder is a string
        .- pathFolder is created if it does not exist
    '''
    if not os.path.exists(pathFolder):
        os.makedirs(pathFolder)
    return None
def N2K(n): # 202109131100 (last read on 2022-06-27)
    ''' comments:
    .- input n is an integer or float
    .- output k is a float
    '''
    k = (n - 1.) / 2.
    return k
def NForRANSAC(eRANSAC, pRANSAC, sRANSAC): # 202211091432 (last read 2022-11-09)
    ''' comments:
    .- input eRANSAC is a float (probability of a point being "bad")
    .- input pRANSAC is a float (goal probability of sRANSAC points being "good")
    .- input sRANSAC is an integer (number of points of the model)
    .- output N is an integer
    .- note that: (1 - e) ** s is the probability of a set of s points being good
    .- note that: 1 - (1 - e) ** s is the probability of a set of s points being bad (at least one is bad)
    .- note that: (1 - (1 - e) ** s) ** N is the probability of choosing N sets all being bad
    .- note that: 1 - (1 - (1 - e) ** s) ** N is the probability of choosing N sets where at least one set if good
    .- note that: from 1 - (1 - (1 - e) ** s) ** N = p -> 1 - p = (1 - (1 - e) ** s) ** N and we get the expression
    '''
    N = int(np.log(1. - pRANSAC) / np.log(1. - (1. - eRANSAC) ** sRANSAC)) + 1
    return N
def NonlinearManualCalibration(dataBasic, dataForCal, freeVariablesKeys, givenVariablesDict, options={}): # 202211091206 (last read 2022-11-09)
    ''' comments:
    .- input dataBasic is a dictionary
    .- input dataForCal is a dictionary (including at least 'nc', 'nr', 'cs', 'rs', 'xs', 'ys', 'zs', 'aG')
        .- optional relevant keys: 'chs', 'rhs', 'aH' and 'mainSetSeeds'
    .- input freeVariablesKeys is a string-list
    .- input givenVariablesDict is a dictionary
    .- output mainSet is a dictionary or None (if it does not succeed)
    .- output errorT is a float or None (if it does not succeed)
    '''
    assert 'aG' in dataForCal.keys() and (('chs' not in dataForCal.keys()) or all([item in dataForCal.keys() for item in ['aH', 'chs', 'rhs']]))
    keys, defaultValues = ['timedeltaMax', 'xc', 'yc', 'zc'], [datetime.timedelta(seconds=100.), None, None, None]
    options = CompleteADictionary(options, keys, defaultValues)
    if len(dataForCal['xs']) < int((len(freeVariablesKeys) + 1.) / 2.):
        return None, None
    imgDiagonal = np.sqrt(dataForCal['nc'] ** 2 + dataForCal['nr'] ** 2)
    if 'mainSetSeeds' in dataForCal.keys() and dataForCal['mainSetSeeds'] is not None and len(dataForCal['mainSetSeeds']) > 0:
        mainSetSeed0, errorTSeed0 = ReadAFirstSeed(dataBasic, dataForCal, freeVariablesKeys, givenVariablesDict, dataForCal['mainSetSeeds'])
        assert all([np.isclose(givenVariablesDict[key], mainSetSeed0[key]) for key in givenVariablesDict.keys()]) # avoidable
    else:
        mainSetSeed0, errorTSeed0 = None, 1.e+11 # IMP* the same values as in ReadAFirstSeed
    if mainSetSeed0 is not None and errorTSeed0 < 0.2 * imgDiagonal: # IMP* WATCH OUT
        mainSetSeed, errorTSeed = [copy.deepcopy(item) for item in [mainSetSeed0, errorTSeed0]]
    else:
        optionsTMP = {key:options[key] for key in ['timedeltaMax', 'xc', 'yc', 'zc']}
        mainSetSeed, errorTSeed = FindAFirstSeed(dataBasic, dataForCal, freeVariablesKeys, givenVariablesDict, options=optionsTMP)
    freeVariablesSeed = AllVariables2SubsetVariables(dataBasic, mainSetSeed['allVariables'], freeVariablesKeys)
    scaledFreeVariablesSeed = VariablesScaling(dataBasic, freeVariablesSeed, freeVariablesKeys, 'scale')
    perturbationFactor, nOfSeeds = ErrorT2PerturbationFactorAndNOfSeeds(errorTSeed)
    (nc, nr), optionsMainSet = (dataForCal['nc'], dataForCal['nr']), {key:dataBasic[key] for key in ['orderOfHorizonPoly', 'radiusOfEarth']}
    theArgs = {'dataBasic':dataBasic, 'dataForCal':dataForCal, 'subsetVariablesKeys':freeVariablesKeys, 'subCsetVariablesDict':givenVariablesDict}
    mainSetO, errorTO, scaledFreeVariablesO = [copy.deepcopy(item) for item in [mainSetSeed, errorTSeed, scaledFreeVariablesSeed]]
    for iOfSeeds in range(nOfSeeds): # monteCarlo
        if iOfSeeds == 0: # IMP* to ensure that the seed is considered
            perturbationFactorH = 0.
        else:
            perturbationFactorH = 1. * perturbationFactor
        scaledFreeVariablesP = PerturbateScaledVariables(dataBasic, scaledFreeVariablesO, freeVariablesKeys, options={'perturbationFactor':perturbationFactorH})
        try: # IMP* to try
            errorTP = ScaledSubsetVariables2FTM(scaledFreeVariablesP, theArgs)
        except: # IMP* not to inform
            continue
        if errorTP >= 1.0 * imgDiagonal:
            continue # IMP* WATCH OUT
        try:
            scaledFreeVariablesP = optimize.minimize(ScaledSubsetVariables2FTM, scaledFreeVariablesP, args=(theArgs), callback=MinimizeStopper(5.)).x # IMP*
            errorTP = ScaledSubsetVariables2FTM(scaledFreeVariablesP, theArgs)
        except: 
            continue
        if errorTP < errorTO:
            if not IsVariablesOK(scaledFreeVariablesP, freeVariablesKeys): 
                continue
            mainSetP = ScaledSubsetVariables2MainSet(dataBasic, scaledFreeVariablesP, freeVariablesKeys, givenVariablesDict, nc, nr, options=optionsMainSet)
            assert np.isclose(errorTP, ErrorT(dataForCal, mainSetP, options={'verbose':False})) # avoidable
            xs, ys, zs = dataForCal['xs'], dataForCal['ys'], dataForCal['zs']
            if not (len(XYZ2PositionsRightSideOfCamera(mainSetP, xs, ys, zs)) == len(xs) == len(ys) == len(zs)): 
                continue
            mainSetO, errorTO, scaledFreeVariablesO = [copy.deepcopy(item) for item in [mainSetP, errorTP, scaledFreeVariablesP]]
            perturbationFactor = ErrorT2PerturbationFactorAndNOfSeeds(errorTO)[0]
        if errorTO < 0.1: # IMP* interesting for RANSAC WATCH OUT
            break
    return mainSetO, errorTO
def NonlinearManualCalibrationOfSeveralImages(dataBasic, dataForCals, freeDVariablesKeys, freeUVariablesKeys, givenVariablesDict, options={}): # 202211101215 (last read 2022-11-10)
    ''' comments:
    .- input dataBasic is a dictionary
    .- input dataForCals is a dictionary (including at least 'ncs', 'nrs', 'css', 'rss', 'xss', 'yss', 'zss', 'aG', 'chss', 'rhss', 'aH' and 'mainSetSeeds')
    .- input freeDVariablesKeys is a string-list (variables that are different -not unique- and not given)
    .- input freeUVariablesKeys is a string-list (variables that are unique and not given)
    .- input givenVariablesDict is a dictionary (variables that are given)
    .- output mainSetsO is a dictionary
    .- output errorTsO is a float
    '''
    keys, defaultValues = ['nOfSeedsForC', 'orderOfHorizonPoly', 'radiusOfEarth'], [20, 5, 6.371e+6]
    options = CompleteADictionary(options, keys, defaultValues)
    freeDVariabless, freeUVariabless = [], []
    for pos in range(len(dataForCals['ncs'])):
        dataForCal = {key:dataForCals[key] for key in ['aG', 'aH', 'mainSetSeeds']}
        for key in ['ncs', 'nrs', 'css', 'rss', 'xss', 'yss', 'zss', 'chss', 'rhss']:
            dataForCal[key[0:-1]] = dataForCals[key][pos]
        mainSet, errorT = NonlinearManualCalibration(dataBasic, dataForCal, freeDVariablesKeys+freeUVariablesKeys, givenVariablesDict, options={})
        freeDVariabless.append(AllVariables2SubsetVariables(dataBasic, mainSet['allVariables'], freeDVariablesKeys, options={}))
        freeUVariabless.append(AllVariables2SubsetVariables(dataBasic, mainSet['allVariables'], freeUVariablesKeys, options={}))
    freeUVariables0 = np.average(np.asarray(freeUVariabless), axis=0)
    assert freeUVariables0.shape == freeUVariabless[0].shape # avoidable
    givenVariablesKeys = list(givenVariablesDict)
    givenVariables = Dictionary2Array(givenVariablesKeys, givenVariablesDict)
    ctrlFirst, mainSetsO, errorTsO = True, [{} for item in range(len(dataForCals['ncs']))], np.zeros(len(dataForCals['ncs']))
    while True:
        print('... iterating (to obtain unique parameters) ...')
        freeUVariablesDict0 = Array2Dictionary(freeUVariablesKeys, freeUVariables0)
        if ctrlFirst:
            freeUVariables = copy.deepcopy(freeUVariables0)
            optionsMainSet = {key:options[key] for key in ['orderOfHorizonPoly', 'radiusOfEarth']}
            mainSets, errorTs = [[] for item in range(2)]
            for pos in range(len(dataForCals['ncs'])):
                dataForCal = {key:dataForCals[key] for key in ['aG', 'aH', 'mainSetSeeds']}
                for key in ['ncs', 'nrs', 'css', 'rss', 'xss', 'yss', 'zss', 'chss', 'rhss']:
                    dataForCal[key[0:-1]] = dataForCals[key][pos]
                subCsetVariablesDictTMP, optionsTMP = {**freeUVariablesDict0, **givenVariablesDict}, {key:dataForCal[key] for key in ['nc', 'nr']}
                allVariables = SubsetVariables2AllVariables(dataBasic, freeDVariabless[pos], freeDVariablesKeys, subCsetVariablesDictTMP, options=optionsTMP)
                mainSet = AllVariables2MainSet(allVariables, dataForCal['nc'], dataForCal['nr'], options=optionsMainSet)
                errorT = ErrorT(dataForCal, mainSet)
                mainSets.append(mainSet); errorTs.append(errorT)
            ctrlFirst = False
        freeUVariablesSeed, optionsTMP = 1. * freeUVariables0, {key:options[key] for key in ['nOfSeedsForC', 'orderOfHorizonPoly', 'radiusOfEarth']}
        freeUVariables, errorT = UpdateFreeUVariables(dataBasic, dataForCals, freeDVariabless, freeDVariablesKeys, freeUVariablesSeed, freeUVariablesKeys, givenVariables, givenVariablesKeys, options=optionsTMP)
        freeUVariablesDict = Array2Dictionary(freeUVariablesKeys, freeUVariables)
        errorTsOld = copy.deepcopy(errorTs)
        for pos in range(len(dataForCals['ncs'])): # IMPROVABLE
            dataForCal = {key:dataForCals[key] for key in ['aG', 'aH', 'mainSetSeeds']}
            for key in ['ncs', 'nrs', 'css', 'rss', 'xss', 'yss', 'zss', 'chss', 'rhss']:
                dataForCal[key[0:-1]] = dataForCals[key][pos]
            dataForCal['mainSetSeeds'] = dataForCal['mainSetSeeds'] + mainSets
            subCsetVariablesDictTMP = {**freeUVariablesDict, **givenVariablesDict}
            mainSet, errorT = NonlinearManualCalibration(dataBasic, dataForCal, freeDVariablesKeys, subCsetVariablesDictTMP, options={})
            freeDVariabless[pos], mainSets[pos], errorTs[pos] = AllVariables2SubsetVariables(dataBasic, mainSet['allVariables'], freeDVariablesKeys, options={}), mainSet, errorT
        if max(errorTs) < max(errorTsOld) * 0.999:
            mainSetsO, errorTsO = mainSets, errorTs
        else:
            break
    return mainSetsO, errorTsO
def NormalizeALine(line, options={}): # 202206201459 (last read 2022-06-29)
    ''' comments:
    .- input line is a dictionary (including at least 'lx', 'ly' and 'lt')
        .- a line is so that line['lx'] * x + line['ly'] * y + line['lt'] = 0
        .- a normalized line is so that line['lx'] ** 2 + line['ly'] ** 2 = 1
    .- output line includes key 'isNormalized' (=True)
    .- output line maintains the orientation of input line
    '''
    keys, defaultValues = ['forceToNormalize'], [False]
    options = CompleteADictionary(options, keys, defaultValues)
    if options['forceToNormalize'] or 'isNormalized' not in line.keys() or not line['isNormalized']:
        lm = np.sqrt(line['lx'] ** 2 + line['ly'] ** 2)
        line = {item:line[item]/lm for item in ['lx', 'ly', 'lt']}
        line['isNormalized'] = True
    return line
def ORBKeypoints(img, options={}): # 202109221400 (last read 202207-05)
    ''' comments:
    .- input img is a cv2-image or a string (path to an image)
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
def ORBKeypointsForAllImagesInAFolder(pathFld, options={}): # 202207051012 (last read 202207-05)
    ''' comments:
    .- input pathFld is a string
    .- output fnsImgs, ncs, nrs, kpss and dess are lists (including Nones when it does not succeed)
    '''
    keys, defaultValues = ['exts', 'mask', 'nOfFeatures', 'recursive'], [['jpg', 'jpeg', 'png'], None, 5000, False]
    options = CompleteADictionary(options, keys, defaultValues)
    if options['recursive']:
        pathsImgs = [os.path.join(x[0], fn) for x in os.walk(pathFld) for fn in x[2] if os.path.splitext[fn][1:] in options['exts']]
    else:
        pathsImgs = [os.path.join(pathFld, fn) for fn in os.listdir(pathFld) if os.path.splitext(fn)[1][1:] in options['exts']] #
    fnsImgs, ncs, nrs, kpss, dess = ORBKeypointsForPathsImages(pathsImgs, options={key:options[key] for key in ['mask', 'nOfFeatures']})
    return fnsImgs, ncs, nrs, kpss, dess
def ORBKeypointsForPathsImages(pathsImgs, options={}): # 202202011455 (last read 2022-05)
    ''' comments:
    .- input pathsImgs is a list of strings
    .- output fnsImgs, ncs, nrs, kpss and dess are lists (including Nones when it does not succeed)
    '''
    keys, defaultValues = ['mask', 'nOfFeatures'], [None, 5000]
    options = CompleteADictionary(options, keys, defaultValues)
    fnsImgs, ncs, nrs, kpss, dess = [[] for item in range(5)]
    for pathImg in pathsImgs:
        fnsImgs.append(os.path.split(pathImg)[1])
        nc, nr, kps, des = ORBKeypoints(pathImg, options={key:options[key] for key in ['mask', 'nOfFeatures']})[0:4]
        ncs.append(nc); nrs.append(nr); kpss.append(kps); dess.append(des)
    return fnsImgs, ncs, nrs, kpss, dess
def ORBMatches(kps1, des1, kps2, des2, options={}): # 202109131700 (last read 202207-05)
    ''' comments:
    .- input kps1 and des1 are ORB keys and descriptions for image1 (see ORBKeypoints)
    .- input kps2 and des2 are ORB keys and descriptions for image2 (see ORBKeypoints)
    .- output cs1, rs1, cs2, rs2 and ers are float-ndarrays
        .- compares both ways, so that commutativity is ensured
    '''
    keys, defaultValues = ['erMaximum', 'nOfStd'], [20., 2.]
    options = CompleteADictionary(options, keys, defaultValues)
    cs1, rs1 = [np.asarray([item.pt[pos] for item in kps1]) for pos in [0, 1]]
    cs2, rs2 = [np.asarray([item.pt[pos] for item in kps2]) for pos in [0, 1]]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches12 = sorted(bf.match(des1, des2), key = lambda x:x.distance)
    matches21 = sorted(bf.match(des2, des1), key = lambda x:x.distance)
    poss1 = [match.queryIdx for match in matches12] + [match.trainIdx for match in matches21]
    poss2 = [match.trainIdx for match in matches12] + [match.queryIdx for match in matches21]
    cs1, rs1, cs2, rs2 = cs1[poss1], rs1[poss1], cs2[poss2], rs2[poss2]
    ers = np.asarray([match.distance for match in matches12] + [match.distance for match in matches21])
    cs1, rs1, cs2, rs2, ers = np.unique(np.asarray([cs1, rs1, cs2, rs2, ers]), axis=1) # IMP* interesting
    ds = np.sqrt((cs1 - cs2) ** 2 + (rs1 - rs2) ** 2)
    possGood = np.where((ers < options['erMaximum']) & (ds < np.mean(ds) + options['nOfStd'] * np.std(ds) + 1.e-8))[0]
    cs1, rs1, cs2, rs2, ers = [item[possGood] for item in [cs1, rs1, cs2, rs2, ers]]
    return cs1, rs1, cs2, rs2, ers
def Pa2Pa11(Pa): # 202201250835 (last read 2022-07-06)
    ''' comments:
    .- input Pa is a 3x4-float-ndarray
    .- output Pa11 is a 11-float-ndarray
    '''
    Pa11 = np.ones(11)
    Pa11[0:4], Pa11[4:8], Pa11[8:11] = Pa[0, 0:4], Pa[1, 0:4], Pa[2, 0:3]
    return Pa11
def PathImgOrImg2Img(img): # 202205140937 (last read 2022-07-05)
    ''' comments:
    .- input img is a cv2-image or a string
    .- output img is a cv2-image
    '''
    try: # test if it an image
        img[img.shape[0]-1, img.shape[1]-1, :] - 1
    except:
        img = cv2.imread(img)
    img[img.shape[0]-1, img.shape[1]-1, :] - 1
    return img
def PerturbateScaledVariables(dataBasic, scaledVariables, variablesKeys, options={}): # 202109241340 (last read 2022-11-09)
    ''' comments:
    .- input dataBasic is a dictionary
    .- input scaledVariables is a float-ndarray
    .- input variablesKeys is a string-list of the same length of scaledVariables
    .- output scaledVariables is a float-ndarray of the same length of input scaledVariables
    '''
    keys, defaultValues = ['perturbationFactor'], [1.]
    options = CompleteADictionary(options, keys, defaultValues)
    variables = VariablesScaling(dataBasic, scaledVariables, variablesKeys, 'unscale')
    variablesDict = Array2Dictionary(variablesKeys, variables)
    for key in variablesKeys:
        variablesDict[key] = variablesDict[key] + options['perturbationFactor'] * Random(-1., +1.) * dataBasic['scalesDict'][key] # IMP*, in unscaled
    variables = Dictionary2Array(variablesKeys, variablesDict)
    scaledVariables = VariablesScaling(dataBasic, variables, variablesKeys, 'scale')
    return scaledVariables
def PixelsErrorOfRotationalHomographyUsingUVUas(x, theArgs): # 190001010000 (last read 2022-11-09)
    ''' comments:
    .- input x is a 3-float-ndarray (ph, sg and ta)
    .- input theArgs is a dictionary (including at least 'R1', 'uUas0', 'vUas0', 'uUas1', 'vUas1' and 'sca')
    '''
    R1, uUas0, vUas0, uUas1, vUas1, sca = [theArgs[item] for item in ['R1', 'uUas0', 'vUas0', 'uUas1', 'vUas1', 'sca']]
    R0 = EulerianAngles2R(x[0], x[1], x[2])
    H01 = np.dot(R1, np.transpose(R0)) # 0 is unknown, 1 is known
    uUas1R, vUas1R = ApplyHomographyHa01(H01, uUas0, vUas0)
    f = np.sqrt(np.mean((uUas1R - uUas1) ** 2 + (vUas1R - vUas1) ** 2)) / sca
    return f
def PlanCR2XY(cs, rs, xUL, yUL, angle, ppm, options={}): # 202211100819 (last read 2022-11-10)
    ''' comments:
    .- input cs and rs are float-arrays of the same length
    .- input xUL and yUL are floats: the upper left corner of the planview (or None, for it is actually optional)
    .- input angle is a float: 0 = E, pi/2 = N (or None, for it is actually optional)
    .- input ppm is a float (or None, for it is actually optional)
    .- output xs and ys are float-arrays of the same length
    .- output possGood is an integer-list or None
    '''
    imgMargins0 = {'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}
    keys, defaultValues = ['ACR2XY', 'imgMargins', 'nc', 'nr', 'returnGoodPositions'], [None, imgMargins0, None, None, False]
    options = CompleteADictionary(options, keys, defaultValues)
    if options['ACR2XY'] is not None: # apply an affine transformation
        xs, ys = ApplyAffineA01(options['ACR2XY'], cs, rs)
    else:
        us = + cs / ppm
        vs = - rs / ppm
        xs = + np.cos(angle) * us - np.sin(angle) * vs + xUL # IMP* angle = 0: xs = us + xUL; angle = 90: xs = -vs + xUL
        ys = + np.sin(angle) * us + np.cos(angle) * vs + yUL # IMP* angle = 0: ys = vs + xUL; angle = 90: ys = +us + yUL
    if options['returnGoodPositions']:
        possGood = CR2PositionsWithinImage(options['nc'], options['nr'], cs, rs, options={'imgMargins':options['imgMargins']})
    else:
        possGood = None
    return xs, ys, possGood
def PlanXY2CR(xs, ys, xUL, yUL, angle, ppm, options={}): # 202205140759 (last read 2022-11-10)
    ''' comments:
    .- input xs and ys are float-arrays of the same length
    .- input xUL and yUL are floats: the upper left corner of the planview (or None, for it is actually optional)
    .- input angle is a float: 0 = E, pi/2 = N (or None, for it is actually optional)
    .- input ppm is a float (or None, for it is actually optional)
    .- output cs and rs are float-arrays of the same length
    .- output possGood is an integer-list or None
    '''
    imgMargins0 = {'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}
    keys, defaultValues = ['AXY2CR', 'imgMargins', 'nc', 'nr', 'returnGoodPositions'], [None, imgMargins0, None, None, False]
    options = CompleteADictionary(options, keys, defaultValues)
    if options['AXY2CR'] is not None: # apply an affine transformation
        cs, rs = ApplyAffineA01(options['AXY2CR'], xs, ys)
    else:
        xs = xs - xUL
        ys = ys - yUL
        us = + np.cos(angle) * xs + np.sin(angle) * ys
        vs = - np.sin(angle) * xs + np.cos(angle) * ys
        cs = + us * ppm
        rs = - vs * ppm
    if options['returnGoodPositions']:
        possGood = CR2PositionsWithinImage(options['nc'], options['nr'], cs, rs, options={'imgMargins':options['imgMargins']})
    else:
        possGood = None
    return cs, rs, possGood
def PlanviewPrecomputations(mainSets, dataPdfTxt, z): # 202205140802 (last read 2022-11-10)
    ''' comments:
    .- input mainSets is a dictionary of dictionaries (the keys are the cameras)
    .- input dataPdfTxt is a dictionary (including at least 'nc', 'nr', 'cs', 'rs', 'xs' and 'ys')
        .- dataPdfTxt['cs'], .['rs'], .['xs'] and .['ys'] are float-ndarrays of the same length
        .- dataPdfTxt['cs'], .['rs'] are pixel coordinates in the planview
    .- input z is a float
    .- output plwPC is a dictionary (including 'nc', 'nr', 'cs', 'rs', 'xs', 'ys', 'zs', 'cameras' and all 'camera')
    '''
    plwPC = {key:dataPdfTxt[key] for key in ['nc', 'nr', 'cs', 'rs', 'xs', 'ys']}
    plwPC['zs'] = z * np.ones(len(plwPC['xs']))
    plwPC['cameras'] = []
    for camera in sorted(mainSets.keys()):
        ncCamera, nrCamera, mainSet = mainSets[camera]['nc'], mainSets[camera]['nr'], mainSets[camera]
        optionsTMP = {'imgMargins':{'c0':2, 'c1':2, 'r0':2, 'r1':2, 'isComplete':True}, 'returnGoodPositions':True}
        csCamera, rsCamera, plwPossInCamera = XYZ2CDRD(mainSet, plwPC['xs'], plwPC['ys'], plwPC['zs'], options=optionsTMP)
        if len(plwPossInCamera) == 0:
            continue
        csCamera, rsCamera = csCamera[plwPossInCamera], rsCamera[plwPossInCamera]
        ws = CRWithinImage2NormalizedLengthsAndAreas(ncCamera, nrCamera, csCamera, rsCamera)[0] # normalized lengths; len(plwPossInCamera)
        csIA, rsIA, wsA1 = CR2CRIntegerAroundAndWeights(csCamera, rsCamera) # len(plwPossInCamera) x 4
        plwPC['cameras'].append(camera) # IMP*
        plwPC[camera] = {'plwPoss':plwPossInCamera, 'ws':ws, 'csIA':csIA, 'rsIA':rsIA, 'wsA1':wsA1}
    return plwPC
def PlotMainSet(img, mainSet, cs, rs, xs, ys, zs, chs, rhs, pathImgOut): # 202111171719 (last read 2022-11-09)
    ''' comments:
    .- input img is a cv2-image or a string
    .- input mainSet is a dictionary (including at least 'horizonLine')
    .- input cs, rs, xs, ys and zs are float-ndarrays of the same length
    .- input chs and rhs are float-ndarrays or None
    .- input pathImgOut is a string
    '''
    img = PathImgOrImg2Img(img)
    nr, nc = img.shape[0:2]
    img = DisplayCRInImage(img, cs, rs, options={'colors':[[0, 0, 0]], 'size':np.sqrt(nc*nr)/200}) # IMP* clicked
    csR, rsR = XYZ2CDRD(mainSet, xs, ys, zs, options={})[0:2]
    img = DisplayCRInImage(img, csR, rsR, options={'colors':[[0, 255, 255]], 'size':np.sqrt(nc*nr)/400}) # IMP* from calibration and x, y, z
    chsR, rhsR = np.arange(0, nc, 1), CDh2RDh(mainSet['horizonLine'], np.arange(0, nc, 1), options={})[0] #!
    img = DisplayCRInImage(img, chsR, rhsR, options={'colors':[[0, 255, 255]], 'size':1}) # IMP* from calibration
    if all([item is not None for item in [chs, rhs]]):
        img = DisplayCRInImage(img, chs, rhs, options={'colors':[[0, 0, 0]], 'size':np.sqrt(nc*nr)/200}) # IMP* clicked
    cv2.imwrite(pathImgOut, img)
    return None
def PointInALineClosestToAPoint(line, x, y): # 202205260853 (last read 2022-06-20, checked graphically with auxiliar code)
    ''' comments:
    .- input line is a dictionary (including at least 'lx', 'ly' and 'lt')
    .- input x and y are floats or float-ndarrays of the same length
    .- output xC and yC are floats or float-ndarrays
    '''
    line = NormalizeALine(line)
    xC = line['ly'] * (line['ly'] * x - line['lx'] * y) - line['lx'] * line['lt']
    yC = line['lx'] * (line['lx'] * y - line['ly'] * x) - line['ly'] * line['lt']
    return xC, yC
def Poss0AndPoss1(n): # 202201250804 (last read 2022-07-06)
    '''
    .- input n is an integer
    .- output poss0 and poss1 are n-integer-list
    '''
    poss0 = [2*pos+0 for pos in range(n)]
    poss1 = [2*pos+1 for pos in range(n)]
    return poss0, poss1
def Poss0AndPoss1InFind2DTransform(n): # 202207112001 (last read 2022-11-09)
    ''' comments:
    .- input n is an integer
    .- output poss0 and poss1 are integer-lists
    '''
    poss0 = [2*pos+0 for pos in range(n)]
    poss1 = [2*pos+1 for pos in range(n)]
    return poss0, poss1
def R2UnitVectors(R): # 202109131100 (last read 2022-06-29)
    ''' comments:
    .- input R is a 3x3-float-ndarray
        .- the rows of R are eu, ev and ef
    .- output eu, ev and ef are 3-float-ndarrays
    '''
    assert R.shape == (3, 3)
    eu, ev, ef = R[0, :], R[1, :], R[2, :]
    return eu, ev, ef
def RANSACForGCPs(cDs, rDs, xs, ys, zs, oca, ora, eRANSAC, pRANSAC, ecRANSAC, NForRANSACMax, options={}): # 202211101252 (last read 2022-11-10 except details)
    if len(cDs) < 6:
        return None, None
    keys, defaultValues = ['nOfK1asa2'], [1000]
    options = CompleteADictionary(options, keys, defaultValues)
    dD2Max = np.max((cDs - oca) ** 2 + (rDs - ora) ** 2)
    k1asa2 = 0. # k1a * sca ** 2
    possGood = RANSACForGCPsAndK1asa2(cDs, rDs, xs, ys, zs, oca, ora, k1asa2, eRANSAC, pRANSAC, 3. * ecRANSAC, NForRANSACMax) # IMP "3x"
    cDsSel, rDsSel, xsSel, ysSel, zsSel = [item[possGood] for item in [cDs, rDs, xs, ys, zs]]
    k1asa2Min, k1asa2Max = -4./(27.*dD2Max)+1.e-11, 4./(27.*dD2Max) # IMP* (see notes)
    k1asa2 = GCPs2K1asa2(cDsSel, rDsSel, xsSel, ysSel, zsSel, oca, ora, k1asa2Min, k1asa2Max, options={'nOfK1asa2':options['nOfK1asa2']})
    possGood = RANSACForGCPsAndK1asa2(cDs, rDs, xs, ys, zs, oca, ora, k1asa2, eRANSAC, pRANSAC, 1. * ecRANSAC, NForRANSACMax)
    cDsSel, rDsSel, xsSel, ysSel, zsSel = [item[possGood] for item in [cDs, rDs, xs, ys, zs]]
    k1asa2Min, k1asa2Max = -4./(27.*dD2Max)+1.e-11, 4./(27.*dD2Max) # IMP* (see notes)
    k1asa2 = GCPs2K1asa2(cDsSel, rDsSel, xsSel, ysSel, zsSel, oca, ora, k1asa2Min, k1asa2Max, options={'nOfK1asa2':options['nOfK1asa2']})
    return possGood, k1asa2
def RANSACForGCPsAndK1asa2(cDs, rDs, xs, ys, zs, oca, ora, k1asa2, eRANSAC, pRANSAC, ecRANSAC, NForRANSACMax): # 202211101247 (last read 2022-11-10 except details)
    ''' comments:
    .- input cDs, rDs, xs, ys and zs are float-ndarrays of the same length
    .- input oca, ora and k1asa2 are floats
    .- input eRANSAC, pRANSAC and ecRANSAC are floats
    .- input NForRANSACMax is an integer
    .- output possGood is a integer-list
    '''
    cUs, rUs = CDRD2CURUForParabolicSquaredDistortion(cDs, rDs, oca, ora, k1asa2)
    AAll, bAll = CURUXYZ2A(cUs, rUs, xs, ys, zs), CURU2B(cUs, rUs)
    sRANSAC = 6 # (to obtain 12 equations >= 11 unkowns)
    N, possGood = min(NForRANSACMax, NForRANSAC(eRANSAC, pRANSAC, sRANSAC)), []
    for iN in range(N):
        possH = random.sample(range(0, len(cUs)), sRANSAC)
        poss01 = [2*item for item in possH] + [2*item+1 for item in possH] # who cares about the order (both A and b suffer the same)
        A, b = AAll[poss01, :], bAll[poss01]
        try:
            Pa11 = AB2Pa11(A, b)
            cUsR, rUsR = XYZPa112CURU(xs, ys, zs, Pa11)
            errors = np.sqrt((cUsR - cUs) ** 2 + (rUsR - rUs) ** 2)
        except:
            continue
        possGoodH = np.where(errors <= ecRANSAC)[0]
        if len(possGoodH) > len(possGood):
            possGood = copy.deepcopy(possGoodH)
    return possGood
def Random(value0, value1, options={}): # 202109131700 (last read 2022-07-11)
    ''' comments:
    .- input value0 and value1 are floats
    .- output randomValues is a float or a float-ndarray
    '''
    keys, defaultValues = ['shape'], None
    options = CompleteADictionary(options, keys, defaultValues)
    if options['shape'] is None:
        randomValues = value0 + (value1 - value0) * np.random.random()
    else:
        randomValues = value0 + (value1 - value0) * np.random.random(options['shape'])
    return randomValues
def ReadAFirstSeed(dataBasic, dataForCal, freeVariablesKeys, givenVariablesDict, mainSetSeeds): # 202211082016 (read 2022-11-09)
    ''' comments:
    .- input dataBasic and dataForCal are dictionaries
    .- input freeVariablesKeys is a string-list (keys of the selected variables that are to optimize)
    .- input givenVariablesDict is a dictionary (keys and values of the selected variables that are given)
    .- input mainSetSeeds is a list of dictionaries
    .- output mainSetSeed is a dictionary or None
    .- output errorTSeed is a float
    '''
    mainSetSeedO, errorTSeedO = None, 1.e+11
    for mainSetSeed in mainSetSeeds:
        freeVariables = AllVariables2SubsetVariables(dataBasic, mainSetSeed['allVariables'], freeVariablesKeys)
        optionsTMP = {key:mainSetSeed[key] for key in ['nc', 'nr']}
        allVariables = SubsetVariables2AllVariables(dataBasic, freeVariables, freeVariablesKeys, givenVariablesDict, options=optionsTMP)
        optionsTMP = {key:dataBasic[key] for key in ['orderOfHorizonPoly', 'radiusOfEarth']}
        mainSetSeed = AllVariables2MainSet(allVariables, mainSetSeed['nc'], mainSetSeed['nr'], options=optionsTMP)
        errorTSeed = ErrorT(dataForCal, mainSetSeed, options={})
        if errorTSeed < errorTSeedO:
            mainSetSeedO, errorTSeedO = [copy.deepcopy(item) for item in [mainSetSeed, errorTSeed]]
    mainSetSeed, errorTSeed = [copy.deepcopy(item) for item in [mainSetSeedO, errorTSeedO]]
    return mainSetSeed, errorTSeed
def ReadCalTxt(pathCalTxt): # 202110131422 (last read 2022-07-01)
    ''' comments:
    .- input pathCalTxt is a string
    .- output allVariables is a 14-float-ndarray
    .- output nc and nr are integers
    .- output errorT is a float
    '''
    rawData = np.asarray(ReadRectangleFromTxt(pathCalTxt, {'c1':1, 'valueType':'float'}))
    allVariables, nc, nr, errorT = rawData[0:14], int(np.round(rawData[14])), int(np.round(rawData[15])), rawData[16]
    return allVariables, nc, nr, errorT
def ReadCdgTxt(pathCdgTxt, options={}): # 202110051016 (last read 2022-07-12)
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
    if options['readOnlyGood']: # disregards negative pixels (not a full control)
        possGood = np.where((cs >= 0.) & (rs >= 0.))[0]
        cs, rs, xs, ys, zs = [item[possGood] for item in [cs, rs, xs, ys, zs]]
    if options['readCodes']:
        codes = ReadRectangleFromTxt(pathCdgTxt, {'c0':5, 'c1':6, 'valueType':'str'}) # can be []
        if len(codes) > 0 and options['readOnlyGood']:
            codes = [codes[pos] for pos in possGood]
    else:
        codes = None
    return cs, rs, xs, ys, zs, codes
def ReadCdhTxt(pathCdhTxt, options={}): # 202110051054 (last read 2022-07-06)
    ''' comments:
    .- input pathCdhTxt is a string
    .- output chs and rhs are float-ndarrays (that can be empty)
    '''
    keys, defaultValues = ['readOnlyGood'], [True]
    options = CompleteADictionary(options, keys, defaultValues)
    rawData = np.asarray(ReadRectangleFromTxt(pathCdhTxt, {'c1':2, 'valueType':'float'}))
    if len(rawData) == 0:
        chs, rhs = [np.asarray([]) for item in range(2)]
    else:
        chs, rhs = [rawData[:, item] for item in range(2)]
        if options['readOnlyGood']: # disregards negative pixels
            possGood = np.where((chs >= 0.) & (rhs >= 0.))[0]
            chs, rhs = [item[possGood] for item in [chs, rhs]]
    return chs, rhs
def ReadPdfTxt(pathFile): # 202211090936 (last read 2022-11-09)
    ''' comments:
    .- input pathFile is a string
    .- output xUL, yUL, angle, xyLengthInC, xyLengthInR, ppm and timedeltaTolInHours are floats
    '''
    rawData = ReadRectangleFromTxt(pathFile, {'c1':1, 'valueType':'float'})
    if len(rawData) == 6:
        (xUL, yUL, angleInDegrees, xyLengthInC, xyLengthInR, ppm), timedeltaTolInHours = rawData[0:6], 0.
    elif len(rawData) == 7:
        xUL, yUL, angleInDegrees, xyLengthInC, xyLengthInR, ppm, timedeltaTolInHours = rawData[0:7]
    else: assert False
    angle = angleInDegrees * np.pi / 180. # IMP*
    return xUL, yUL, angle, xyLengthInC, xyLengthInR, ppm, timedeltaTolInHours
def ReadRectangleFromTxt(pathFile, options={}): # 202109141200 (last read 2022-11-10) avoid its use -> np.loadtxt
    assert os.path.isfile(pathFile)
    keys, defaultValues = ['c0', 'c1', 'r0', 'r1', 'valueType', 'nullLine'], [0, 0, 0, 0, 'str', None]
    options = CompleteADictionary(options, keys, defaultValues)
    openedFile = open(pathFile, 'r')
    listOfLines = openedFile.readlines()
    openedFile.close()
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
def ScaledFreeUVariables2FTM(scaledFreeUVariables, theArgs): # 202211091424 (last read 2022-11-09)
    ''' comments:
    .- input scaledFreeUVariables is a float-ndarray ("U" stands for Unique for all images)
    .- input theArgs is a dictionary
    .- output errorT is a float (errorC is not included)
    '''
    ncs, nrs, css, rss, xss, yss, zss, aG, chss, rhss, aH = [theArgs[item] for item in ['ncs', 'nrs', 'css', 'rss', 'xss', 'yss', 'zss', 'aG', 'chss', 'rhss', 'aH']]
    freeUVariables = VariablesScaling(theArgs['dataBasic'], scaledFreeUVariables, theArgs['freeUVariablesKeys'], 'unscale')
    freeUVariablesDict = Array2Dictionary(theArgs['freeUVariablesKeys'], freeUVariables)
    errorT, optionsMainSet = 0., {key:theArgs[key] for key in ['orderOfHorizonPoly', 'radiusOfEarth']}
    for pos in range(len(ncs)): # run through images
        restVariables, restVariablesKeys = np.concatenate((theArgs['freeDVariabless'][pos], theArgs['givenVariables'])), theArgs['freeDVariablesKeys'] + theArgs['givenVariablesKeys']
        allVariables = SubsetVariables2AllVariables(theArgs['dataBasic'], restVariables, restVariablesKeys, freeUVariablesDict, options={'nc':ncs[pos], 'nr':nrs[pos]})
        mainSet = AllVariables2MainSet(allVariables, ncs[pos], nrs[pos], options=optionsMainSet)
        dataForCal = {'cs':css[pos], 'rs':rss[pos], 'xs':xss[pos], 'ys':yss[pos], 'zs':zss[pos], 'aG':aG}
        if all([item is not None and len(item) > 0 for item in [chss[pos], rhss[pos]]]) and aH > 1.e-8:
            dataForCal['chs'], dataForCal['rhs'], dataForCal['aH'] = chss[pos], rhss[pos], aH
        errorT = errorT + ErrorT(dataForCal, mainSet, options={'verbose':False})
    return errorT
def ScaledSubsetVariables2ErrorT(dataBasic, dataForCal, scaledSubsetVariables, subsetVariablesKeys, subCsetVariablesDict, options={}): # 202109241454 (last read 2022-07-06)
    ''' comments:
    .- input dataBasic is a dictionary
    .- input dataForCal is a dictionary (including at least 'nc' and 'nr')
    .- input scaledSubsetVariables is a float-ndarray
    .- input subsetVariablesKeys is a string-list of the same length of scaledSubsetVariables
    .- input subCsetVariablesDict is a dictionary
    .- output errorT is a float
    '''
    keys, defaultValues = ['orderOfHorizonPoly', 'radiusOfEarth'], [5, 6.371e+6]
    options = CompleteADictionary(options, keys, defaultValues)
    nc, nr, optionsTMP = dataForCal['nc'], dataForCal['nr'], {key:options[key] for key in ['orderOfHorizonPoly', 'radiusOfEarth']}
    mainSet = ScaledSubsetVariables2MainSet(dataBasic, scaledSubsetVariables, subsetVariablesKeys, subCsetVariablesDict, nc, nr, options=optionsTMP)
    errorT = ErrorT(dataForCal, mainSet, options={})
    return errorT
def ScaledSubsetVariables2FTM(scaledSubsetVariables, theArgs): # 202109241454 (last read 2022-11-08)
    ''' comments:
    .- input scaledSubsetVariables is a float-ndarray (<-> 'scaledFreeVariables')
    .- input theArgs is a dictionary (including at least 'dataBasic', 'dataForCal', 'subsetVariablesKeys' and 'subCsetVariablesDict')
        .- theArgs['subCsetVariablesDict'] <-> 'givenVariablesDict'
    .- output errorT is a float
    '''
    dataBasic, dataForCal, subsetVariablesKeys, subCsetVariablesDict = [theArgs[key] for key in ['dataBasic', 'dataForCal', 'subsetVariablesKeys', 'subCsetVariablesDict']]
    optionsTMP = {key:dataBasic[key] for key in ['orderOfHorizonPoly', 'radiusOfEarth']}
    errorT = ScaledSubsetVariables2ErrorT(dataBasic, dataForCal, scaledSubsetVariables, subsetVariablesKeys, subCsetVariablesDict, options=optionsTMP)
    return errorT
def ScaledSubsetVariables2MainSet(dataBasic, scaledSubsetVariables, subsetVariablesKeys, subCsetVariablesDict, nc, nr, options={}): # 202207061424 (last read 2022-07-06)
    ''' comments:
    .- input dataBasic is a dictionary
    .- input scaledSubsetVariables is a float-ndarray
    .- input subsetVariablesKeys is a string-list of the same length of scaledSubsetVariables
    .- input subCsetVariablesDict is a dictionary
    .- input nc and nr are integers or floats
    .- output mainSet is a dictionary
    '''
    keys, defaultValues = ['orderOfHorizonPoly', 'radiusOfEarth'], [5, 6.371e+6]
    options = CompleteADictionary(options, keys, defaultValues)
    subsetVariables = VariablesScaling(dataBasic, scaledSubsetVariables, subsetVariablesKeys, 'unscale')
    allVariables = SubsetVariables2AllVariables(dataBasic, subsetVariables, subsetVariablesKeys, subCsetVariablesDict, options={'nc':nc, 'nr':nr})
    mainSet = AllVariables2MainSet(allVariables, nc, nr, options={key:options[key] for key in ['orderOfHorizonPoly', 'radiusOfEarth']})
    return mainSet
def SelectPixelsInGrid(nOfBands, nc, nr, cs, rs, es, options={}): # 202211101221 (last read 2022-11-10)
    ''' comments:
    .- input nOfBands is an integer (or None, for it is actually optional)
    .- input nc and nr are integers or floats
    .- input cs and rs are integers- or floats-ndarrays of the same length
    .- input es is a float-ndarray of the same length as cs and rs
    .- output possSelected, bandCsSelected and bandRsSelected are integer-list or Nones (if it does not succeed)
    '''
    keys, defaultValues = ['nOfCBands', 'nOfRBands'], None
    options = CompleteADictionary(options, keys, defaultValues)
    if options['nOfCBands'] is None or options['nOfRBands'] is None:
        nOfCBands, nOfRBands = nOfBands, nOfBands
    else:
        nOfCBands, nOfRBands = options['nOfCBands'], options['nOfRBands']
    if len(cs) == 0:
        return None, None, None
    bandCs = (cs * nOfCBands / nc).astype(int) # cs=0 -> bandCs=0; cs=nc-1 -> bandCs=int((nc-1)*nOfCBands/nc) = nOfCBands-1 (and cs=nc gives nOfCBands)
    bandRs = (rs * nOfRBands / nr).astype(int) # rs=0 -> bandRs=0; rs=nr-1 -> bandRs=int((nr-1)*nOfRBands/nr) = nOfRBands-1 (and rs=nr gives nOfRBands)
    bandGs = bandCs * 1 * (nOfRBands + 1) + bandRs # global counter
    bandGsU = np.asarray(list(set(list(bandGs))))
    possSelected, bandCsSelected, bandRsSelected = [[] for item in range(3)]
    for pos, bandGU in enumerate(bandGsU):
        possOfBandGU = np.where(bandGs == bandGU)[0] # list of global positions
        if len(possOfBandGU) == 1:
            posOfBandGU = possOfBandGU[0] # global position
        else:
            posOfBandGU = possOfBandGU[np.argmin(es[possOfBandGU])] # global position
        possSelected.append(posOfBandGU)
        bandCsSelected.append(bandCs[posOfBandGU])
        bandRsSelected.append(bandRs[posOfBandGU])
    return possSelected, bandCsSelected, bandRsSelected
def SelectedVariables2AllVariables(selectedVariables, selectedVariablesKeys, options={}): # 202201101351 (last read 2022-07-01)
    ''' comments:
    .- input selectedVariables a float-ndarray
    .- input selectedVariablesKeys is a string-list
    .- output allVariables is a float-ndarray
        .- if not in selectedVariables, then k1a, k2a, p1a and p2a are set to 0
        .- if not in selectedVariables, then sra is set to sca
        .- if not in selectedVariables, then oc and or are respectively set to kc and kr
    '''
    keys, defaultValues = ['nc', 'nr'], None
    options = CompleteADictionary(options, keys, defaultValues)
    allVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or'] # IMP* order matters
    allVariablesDict = Array2Dictionary(selectedVariablesKeys, selectedVariables) # initialize as selectedVariablesDict
    for key in [item for item in allVariablesKeys if item not in selectedVariablesKeys]:
        if key in ['k1a', 'k2a', 'p1a', 'p2a']:
            allVariablesDict[key] = 0.
        elif key == 'sra':
            allVariablesDict[key] = allVariablesDict['sca']
        elif key == 'oc':
            allVariablesDict[key] = N2K(options['nc'])
        elif key == 'or':
            allVariablesDict[key] = N2K(options['nr'])
        else:
            assert False
    allVariables = Dictionary2Array(allVariablesKeys, allVariablesDict)
    return allVariables
def SubsetVariables2AllVariables(dataBasic, subsetVariables, subsetVariablesKeys, subCsetVariablesDict, options={}): # 202109251454 (last read 2022-11-09)
    ''' comments:
    .- input dataBasic is a dictionary (including at least 'selectedVariablesKeys')
    .- input subsetVariables is a float-ndarray
    .- input subsetVariablesKeys is a string-list
    .- input subCsetVariablesDict is a dictionary
    .- output allVariables is a float-ndarray
    '''
    keys, defaultValues = ['nc', 'nr'], None
    options = CompleteADictionary(options, keys, defaultValues)
    subsetVariablesDict = Array2Dictionary(subsetVariablesKeys, subsetVariables)
    selectedVariablesDict = {**subsetVariablesDict, **subCsetVariablesDict} # IMP*
    selectedVariables = Dictionary2Array(dataBasic['selectedVariablesKeys'], selectedVariablesDict)
    optionsTMP = {key:options[key] for key in ['nc', 'nr']}
    allVariables = SelectedVariables2AllVariables(selectedVariables, dataBasic['selectedVariablesKeys'], options=optionsTMP)
    return allVariables
def UDaVDa2UUaVUa(mainSet, uDas, vDas): # uD* and vD* -> uU* and vU* 202207061139 (last read 2022-07-06, checked graphically with auxiliar code) potentially expensive
    ''' comments:
    .- input mainSet is a dictionary (including at least 'k1a', 'k2a', 'p1a', 'p2a')
    .- input uDas and vDas are float-ndarrays of the same length
    .- output uUas and vUas are float-ndarrays of the same length of uDas and vDas or Nones (if it does not succeed)
    .- the funcion is implicit unless k2a = p1a = p2a = 0
    '''
    def DeltaAndError220706(mainSet, uDas, vDas, uUas, vUas): # 202109131500
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
            xis = mainSet['k1a'] * (uDas ** 2 + vDas ** 2)
            xs = Xi2XForParabolicDistortion(xis) # = mainSet['k1a'] * (uUas ** 2 + vUas ** 2)
            uUas, vUas = [item / (1 + xs) for item in [uDas, vDas]]
        converged = True
    else: # implicit (Newton using DeltaAndError220706)
        uUas, vUas, error, converged, counter, speed = 1. * uDas, 1. * vDas, 1.e+11, False, 0, 1. # initialize undistorted with distorted
        while not converged and counter <= 20:
            duUas, dvUas, errorN = DeltaAndError220706(mainSet, uDas, vDas, uUas, vUas)
            if errorN > 2. * error:
                break
            uUas, vUas, error = uUas + speed * duUas, vUas + speed * dvUas, 1. * errorN
            converged, counter = error <= 1.e-11, counter + 1
    if not converged:
        uUas, vUas = None, None
    else:
        if len(possZero) > 0:
            uDas[possZero], vDas[possZero] = 0., 0.
            uUas[possZero], vUas[possZero] = 0., 0.
        uDasR, vDasR = UUaVUa2UDaVDa(mainSet, uUas, vUas)
        assert max([np.max(np.abs(uDasR - uDas)), np.max(np.abs(vDasR - vDas))]) < 5. * 1.e-11
    return uUas, vUas
def UUaVUa2UDaVDa(mainSet, uUas, vUas): # uU* and vU* -> uD* and vD* 202109131500 (last read 2022-07-06)
    ''' comments:
    .- input mainSet is a dictionary (including at least 'k1a', 'k2a', 'p1a' and 'p2a')
    .- input uUas and vUas are floats or float-ndarrays of the same length
    .- output uDas and vDas are floats or float-ndarrays of the same length of uUas and vUas
    '''
    aux1s = uUas ** 2 + vUas ** 2
    aux2s = 1. + mainSet['k1a'] * aux1s + mainSet['k2a'] * aux1s ** 2
    aux3s = 2. * uUas * vUas
    aux4s = aux1s + 2. * uUas ** 2
    aux5s = aux1s + 2. * vUas ** 2
    uDas = uUas * aux2s + mainSet['p2a'] * aux4s + mainSet['p1a'] * aux3s
    vDas = vUas * aux2s + mainSet['p1a'] * aux5s + mainSet['p2a'] * aux3s
    return uDas, vDas
def UUaVUa2XYZ(mainSet, planes, uUas, vUas, options={}): # 202109141800 (last read 2022-07-12) potentially expensive
    ''' comments:
    .- input mainSet is a dictionary (including at least 'eu', 'ev', 'ef' and 'pc')
    .- input planes is a dictionary (including at least 'pxs', 'pys', 'pzs' and 'pts')
        .- input planes['pxs'/'pys'/'pzs'/'pts'] is a float or a float-ndarray of the same length of uUas and vUas
    .- input uUas and vUas are float-ndarrays of the same length
    .- output xs, ys, zs are float-ndarrays of the same length of uUas and vUas
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
    dens = A11s * (A22s * A33s - A23s * A32s) + A12s * (A23s * A31s - A21s * A33s) + A13s * (A21s * A32s - A22s * A31s)
    dens = ClipWithSign(dens, 1.e-11, 1.e+11) # it was 1.e-8, 1.e+8
    xs = (bb1s * (A22s * A33s - A23s * A32s) + A12s * (A23s * bb3s - bb2s * A33s) + A13s * (bb2s * A32s - A22s * bb3s)) / dens
    ys = (A11s * (bb2s * A33s - A23s * bb3s) + bb1s * (A23s * A31s - A21s * A33s) + A13s * (A21s * bb3s - bb2s * A31s)) / dens
    zs = (A11s * (A22s * bb3s - bb2s * A32s) + A12s * (bb2s * A31s - A21s * bb3s) + bb1s * (A21s * A32s - A22s * A31s)) / dens
    poss = np.where(np.max(np.asarray([np.abs(xs), np.abs(ys), np.abs(zs)]), axis=0) < 1.e+8)[0]
    if isinstance(planes['pxs'], np.ndarray):
        auxs = planes['pxs'][poss] * xs[poss] + planes['pys'][poss] * ys[poss] + planes['pzs'][poss] * zs[poss] + planes['pts'][poss]
    else:
        auxs = planes['pxs'] * xs[poss] + planes['pys'] * ys[poss] + planes['pzs'] * zs[poss] + planes['pts']
    assert np.allclose(auxs, np.zeros(len(poss)))
    poss = np.where(np.max(np.asarray([np.abs(xs), np.abs(ys), np.abs(zs)]), axis=0) < 1.e+8)[0]
    uUasR, vUasR = XYZ2UUaVUa(mainSet, xs[poss], ys[poss], zs[poss], options={})[0:2]
    assert (np.allclose(uUasR, uUas[poss]) and np.allclose(vUasR, vUas[poss]))
    if options['returnPositionsRightSideOfCamera']:
        possRightSideOfCamera = XYZ2PositionsRightSideOfCamera(mainSet, xs, ys, zs)
    else:
        possRightSideOfCamera = None
    return xs, ys, zs, possRightSideOfCamera
def UaVa2CR(mainSet, uas, vas): # 202109101200 (last read 2022-07-06)
    ''' comments:
    .- input mainSet is a dictionary (including at least 'sca', 'sra', 'oc' and 'or')
        .- mainSet['sca'] and mainSet['sra'] are non-zero, but allowed to be negative
    .- input uas and vas are floats or float-ndarrays of the same length
    .- output cs and rs are floats or float-ndarrays of the same length of uas and vas
    '''
    cs = uas / mainSet['sca'] + mainSet['oc']
    rs = vas / mainSet['sra'] + mainSet['or']
    return cs, rs
def UnitVectors2R(eu, ev, ef): # 202109231416 (last read 2022-06-29)
    ''' comments:
    .- input eu, ev and ef are 3-float-ndarrays
    .- output R is a 3x3-float-ndarray
        .- the rows of R are eu, ev and ef
    '''
    R = np.asarray([eu, ev, ef])
    return R
def UpdateFreeUVariables(dataBasic, dataForCals, freeDVariabless, freeDVariablesKeys, freeUVariablesSeed, freeUVariablesKeys, givenVariables, givenVariablesKeys, options={}): # 202211101206 (last read 2022-11-10)
    ''' comments:
    .- input dataBasic is a dictionary
    .- input dataForCals is a dictionary
    .- input freeDVariabless are float-ndarrays-lists
    .- input freeDVariablesKeys is a string-list
    .- input freeUVariablesSeed is a float-ndarray
    .- input freeUVariablesKeys is a string-list
    .- input givenVariables is a float-ndarray
    .- input givenVariablesKeys is a string-list
    .- output freeUVariables is a float-ndarray
    .- output errorT is a float
    '''
    keys, defaultValues = ['nOfSeedsForC', 'orderOfHorizonPoly', 'radiusOfEarth'], [20, 5, 6.371e+6]
    options = CompleteADictionary(options, keys, defaultValues)
    auxTMP = locals() # IMP*
    theArgs = {key:auxTMP[key] for key in ['dataBasic', 'freeDVariabless', 'freeDVariablesKeys', 'freeUVariablesKeys', 'givenVariables', 'givenVariablesKeys']}
    for key in ['ncs', 'nrs', 'css', 'rss', 'xss', 'yss', 'zss', 'aG', 'chss', 'rhss', 'aH']:
        theArgs[key] = dataForCals[key]
    for key in ['orderOfHorizonPoly', 'radiusOfEarth']:
        theArgs[key] = options[key]
    scaledFreeUVariablesSeed = VariablesScaling(dataBasic, freeUVariablesSeed, freeUVariablesKeys, 'scale')
    errorTSeed = ScaledFreeUVariables2FTM(scaledFreeUVariablesSeed, theArgs)
    scaledFreeUVariablesO, errorTO = [copy.deepcopy(item) for item in [scaledFreeUVariablesSeed, errorTSeed]]
    for iOfMonteCarlo in range(options['nOfSeedsForC']):
        scaledFreeUVariablesP = scaledFreeUVariablesO * (0.8 + 0.4 * np.random.random(len(scaledFreeUVariablesO))) # IMP* WATCH OUT
        scaledFreeUVariablesP = optimize.minimize(ScaledFreeUVariables2FTM, scaledFreeUVariablesP, args = (theArgs)).x
        errorTP = ScaledFreeUVariables2FTM(scaledFreeUVariablesP, theArgs)
        if errorTP < errorTO * 0.999:
            scaledFreeUVariablesO, errorTO = [copy.deepcopy(item) for item in [scaledFreeUVariablesP, errorTP]]
    scaledFreeUVariables, errorT = [copy.deepcopy(item) for item in [scaledFreeUVariablesO, errorTO]]
    freeUVariables = VariablesScaling(dataBasic, scaledFreeUVariables, freeUVariablesKeys, 'unscale')
    return freeUVariables, errorT
def VariablesScaling(dataBasic, variables, variablesKeys, direction): # 202109241439 (last read 2022-06-29)
    ''' comments:
    .- input dataBasic is a dictionary (including at least 'scalesDict')
    .- input variables is a float-ndarray
    .- input variablesKeys is a string-list of the same length of input variables
    .- input direction is a string ('scale' or 'unscale')
    .- output variables is a float-ndarray of the same length of input variables
    '''
    scales = Dictionary2Array(variablesKeys, dataBasic['scalesDict'])
    if direction == 'scale':
        variables = variables / scales
    elif direction == 'unscale':
        variables = variables * scales
    else:
        assert False
    return variables
def WriteCalTxt(pathCalTxt, allVariables, nc, nr, errorT): # 202110131423 (last read 2022-07-01)
    ''' comments:
    .- input pathCalTxt is a string
    .- input allVariables is a 14-float-ndarray
    .- input nc and nr are integers
    .- input errorT is a float
    '''
    allVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or'] # IMP* order matters
    MakeFolder(os.path.split(pathCalTxt)[0])
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
def WriteDataPdfTxt(dataPdfTxt): # 202211091359 (last read 2022-11-09) avoid its use -> WritePdfTxt
    ''' comments:
    .- input data is a dictionary (including at least 'pathFile', 'xUL', 'yUL', 'angle', 'xyLengthInC', 'xyLengthInR', 'ppm' and 'timedeltaTol')
    '''
    if dataPdfTxt['pathFile'] is not None:
        pathPdfTxt, timedeltaTolInHours = dataPdfTxt['pathFile'], dataPdfTxt['timedeltaTol'].total_seconds() / 3600.
        xUL, yUL, angle, xyLengthInC, xyLengthInR, ppm = [dataPdfTxt[key] for key in ['xUL', 'yUL', 'angle', 'xyLengthInC', 'xyLengthInR', 'ppm']]
        WritePdfTxt(pathPdfTxt, xUL, yUL, angle, xyLengthInC, xyLengthInR, ppm, timedeltaTolInHours)
    return None
def WritePdfTxt(pathFile, xUL, yUL, angle, xyLengthInC, xyLengthInR, ppm, timedeltaTolInHours): # 202211090938 (last read 2022-11-09)
    ''' comments:
    .- input pathFile is a string
    .- input xUL, yUL, angle, xyLengthInC, xyLengthInR, ppm and timedeltaTolInHours are floats
    '''
    fileout = open(pathFile, 'w')
    fileout.write('{:20.8f} real world x for upper left corner, in meters \n'.format(xUL))
    fileout.write('{:20.8f} real world y for upper left corner, in meters \n'.format(yUL))
    fileout.write('{:20.8f} angle, in degrees: 0 = E, 90 = N \n'.format(angle * 180. / np.pi))
    fileout.write('{:20.8f} real world length column-wise, in meters \n'.format(xyLengthInC))
    fileout.write('{:20.8f} real world length row-wise, in meters \n'.format(xyLengthInR))
    fileout.write('{:20.8f} pixels per meter \n'.format(ppm))
    fileout.write('{:20.8f} time delta tolerance, in hours \n'.format(timedeltaTolInHours))
    fileout.close()
    return None
def XYZ2A0(xs, ys, zs): # 202201250808 (last read 2022-07-06)
    '''
    .- input xs, ys and zs are float-ndarrays of the same length
    .- output A0 is a (2*len(xs)x8)-float-ndarray
    '''
    poss0, poss1 = Poss0AndPoss1(len(xs))
    A0 = np.zeros((2 * len(xs), 8)) # IMP* initialize with zeroes
    A0[poss0, 0], A0[poss0, 1], A0[poss0, 2], A0[poss0, 3] = xs, ys, zs, np.ones(xs.shape)
    A0[poss1, 4], A0[poss1, 5], A0[poss1, 6], A0[poss1, 7] = xs, ys, zs, np.ones(xs.shape)
    return A0
def XYZ2CDRD(mainSet, xs, ys, zs, options={}): # 202109131600 (last read 2022-07-12)
    ''' comments:
    .- input mainSet is a dictionary (including at least 'nc' and 'nr')
    .- input xs, ys and zs are float-ndarrays of the same length
    .- output cDs and rDs are float-ndarrays of the same length of xs, ys and zs
    .- output possGood is a list of integers or None (if not options['returnGoodPositions'])
    '''
    keys, defaultValues = ['imgMargins', 'returnGoodPositions'], [{'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}, False]
    options = CompleteADictionary(options, keys, defaultValues)
    uUas, vUas, possGood = XYZ2UUaVUa(mainSet, xs, ys, zs, options={'returnPositionsRightSideOfCamera':options['returnGoodPositions']})
    uDas, vDas = UUaVUa2UDaVDa(mainSet, uUas, vUas)
    cDs, rDs = UaVa2CR(mainSet, uDas, vDas)
    if options['returnGoodPositions']: # so far possGood are at the right side of the camera
        if len(possGood) > 0:
            nc, nr, cDsGood, rDsGood = mainSet['nc'], mainSet['nr'], cDs[possGood], rDs[possGood]
            possGoodInGood = CR2PositionsWithinImage(nc, nr, cDsGood, rDsGood, options={'imgMargins':options['imgMargins']})
            possGood = [possGood[item] for item in possGoodInGood]
        if len(possGood) > 0:
            xsGood, ysGood, zsGood, cDsGood, rDsGood = [item[possGood] for item in [xs, ys, zs, cDs, rDs]]
            xsGoodR, ysGoodR = CDRDZ2XY(mainSet, cDsGood, rDsGood, zsGood, options={})[0:2] # all, not only good positions; potentially expensive
            possGoodInGood = np.where(np.sqrt((xsGood - xsGoodR) ** 2 + (ysGood - ysGoodR) ** 2) < 1.e-5)[0] # 1.e-5 could be changed
            possGood = [possGood[item] for item in possGoodInGood]
    else: # possGood is None from XYZ2UUaVUa above
        assert possGood is None
    return cDs, rDs, possGood
def XYZ2PositionsRightSideOfCamera(mainSet, xs, ys, zs): # 202109231412 (last read 2022-07-06)
    '''
    .- input mainSet is a dictionary (including 'xc', 'yc', 'zc' and 'ef')
    .- input xs, ys and zs are float-ndarrays of the same length
    .- output possRightSideOfCamera is a integer-list
    '''
    xas, yas, zas = xs - mainSet['xc'], ys - mainSet['yc'], zs - mainSet['zc']
    possRightSideOfCamera = np.where(xas * mainSet['ef'][0] + yas * mainSet['ef'][1] + zas * mainSet['ef'][2] > 0)[0]
    return possRightSideOfCamera
def XYZ2UUaVUa(mainSet, xs, ys, zs, options={}): # 202109231411 (last read 2022-07-06)
    ''' comments:
    .- input mainSet is a dictionary (including at least 'xc', 'yc', 'zc', 'eu', 'ev' and 'ef')
    .- input xs, ys and zs are float-ndarrays of the same length
    .- output uUas and vUas are float-ndarrays of the same length of xs, ys and zs
    .- output possRightSideOfCamera is a list of integers or None (if not options['returnPositionsRightSideOfCamera'])
    '''
    keys, defaultValues = ['returnPositionsRightSideOfCamera'], [False]
    options = CompleteADictionary(options, keys, defaultValues)
    xas, yas, zas = xs - mainSet['xc'], ys - mainSet['yc'], zs - mainSet['zc']
    dns = xas * mainSet['ef'][0] + yas * mainSet['ef'][1] + zas * mainSet['ef'][2]
    dns = ClipWithSign(dns, 1.e-8, 1.e+8)
    uUas = (xas * mainSet['eu'][0] + yas * mainSet['eu'][1] + zas * mainSet['eu'][2]) / dns
    vUas = (xas * mainSet['ev'][0] + yas * mainSet['ev'][1] + zas * mainSet['ev'][2]) / dns
    if options['returnPositionsRightSideOfCamera']:
        possRightSideOfCamera = XYZ2PositionsRightSideOfCamera(mainSet, xs, ys, zs)
    else:
        possRightSideOfCamera = None
    return uUas, vUas, possRightSideOfCamera
def XYZPa112CURU(xs, ys, zs, Pa11): # 202207061412 (last read 2022-07-06)
    ''''
    .- input xs, ys and zs are floats or float-ndarrays of the same length
    .- input Pa11 is a 11-float-ndarray
    .- output cUs and rUs are floats or float-ndarrays of the same length of xs, ys and zs
    '''
    dens = Pa11[8] * xs + Pa11[9] * ys + Pa11[10] * zs + 1.
    cUs = (Pa11[0] * xs + Pa11[1] * ys + Pa11[2] * zs + Pa11[3]) / dens
    rUs = (Pa11[4] * xs + Pa11[5] * ys + Pa11[6] * zs + Pa11[7]) / dens
    return cUs, rUs
def Xi2XForParabolicDistortion(xis): # 202207060912 (last read 2022-07-06, checked graphically with auxiliar code)
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
    return xs
