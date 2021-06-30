'''
Created on 2021 by Gonzalo Simarro, Daniel Calvete and Paola Souto
'''
#
import copy
import cv2
import datetime
import numpy as np
import os
import random
from scipy import optimize
#
''' -------------------------------------------------------------------------------------- '''
''' --- basic functions ------------------------------------------------------------------ '''
''' -------------------------------------------------------------------------------------- '''
#
# dictionaries and arrays
def CompleteADictionary(theDictionary, keys, defaultValues): # 202106141800
    #
    ''' comments:
    .- input theDictionary is a dictionary
    .- input keys is a list of strings
    .- input defaultValues is a list or a single value
      .- if defaultValues is a list, it must have the same length as keys
    .- output theDictionary is a dictionary that includes keys and defaultValues for keys not in input theDictionary
    '''
    #
    # complete dictionary
    try:
        if set(keys) <= set(theDictionary.keys()): # all keys are already in theDictionary
            pass
        else:
            if isinstance(defaultValues, (list)): # defaultValues is a list
                if not len(keys) == len(defaultValues):
                    print('*** CompleteADictionary: check the lengths of keys and defaultValues'); assert False
                for posKey, key in enumerate(keys):
                    if key not in theDictionary.keys(): # IMPORTANT (only assign if there is no key)
                        theDictionary[key] = defaultValues[posKey]
            else: # defaultValues is a single value
                for key in keys:
                    if key not in theDictionary.keys(): # IMPORTANT (only assign if there is no key)
                        theDictionary[key] = defaultValues
    except:
        print('*** CompleteADictionary: error completing dictionary'); assert False
    #
    return theDictionary
def Array2Dictionary(keys, theArray): # 202106141800
    #
    ''' comments:
    .- input keys is a list of strings
    .- input theArray is a ndarray
      .- keys and theArray must have the same length
    .- output theDictionary is a dictionary
    '''
    #
    # obtain theDictionary
    try:
        # check
        if not (len(set(keys)) == len(keys) == len(theArray)):
            print('*** Array2Dictionary: check lengths of keys and theArray'); assert False
        #
        # obtain theDictionary
        theDictionary = {}
        for posKey, key in enumerate(keys):
            theDictionary[key] = theArray[posKey]
    except:
        print('*** Array2Dictionary: error obtaining theDictionary'); assert False
    #
    return theDictionary
def Dictionary2Array(keys, theDictionary): # 202106141800
    #
    ''' comments:
    .- input keys is a list of strings
    .- input theDictionary is a dictionary
    .- output theArray is a ndarray
      .- theArray contains the values of the keys in theDictionary
    '''
    #
    # obtain theArray
    try:
        theArray = np.zeros(len(keys))
        for posKey, key in enumerate(keys):
            theArray[posKey] = theDictionary[key]
    except:
        if not (set(keys) <= set(theDictionary.keys())):
            print('*** Dictionary2Array: check keys and theDictionary.keys()'); assert False
        else:
            print('*** Dictionary2Array: error obtaining theArray'); assert False
    #
    return theArray
#
# mathematics
def DistanceFromAPointToAPoint(x0, y0, x1, y1): # 202106141800
    #
    ''' comments:
    .- input x0 and y0 are floats or float-ndarrays
    .- input x1 and y1 are floats or float-ndarrays
      .- if both ({x0, y0} and {x1, y1}) are ndarrays, they must have the same length
    .- output distance is float or float-ndarray
    '''
    #
    # obtain distance
    try:
        distance = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    except:
        print('*** DistanceFromAPointToAPoint: error obtaining distance'); assert False
    #
    return distance
def ClipWithSign(xs, x0, x1): # 202106141800
    #
    ''' comments:
    .- input xs is a float or a float-ndarray
    .- input x0 and x1 are floats so that x1 >= x0 >= 0
    .- output xs is in [-x1, -x0] U [x0, x1] retaining the signs of input xs
    '''
    #
    # check x0 and x1
    try:
        if not (x1 >= x0 and x0 >= 0.):
            print('*** ClipWithSign: check x0 and x1'); assert False
    except:
        print('*** ClipWithSign: error checking x0 and x1'); assert False
    #
    # clip xs
    try:
        signs = np.sign(xs)
        if isinstance(signs, (np.ndarray)): # float ndarray
            signs[signs == 0] = 1
        else: # float
            if signs == 0:
                signs = 1
        xs = signs * np.clip(np.abs(xs), x0, x1)
    except:
        print('*** ClipWithSign: error clipping xs')
    #
    return xs
#
# images
def CompleteImgMargins(imgMargins): # 202106141800
    #
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
    #
    # manage isComplete
    try:
        if imgMargins is not None and 'isComplete' in imgMargins.keys() and imgMargins['isComplete']:
            return imgMargins
    except:
        print('*** CompleteImgMargins: error managing isComplete'); assert False
    #
    # initialize imgMargins (if necessary)
    try:
        if imgMargins is None:
            imgMargins = {'c':0, 'r':0}
    except:
        print('*** CompleteImgMargins: error initializing imgMargins'); assert False
    #
    # complete imgMargins
    try:
        # complete imgMargins
        for letter in ['c', 'r']:
            # check if imgMargins[letter] contains an integer, and otherwise imgMargins[letter+number] must contain an integer
            try:
                assert int(imgMargins[letter]) == imgMargins[letter]
            except: # imgMargins[letter] is not an integer (it is None or whatever)
                for number in ['0', '1']:
                    try:
                        assert int(imgMargins[letter+number]) == imgMargins[letter+number]
                    except:
                        print('*** CompleteImgMargins: check {:} and {:}'.format(letter, letter+number)); assert False
                continue # go to the next letter since letter+number already ok for this letter
            # load imMargins if imgMargins[letter] contains an integer
            for number in ['0', '1']:
                try: 
                    assert int(imgMargins[letter+number]) == imgMargins[letter+number]
                except:
                    imgMargins[letter+number] = imgMargins[letter]
        #
        # load isComplete
        imgMargins['isComplete'] = True
    except:
        print('*** CompleteImgMargins: error completing imgMargins'); assert False
    #
    return imgMargins
def AreImgMarginsOK(nc, nr, imgMargins): # 202106141800
    #
    ''' comments:
    .- input nc and nr are integers
    .- input imgMargins is a dictionary
    '''
    #
    # obtain areMarginsOK
    try:
        imgMargins = CompleteImgMargins(imgMargins)
        condC = min([imgMargins['c0'], imgMargins['c1'], nc-1-(imgMargins['c0']+imgMargins['c1'])]) >= 0 # to leave a pixel at least
        condR = min([imgMargins['r0'], imgMargins['r1'], nr-1-(imgMargins['r0']+imgMargins['r1'])]) >= 0 # to leave a pixel at least
        areMarginsOK = condC and condR
    except:
        if not (set(['c0', 'c1', 'r0', 'r1']) <= set(imgMargins.keys())):
            print('*** AreImgMarginsOK: check the keys of imgMargins'); assert False
        else:
            print('*** AreImgMarginsOK: error obtaining areMarginsOK'); assert False
    #
    return areMarginsOK
def CR2PositionsWithinImage(nc, nr, cs, rs, options={}): # 202106141800
    #
    ''' comments:
    .- input nc and nr are integers or floats
    .- input cs and rs are integer- or float-ndarrays of the same length
    .- output possWithin is an integer-list
    '''
    #
    # complete options
    try:
        keys, defaultValues = ['imgMargins', 'rounding'], [{'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}, False]
        options = CompleteADictionary(options, keys, defaultValues)
        ''' comments:
        .- options['imgMargins'] is a dictionary or None (see CompleteImgMargins)
        .- options['rounding'] is a boolean (if True, it first runs CR2CRInteger)
        '''
    except:
        print('*** CR2PositionsWithinImage: error completing options'); assert False
    #
    # obtain possWithin
    try:
        # complete and check imgMargins
        imgMargins = CompleteImgMargins(options['imgMargins'])
        if not AreImgMarginsOK(nc, nr, imgMargins):
            print('*** CR2PositionsWithinImage: check imgMargins'); assert False
        #
        # obtain possWithin
        if options['rounding']:
            cs, rs = CR2CRInteger(cs, rs)
        cMin, cMax = imgMargins['c0'], nc-1-imgMargins['c1'] # recall that img[:, nc-1, :] is OK, but not img[:, nc, :]
        rMin, rMax = imgMargins['r0'], nr-1-imgMargins['r1'] # recall that img[nr-1, :, :] is OK, but not img[nr, :, :]
        possWithin = np.where((cs >= cMin) & (cs <= cMax) & (rs >= rMin) & (rs <= rMax))[0]
    except:
        if not (all([isinstance(item, (np.ndarray)) for item in [cs, rs]]) and len(cs) == len(rs)):
            print('*** CR2PositionsWithinImage: check formats and lengths of cs and rs'); assert False
        else:
            print('*** CR2PositionsWithinImage: error obtaining possWithin'); assert False
    #
    return possWithin
def CR2CRInteger(cs, rs): # 202106141800
    #
    ''' comments:
    .- input cs and rs are integer- or float-ndarrays
    .- output cs and rs are integer-ndarrays
    '''
    #
    # obtain cs and rs
    try:
        cs = np.round(cs).astype(int)
        rs = np.round(rs).astype(int)
    except:
        if not all([isinstance(item, (np.ndarray)) for item in [cs, rs]]):
            print('*** CR2CRInteger: check the formats of cs and rs'); assert False
        else:
            print('*** CR2CRInteger: error obtaining cs and rs'); assert False
    #
    return cs, rs
def ORBKeypoints(img, options={}): # 202106141800
    #
    ''' comments:
    .- input img can be an image or its path
    .- output nc and nr are integers
    .- output kps are ORB keypoints
    .- output des are ORB descriptions
    .- output ctrl is False if it does not succeed
    '''
    #
    # complete options
    try:
        keys, defaultValues = ['nOfFeatures'], [10000]
        options = CompleteADictionary(options, keys, defaultValues)
        ''' comments:
        .- options['nOfFeatures'] is for ORB
        '''
    except:
        print('*** ORBKeypoints: error completing options'); assert False
    #
    # load and check image
    try:
        if type(img) == str:
            img = cv2.imread(img)
        else:
            pass
        nr, nc = img.shape[0:2]
        img[nr-1, nc-1, 0]
    except:
        print('*** ORBKeypoints: error loading and checking image'); assert False
    #
    # obtain kps and des
    try:
        orb = cv2.ORB_create(nfeatures=options['nOfFeatures'], scoreType=cv2.ORB_FAST_SCORE)
        kps, des = orb.detectAndCompute(img, None)
        try:
            assert len(kps) == len(des) > 0
            ctrl = True
        except:
            ctrl = False
    except:
        print('*** ORBKeypoints: error obtaining kps and des'); assert False
    #
    return nc, nr, kps, des, ctrl
def ORBMatches(kps1, des1, kps2, des2, options={}): # 202106141800
    #
    ''' comments:
    .- input kps1 and des1 are ORB keys and descriptions for image1 (see ORBKeypoints)
    .- input kps2 and des2 are ORB keys and descriptions for image2 (see ORBKeypoints)
    .- output cs1, rs1, cs2, rs2 and ers are float-ndarrays
    .- compares both ways, so that commutativite property is ensured
    '''
    #
    # complete options
    try:
        keys, defaultValues = ['erMaximum', 'nOfStd'], [30., 0.5]
        options = CompleteADictionary(options, keys, defaultValues)
        ''' comments:
        .- output cs1, rs1, cs2, rs2 and ers are for positions where ers0 < erMaximum and ds < ds.median + nOfStd * ds.std
        '''
    except:
        print('*** ORBMatches: error completing options'); assert False
    #
    # obtain pairs
    try:
        # obtain pixels of keypoints
        cs1, rs1 = np.asarray([item.pt[0] for item in kps1]), np.asarray([item.pt[1] for item in kps1])
        cs2, rs2 = np.asarray([item.pt[0] for item in kps2]), np.asarray([item.pt[1] for item in kps2])
        #
        # obtain matches positions
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches12 = sorted(bf.match(des1, des2), key = lambda x:x.distance)
        matches21 = sorted(bf.match(des2, des1), key = lambda x:x.distance)
        #
        # obtain pixels and Hamming distances (ers) of matches
        poss1 = [match.queryIdx for match in matches12] + [match.trainIdx for match in matches21]
        poss2 = [match.trainIdx for match in matches12] + [match.queryIdx for match in matches21]
        cs1, rs1, cs2, rs2 = cs1[poss1], rs1[poss1], cs2[poss2], rs2[poss2]
        ers = np.asarray([match.distance for match in matches12] + [match.distance for match in matches21])
        assert len(cs1) == len(rs1) == len(cs2) == len(rs2) == len(ers)
        #
        # obtain pixels and Hamming distances (ers) using the condition
        distances = np.sqrt((cs1 - cs2) ** 2 + (rs1 - rs2) ** 2)
        possGood = np.where((ers < options['erMaximum']) & (distances < np.median(distances) + options['nOfStd'] * np.std(distances) + 1.e-3))[0]
        cs1, rs1, cs2, rs2, ers = [item[possGood] for item in [cs1, rs1, cs2, rs2, ers]]
    except:
        print('*** ORBMatches: error obtaining pairs'); assert False
    #
    return cs1, rs1, cs2, rs2, ers

def CR2BandsCR(nOfBands, nc, nr, cs, rs): # 202106141800
    #
    ''' comments:
    .- input nOfBands is an integer
    .- input nc and nr are integers
    .- input cs and rs are integers- or floats-ndarrays of the same length
    .- output bandCs and bandRs are lists of integers of the same length as cs and rs
    '''
    #
    # obtain bandCs and bandRs
    try:
        # obtain bandCs and bandRs
        bandWidthInC, bandWidthInR = nc / nOfBands, nr / nOfBands
        bandCs = (cs / bandWidthInC).astype(int) # cs=0 -> cs/bandWidthInC=0; cs=nc-1 -> cs/bandWidthInC=(nc-1)*nOfBands/nc < nOfBands
        bandRs = (rs / bandWidthInR).astype(int) # rs=0 -> rs/bandWidthInR=0; rs=nr-1 -> rs/bandWidthInR=(nr-1)*nOfBands/nr < nOfBands
        #
        # check and make lists
        if not (np.min(bandCs) >= 0 and np.max(bandCs) <= nOfBands - 1 and np.min(bandRs) >= 0 and np.max(bandRs) <= nOfBands - 1):
            print('*** CR2BandsCR: error obtaining bandCs and bandRs'); assert False
        bandCs, bandRs = [list(item) for item in [bandCs, bandRs]]
    except:
        if not all([isinstance(item, (np.ndarray)) for item in [cs, rs]]):
            print('*** CR2BandsCR: check the formats of cs and rs'); assert False
        else:
            print('*** CR2BandsCR: error obtaining bandCs and bandRs'); assert False
    #
    return bandCs, bandRs
def SelectPixelsInGrid(nOfBands, nc, nr, cs, rs, es): # 202106141800
    #
    ''' comments:
    .- input nOfBands is an integer
    .- input nc and nr are integers or floats
    .- input cs and rs are integers- or floats-ndarrays of the same length
    .- input es is a float-ndarrays of the same length as cs and rs
    .- output possSelected is a list of integers or None (if it does not succeed)
    .- output bandCsSelected and bandRsSelected are integer-ndarrays or Nones (if it does not succeed)
    '''
    #
    # manage len(cs) == 0
    try:
        nOfInitialPixels = len(cs)
        if nOfInitialPixels == 0:
            possSelected, bandCsSelected, bandRsSelected = None, None, None
            return possSelected, bandCsSelected, bandRsSelected
    except:
        print('*** SelectPixelsInGrid: error managing len(cs) == 0'); assert False
    #
    # obtain unique bands
    try:
        bandCs, bandRs = CR2BandsCR(nOfBands, nc, nr, cs, rs) # lists
        bandCs = np.asarray(bandCs).astype(int)
        bandRs = np.asarray(bandRs).astype(int)
        bandGs = bandCs * 5 * nOfBands + bandRs
        bandGsUnique = np.asarray(list(set(list(bandGs))))
        nOfFinalPixels = len(bandGsUnique)
        if not (nOfFinalPixels > 0):
            print('*** SelectPixelsInGrid: error obtaining unique bands (nOfFinalPixels)'); assert False
    except:
        print('*** SelectPixelsInGrid: error obtaining unique bands'); assert False
    #
    # obtain possSelected, bandCsSelected and bandRsSelected
    try:
        possSelected, bandCsSelected, bandRsSelected = [np.zeros(nOfFinalPixels).astype(int) for item in range(3)]
        for pos, bandGUnique in enumerate(bandGsUnique):
            possOfBandGUnique = np.where(bandGs == bandGUnique)[0] # global positions
            if not len(possOfBandGUnique) > 0:
                print('*** SelectPixelsInGrid: error obtaining possOfBandGUnique'); assert False
            if len(possOfBandGUnique) == 1:
                posOfBandGUnique = possOfBandGUnique[0]
            else:
                posOfBandGUnique = possOfBandGUnique[np.argmin(es[possOfBandGUnique])]
            possSelected[pos] = posOfBandGUnique
            bandCsSelected[pos] = bandCs[posOfBandGUnique]
            bandRsSelected[pos] = bandRs[posOfBandGUnique]
        possSelected = list(possSelected)
    except:
        print('*** SelectPixelsInGrid: error obtaining possSelected, bandCsSelected and bandRsSelected'); assert False
    #
    return possSelected, bandCsSelected, bandRsSelected
#
''' -------------------------------------------------------------------------------------- '''
''' --- selection of ulises functions ---------------------------------------------------- '''
''' -------------------------------------------------------------------------------------- '''
#
# basic functions
def LoadDataBasic0(options={}): # 202106141800
    #
    ''' comments:
    .- output data is a dictionary
    '''
    #
    # complete options
    try:
        keys, defaultValues = ['selectedUnifiedVariablesKeys'], [['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'sca']]
        options = CompleteADictionary(options, keys, defaultValues)
    except:
        print('*** LoadDataBasic0: error completing options'); assert False
    #
    # obtain data
    try:
        # initialize data
        data = {}
        #
        # load date of the beginning and the end of the world
        data['date0OfTheWorld'] = '19000101000000000'
        data['date1OfTheWorld'] = '40000101000000000'
        #
        # load lists of keys
        data['intrinsicVariablesKeys'] = ['k1', 'k2', 'p1', 'p2', 'sc', 'sr', 'oc', 'or'] # IMP* assumed anyway in the codes
        data['extrinsicVariablesKeys'] = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'fd'] # IMP* assumed anyway in the codes
        data['unifiedVariablesKeys'] = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or'] # IMP* assumed anyway in the codes
        data['selectedUnifiedVariablesKeys'] = options['selectedUnifiedVariablesKeys'] # IMP*
        #
        # check selectedUnifiedVariablesKeys
        if not (set(['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'sca']) <= set(data['selectedUnifiedVariablesKeys']) and set(data['selectedUnifiedVariablesKeys']) <= set(data['unifiedVariablesKeys'])):
            print('*** LoadDataBasic0: check selectedUnifiedVariablesKeys'); assert False
        #
        # load dictionaries referenceValuesDictionary, referenceRangesDictionary and scales: initialize
        data['referenceValuesDictionary'], data['referenceRangesDictionary'], data['scalesDictionary'] = {}, {}, {}
        # load dictionaries referenceValuesDictionary, referenceRangesDictionary and scales: keys xc, yc, zc, ph, sg, ta and fd
        data['referenceValuesDictionary']['xc'], data['referenceRangesDictionary']['xc'], data['scalesDictionary']['xc'] = None, 1.0e+1, 1.0e+1
        data['referenceValuesDictionary']['yc'], data['referenceRangesDictionary']['yc'], data['scalesDictionary']['yc'] = None, 1.0e+1, 1.0e+1
        data['referenceValuesDictionary']['zc'], data['referenceRangesDictionary']['zc'], data['scalesDictionary']['zc'] = None, 1.0e+1, 1.0e+1
        data['referenceValuesDictionary']['ph'], data['referenceRangesDictionary']['ph'], data['scalesDictionary']['ph'] = 0.*np.pi/2., np.pi/1., 1.0e+0
        data['referenceValuesDictionary']['sg'], data['referenceRangesDictionary']['sg'], data['scalesDictionary']['sg'] = 0.*np.pi/2., np.pi/4., 1.0e+0
        data['referenceValuesDictionary']['ta'], data['referenceRangesDictionary']['ta'], data['scalesDictionary']['ta'] = 1.*np.pi/2., np.pi/2., 1.0e+0 # IMP*
        data['referenceValuesDictionary']['fd'], data['referenceRangesDictionary']['fd'], data['scalesDictionary']['fd'] = 1.0e+0, 1.0e-1, 1.0e-2
        # load dictionaries referenceValuesDictionary, referenceRangesDictionary and scales: keys k1a, k2a, p1a, p2a, sca and sra
        data['referenceValuesDictionary']['k1a'], data['referenceRangesDictionary']['k1a'], data['scalesDictionary']['k1a'] = 0.0e+0, 1.0e+0, 1.e-1
        data['referenceValuesDictionary']['k2a'], data['referenceRangesDictionary']['k2a'], data['scalesDictionary']['k2a'] = 0.0e+0, 1.0e+0, 1.e-0
        data['referenceValuesDictionary']['p1a'], data['referenceRangesDictionary']['p1a'], data['scalesDictionary']['p1a'] = 0.0e+0, 1.0e-2, 1.e-2
        data['referenceValuesDictionary']['p2a'], data['referenceRangesDictionary']['p2a'], data['scalesDictionary']['p2a'] = 0.0e+0, 1.0e-2, 1.e-2
        data['referenceValuesDictionary']['sca'], data['referenceRangesDictionary']['sca'], data['scalesDictionary']['sca'] = 1.0e-3, 1.0e-3, 1.e-4
        data['referenceValuesDictionary']['sra'], data['referenceRangesDictionary']['sra'], data['scalesDictionary']['sra'] = 1.0e-3, 1.0e-3, 1.e-4
        # load dictionaries referenceValuesDictionary, referenceRangesDictionary and scales: keys oc and or
        data['referenceValuesDictionary']['oc'], data['referenceRangesDictionary']['oc'], data['scalesDictionary']['oc'] = None, 2.0e+2, 1.e+1
        data['referenceValuesDictionary']['or'], data['referenceRangesDictionary']['or'], data['scalesDictionary']['or'] = None, 2.0e+2, 1.e+1
        #
        # check referenceValuesDictionary, referenceRangesDictionary and scalesDictionary
        if not (all([set(data[item].keys()) == set(data['unifiedVariablesKeys'] + ['fd']) for item in ['referenceValuesDictionary', 'referenceRangesDictionary', 'scalesDictionary']])):
            print('*** LoadDataBasic0: referenceValuesDictionary, referenceRangesDictionary or scalesDictionary'); assert False
        #
        # load scales for extrinsicVariables, unifiedVariables and selectedUnifiedVariables (not useful for intrinsicVariables)
        data['extrinsicVariablesScales'] = Dictionary2Array(data['extrinsicVariablesKeys'], data['scalesDictionary'])
        data['unifiedVariablesScales'] = Dictionary2Array(data['unifiedVariablesKeys'], data['scalesDictionary'])
        data['selectedUnifiedVariablesScales'] = Dictionary2Array(data['selectedUnifiedVariablesKeys'], data['scalesDictionary'])
        #
        # load radius of the Earth and order of the polynomial of the horizon for the distorted image
        data['radiusOfEarth'] = 6.371e+6
        data['orderOfTheHorizonPolynomial'] = 5 # IMP*
        #
    except:
        print('*** LoadDataBasic0: error obtaining data'); assert False
    #
    return data
def VariablesScaling(dataBasic, variables, variablesName, direction): # 202106141800
    #
    ''' comments:
    .- input dataBasic is a dictionary (including at least variablesName + 'VariablesScales')
    .- input variables is a float-ndarray
    .- input variablesName is a string ('extrinsic', 'unified' or 'selectedUnified')
    .- input direction is a string ('scale' or 'unscale')
    .- output variables is a float-ndarray of the same length of input variables
    '''
    #
    # obtain scales
    try:
        scales = dataBasic[variablesName + 'VariablesScales']
    except:
        if not (variablesName + 'VariablesScales' in dataBasic.keys()):
            print('*** VariablesScaling: check keys of dataBasic'); assert False
        else:
            print('*** VariablesScaling: error obtaining scales'); assert False
    #
    # scale or unscale variables
    try:
        if direction == 'scale':
            variables = variables / scales
        elif direction == 'unscale':
            variables = variables * scales
        else:
            print('*** VariablesScaling: check direction'); assert False
    except:
        if not (all([isinstance(item, (np.ndarray)) for item in [variables, scales]]) and len(variables) == len(scales)):
            print('*** VariablesScaling: check formats and lenghts of variables and scales'); assert False
        else:
            print('*** VariablesScaling: error scaling variables'); assert False
    #
    return variables
def N2K(n): # 202106141800
    #
    ''' comments:
    .- input n is an integer or a float
    .- output k is a float
    '''
    #
    # obtain k
    try:
        k = (n - 1.) / 2.
    except:
        print('*** N2K: error obtaining k'); assert False
    #
    return k
def K2N(k): # 202106141800
    #
    ''' comments:
    .- input k is a float
    .- output n is a float
    '''
    #
    # obtain n
    try:
        n = 2. * k + 1.
    except:
        print('*** K2N: error obtaining n'); assert False
    #
    return n
#
# angles, unit vectors and rotations
def EulerianAngles2UnitVectors(ph, sg, ta): # 202106141800
    #
    ''' comments:
    .- input ph, sg and ta are floats
    .- output eu, ev and ef are 3-float-ndarrays which are orthonormal and positively oriented
    '''
    #
    # obtain unit vectors
    try:
        # obtain sin and cos of the eulerian angles
        sph, cph = np.sin(ph), np.cos(ph)
        ssg, csg = np.sin(sg), np.cos(sg)
        sta, cta = np.sin(ta), np.cos(ta)
        #
        # obtain unit vector eu
        eux = +csg * cph - ssg * sph * cta
        euy = -csg * sph - ssg * cph * cta
        euz = -ssg * sta
        eu = np.asarray([eux, euy, euz])
        #
        # obtain unit vector ev
        evx = -ssg * cph - csg * sph * cta
        evy = +ssg * sph - csg * cph * cta
        evz = -csg * sta
        ev = np.asarray([evx, evy, evz])
        #
        # obtain unit vector ef
        efx = +sph * sta
        efy = +cph * sta
        efz = -cta
        ef = np.asarray([efx, efy, efz])
        #
        # check (avoidable?)
        R = UnitVectors2R(eu, ev, ef)
        if not (np.allclose(np.dot(R, np.transpose(R)), np.eye(3)) and np.allclose(np.linalg.det(R), 1.)):
            print('*** EulerianAngles2UnitVectors: error checking unit vectors'); assert False
    except:
        print('*** EulerianAngles2UnitVectors: error obtaining unit vectors'); assert False
    #
    return eu, ev, ef
def UnitVectors2R(eu, ev, ef): # 202106141800
    #
    ''' comments:
    .- input eu, ev and ef are 3-float-ndarrays
    .- output R is a 3x3-float-ndarray
      .- the rows of R are eu, ev and ef
    '''
    #
    # obtain R
    try:
        # obtain R
        R = np.asarray([eu, ev, ef])
        #
        # check (avoidable?)
        if not (np.allclose(R[0, :], eu) and np.allclose(R[1, :], ev) and np.allclose(R[2, :], ef)):
            print('*** UnitVectors2R: error checking R'); assert False
    except:
        print('*** UnitVectors2R: error obtaining R'); assert False
    #
    return R
def EulerianAngles2R(ph, sg, ta): # 202106141800
    #
    ''' comments:
    .- input ph, sg and ta are floats
    .- output R is a orthonormal 3x3-float-ndarray with det = +1
    '''
    #
    # obtain R
    try:
        eu, ev, ef = EulerianAngles2UnitVectors(ph, sg, ta)
        R = UnitVectors2R(eu, ev, ef)
    except:
        print('*** EulerianAngles2R: error obtaining R'); assert False
    #
    return R
def R2UnitVectors(R): # 202106141800
    #
    ''' comments:
    .- input R is a 3x3-float-ndarray
      .- the rows of R are eu, ev and ef
    .- output eu, ev and ef are 3-float-ndarrays
    '''
    #
    # obtain unit vectors
    try:
        # check
        if not R.shape == (3, 3):
            print('*** R2UnitVectors: check shape of R'); assert False
        #
        # obtain unit vectors
        eu, ev, ef = R[0, :], R[1, :], R[2, :]
    except:
        print('*** R2UnitVectors: error obtaining unit vectors'); assert False
    #
    return eu, ev, ef
#
# transformations: mainSet and XYZ <-> pixels
def UnifiedVariables2MainSet(nc, nr, unifiedVariables): # 202106141800
    #
    ''' comments:
    .- input nc and nr are integers or floats
    .- input unifiedVariables is a 14-float-ndarray (xc, yc, zc, ph, sg, ta, k1a, k2a, p1a, p2a, sca, sra, oc, or)
    .- output mainSet is a dictionary
    '''
    #
    unifiedVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']
    #
    # obtain mainSet
    try:
        # initialize mainSet with nc and nr
        mainSet = {'nc':nc, 'nr':nr}
        #
        # obtain and load unifiedVariablesDictionary and unifiedVariables ensuring sca and sra are not null
        unifiedVariablesDictionary = Array2Dictionary(unifiedVariablesKeys, unifiedVariables)
        unifiedVariablesDictionary['sca'] = ClipWithSign(unifiedVariablesDictionary['sca'], 1.e-8, 1.e+8)
        unifiedVariablesDictionary['sra'] = ClipWithSign(unifiedVariablesDictionary['sra'], 1.e-8, 1.e+8)
        unifiedVariables = Dictionary2Array(unifiedVariablesKeys, unifiedVariablesDictionary)
        mainSet['unifiedVariablesDictionary'] = unifiedVariablesDictionary
        mainSet.update(mainSet['unifiedVariablesDictionary']) # IMP* (absorb)
        mainSet['unifiedVariables'] = unifiedVariables
        #
        # load sua, sva and pc
        mainSet['sua'], mainSet['sva'] = mainSet['sca'], mainSet['sra'] # IMP* ('sua' and 'sva' are not in unifiedVariablesDictionary)
        mainSet['pc'] = np.asarray([mainSet['xc'], mainSet['yc'], mainSet['zc']])
        #
        # obtain and load orthonormal matrix R and orthonormal unit vectors eu, ev and ef (rows of R)
        R = EulerianAngles2R(mainSet['ph'], mainSet['sg'], mainSet['ta'])
        eu, ev, ef = R2UnitVectors(R)
        mainSet['R'] = R
        mainSet['eu'], (mainSet['eux'], mainSet['euy'], mainSet['euz']) = eu, eu
        mainSet['ev'], (mainSet['evx'], mainSet['evy'], mainSet['evz']) = ev, ev
        mainSet['ef'], (mainSet['efx'], mainSet['efy'], mainSet['efz']) = ef, ef
    except:
        print('*** UnifiedVariables2MainSet: error obtaining mainSet'); assert False
    #
    return mainSet
def XYZ2PositionsRightSideOfCamera(mainSet, xs, ys, zs): # 202106141800
    #
    '''
    .- input mainSet is a dictionary (including at least 'xc', 'yc', 'zc' and 'ef')
    .- input xs, ys and zs are float-ndarrays of the same length
    .- output possRightSideOfCamera is a integer-list
    '''
    #
    # obtain possRightSideOfCamera
    try:
        xas, yas, zas = xs - mainSet['xc'], ys - mainSet['yc'], zs - mainSet['zc']
        possRightSideOfCamera = np.where(xas * mainSet['ef'][0] + yas * mainSet['ef'][1] + zas * mainSet['ef'][2] > 0)[0]
    except:
        if not (set(['xc', 'yc', 'zc', 'ef']) <= set(mainSet.keys())):
            print('*** XYZ2PositionsRightSideOfCamera: check keys of mainSet'); assert False
        elif not (all([isinstance(item, (np.ndarray)) for item in [xs, ys, zs]]) and len(xs) == len(ys) == len(zs)):
            print('*** XYZ2PositionsRightSideOfCamera: check formats and lengths of xs, ys and zs'); assert False
        else:
            print('*** XYZ2PositionsRightSideOfCamera: error obtaining possRightSideOfCamera'); assert False
    #
    return possRightSideOfCamera
def CR2UaVa(mainSet, cs, rs): # c and r -> u* and v* 202106141800
    #
    ''' comments:
    .- input mainSet is a dictionary (including at least 'sua', 'sva', 'oc' and 'or')
      .- mainSet['sua'] and mainSet['sva'] are non-zero, but allowed to be negative
    .- input cs and rs are floats or float-ndarrays
    .- output uas and vas are floats or float-ndarrays
    '''
    #
    # obtain uas and vas
    try:
        uas = (cs - mainSet['oc']) * mainSet['sua']
        vas = (rs - mainSet['or']) * mainSet['sva']
    except:
        if not (set(['sua', 'sva', 'oc', 'or']) <= set(mainSet.keys())):
            print('*** CR2UaVa: check keys of mainSet'); assert False
        else:
            print('*** CR2UaVa: error obtaining uas and vas'); assert False
    #
    return uas, vas
def UaVa2CR(mainSet, uas, vas): # u* and v* -> c and r 202106141800
    #
    ''' comments:
    .- input mainSet is a dictionary (including at least 'sua', 'sva', 'oc' and 'or')
      .- mainSet['sua'] and mainSet['sva'] are non-zero, but allowed to be negative
    .- input uas and vas are floats or float-ndarrays
    .- output cs and rs are floats or float-ndarrays
    '''
    #
    # obtain cs and rs
    try:
        cs = uas / mainSet['sua'] + mainSet['oc']
        rs = vas / mainSet['sva'] + mainSet['or']
    except:
        if not (set(['sua', 'sva', 'oc', 'or']) <= set(mainSet.keys())):
            print('*** UaVa2CR: check keys of mainSet'); assert False
        elif not all([np.abs(mainSet[item]) > 0. for item in ['sua', 'sva']]):
            print('*** UaVa2CR: check values of sua and sva in mainSet'); assert False
        else:
            print('*** UaVa2CR: error obtaining cs and rs'); assert False
    #
    return cs, rs

def UUaVUa2UDaVDa(mainSet, uUas, vUas): # uU* and vU* -> uD* and vD* 202106141800
    #
    ''' comments:
    .- input mainSet is a dictionary (including at least 'k1a', 'k2a', 'p1a' and 'p2a')
    .- input uUas and vUas are floats or float-ndarrays of the same length
    .- output uDas and vDas are floats or float-ndarrays of the same length as uUas and vUas
    '''
    #
    # obtain uDas and vDas
    try:
        aux1s = uUas ** 2 + vUas ** 2
        aux2s = 1. + mainSet['k1a'] * aux1s + mainSet['k2a'] * aux1s ** 2
        aux3s = 2. * uUas * vUas
        aux4s = aux1s + 2. * uUas ** 2
        aux5s = aux1s + 2. * vUas ** 2
        uDas = uUas * aux2s + mainSet['p2a'] * aux4s + mainSet['p1a'] * aux3s
        vDas = vUas * aux2s + mainSet['p1a'] * aux5s + mainSet['p2a'] * aux3s
    except:
        if not (set(['k1a', 'k2a', 'p1a', 'p2a']) <= set(mainSet.keys())):
            print('*** UUaVUa2UDaVDa: check keys of mainSet'); assert False
        else:
            print('*** UUaVUa2UDaVDa: error obtaining uDas and vDas'); assert False
    #
    return uDas, vDas
def UDaVDa2UUaVUa(mainSet, uDas, vDas): # uD* and vD* -> uU* and vU* can be expensive 202106141800
    #
    ''' comments:
    .- input mainSet is a dictionary (including at least 'k1a', 'k2a', 'p1a', 'p2a')
    .- input uDas and vDas are float-ndarrays of the same length
    .- output uUas and vUas are float-ndarrays of the same length or Nones (if it does not succeed)
    .- the funcion is implicit unless k2a = p1a = p2a = 0
    '''
    #
    def DeltaAndError(mainSet, uDas, vDas, uUas, vUas): # 202106141800
        #
        ''' comments:
        .- input mainSet is a dictionary (including at least 'k1a', 'k2a', 'p1a', 'p2a')
        .- input uDas, vDas, uUas and vUas are float-ndarrays of the same length
        .- ouput duUas and dvUas are float-ndarrays of the same length as uDas, vDas, uUas and vUas
        .- output error is a float
        '''
        #
        # obtain duUas, dvUas and error
        try:
            uDasN, vDasN = UUaVUa2UDaVDa(mainSet, uUas, vUas)
            fxs, fys = uDasN - uDas, vDasN - vDas # errors
            error = np.max([np.max(np.abs(fxs)), np.max(np.abs(fys))])
            # aux1s = uUas ** 2 + vUas ** 2
            aux1s = uUas ** 2 + vUas ** 2
            aux1suUa = 2. * uUas
            aux1svUa = 2. * vUas
            # aux2s = 1. + mainSet['k1a'] * aux1s + mainSet['k2a'] * aux1s ** 2
            aux2s = 1. + mainSet['k1a'] * aux1s + mainSet['k2a'] * aux1s ** 2
            aux2suUa = mainSet['k1a'] * aux1suUa + mainSet['k2a'] * 2. * aux1s * aux1suUa
            aux2svUa = mainSet['k1a'] * aux1svUa + mainSet['k2a'] * 2. * aux1s * aux1svUa
            # aux3s = 2. * uUas * vUas
            aux3suUa = 2. * vUas
            aux3svUa = 2. * uUas
            # aux4s = aux1s + 2. * uUas ** 2
            aux4suUa = aux1suUa + 4. * uUas
            aux4svUa = aux1svUa
            # aux5s = aux1s + 2. * vUas ** 2
            aux5suUa = aux1suUa
            aux5svUa = aux1svUa + 4. * vUas
            # uDas = uUas * aux2s + mainSet['p2a'] * aux4s + mainSet['p1a'] * aux3s
            JuUasuUa = aux2s + uUas * aux2suUa + mainSet['p2a'] * aux4suUa + mainSet['p1a'] * aux3suUa
            JuUasvUa = uUas * aux2svUa + mainSet['p2a'] * aux4svUa + mainSet['p1a'] * aux3svUa
            # vDas = vUas * aux2s + mainSet['p1a'] * aux5s + mainSet['p2a'] * aux3s
            JvUasuUa = vUas * aux2suUa + mainSet['p1a'] * aux5suUa + mainSet['p2a'] * aux3suUa
            JvUasvUa = aux2s + vUas * aux2svUa + mainSet['p1a'] * aux5svUa + mainSet['p2a'] * aux3svUa
            # inversion of the Jacobian
            determinants = JuUasuUa * JvUasvUa - JuUasvUa * JvUasuUa
            determinants = ClipWithSign(determinants, 1.e-8, 1.e+8)
            JinvuUasuUa = + JvUasvUa / determinants
            JinvvUasvUa = + JuUasuUa / determinants
            JinvuUasvUa = - JuUasvUa / determinants
            JinvvUasuUa = - JvUasuUa / determinants
            duUas = - JinvuUasuUa * fxs - JinvuUasvUa * fys
            dvUas = - JinvvUasuUa * fxs - JinvvUasvUa * fys
        except:
            if not (set(['k1a', 'k2a', 'p1a', 'p2a']) <= set(mainSet.keys())):
                print('*** DeltaAndError: check keys of mainSet'); assert False
            else:
                print('*** DeltaAndError: error obtaining duUas, dvUas and error'); assert False
        #
        return duUas, dvUas, error
    #
    # obtain uUas and vUas
    try:
        # manage points at the origin (singularity)
        possZero = np.where(np.sqrt(uDas ** 2 + vDas ** 2) < 1.e-11)[0]
        if len(possZero) > 0:
            uDas[possZero], vDas[possZero] = [np.ones(len(possZero)) for item in range(2)] # set to one to proceed (see possZero below)
        #
        # obtain uUas and vUas
        if np.allclose([mainSet['k2a'], mainSet['p1a'], mainSet['p2a']], [0., 0., 0.]): # explicit
            if np.allclose(mainSet['k1a'], 0.):
                uUas, vUas = uDas * 1., vDas * 1.
            else: # Cardano's solution for cubic equations
                aux0s = np.sqrt(uDas ** 2 + vDas ** 2)
                p, qs = 1. / mainSet['k1a'], - aux0s / mainSet['k1a']
                Ds = qs ** 2 + 4. / 27. * p ** 3 # discriminant
                aux1s = np.zeros(Ds.shape)
                pos0, posP, posN = np.where(Ds == 0.)[0], np.where(Ds > 0.)[0], np.where(Ds < 0.)[0]
                if not (len(posP) + len(posN) + len(pos0) == len(Ds)):
                    print('*** UDaVDa2UUaVUa: error in the Cardano solution'); assert False
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
        else: # implicit
            uUas, vUas, converged, speed, counter, error = 1. * uDas, 1. * vDas, False, 1., 0, 1.e+10
            while not converged and counter <= 20: # IMP*
                duUas, dvUas, errorN = DeltaAndError(mainSet, uDas, vDas, uUas, vUas)
                if errorN > 2. * error:
                    break
                uUas, vUas, error = uUas + speed * duUas, vUas + speed * dvUas, 1. * errorN
                converged, counter = error <= 1.e-11, counter + 1
        #
        # manage not converged
        if not converged:
            return None, None
        #
        # manage points at the origin (singularity)
        if len(possZero) > 0:
            uDas[possZero], vDas[possZero] = [np.zeros(len(possZero)) for item in range(2)]
            uUas[possZero], vUas[possZero] = [np.zeros(len(possZero)) for item in range(2)]
        #
        # check
        uDasR, vDasR = UUaVUa2UDaVDa(mainSet, uUas, vUas)
        if not (max([np.max(np.abs(uDasR - uDas)), np.max(np.abs(vDasR - vDas))]) < 5. * 1.e-11):
            print('*** UDaVDa2UUaVUa: the solution is not good enough'); assert False
    except:
        if not (set(['k1a', 'k2a', 'p1a', 'p2a']) <= set(mainSet.keys())):
            print('*** UDaVDa2UUaVUa: check keys of mainSet'); assert False
        elif not (all([isinstance(item, (np.ndarray)) for item in [uDas, vDas]]) and len(uDas) == len(vDas)):
            print('*** UDaVDa2UUaVUa: check formats and lengths of uDas and rDas'); assert False
        else:
            print('*** UDaVDa2UUaVUa: error obtaining uUas and vUas'); assert False
    #
    return uUas, vUas
def CURU2CDRD(mainSet, cUs, rUs): # 202106141800
    #
    ''' comments:
    .- input mainSet is a dictionary
    .- input cUs and rUs are floats or float-ndarrays of the same length
    .- output cDs and rDs are floats or float-ndarrays of the same length as cUs and rUs
    '''
    #
    # obtain cDs ad rDs
    try:
        uUas, vUas = CR2UaVa(mainSet, cUs, rUs)
        uDas, vDas = UUaVUa2UDaVDa(mainSet, uUas, vUas)
        cDs, rDs = UaVa2CR(mainSet, uDas, vDas)
    except:
        print('*** CURU2CDRD: error obtaining cDs and rDs'); assert False
    #
    return cDs, rDs
def CDRD2CURU(mainSet, cDs, rDs): # can be expensive 202106141800
    #
    ''' comments:
    .- input mainSet is a dictionary
    .- input cDs and rDs are float-ndarrays of the same length
    .- output cUs and rUs are float-ndarrays of the same length as cDs and rDs or None (if it does not succeed)
    '''
    #
    # obtain cUs ad rUs
    try:
        uDas, vDas = CR2UaVa(mainSet, cDs, rDs)
        uUas, vUas = UDaVDa2UUaVUa(mainSet, uDas, vDas) # can be expensive
        if any([item is None for item in [uUas, vUas]]):
            cUs, rUs = None, None
        else:
            cUs, rUs = UaVa2CR(mainSet, uUas, vUas)
    except:
        print('*** CDRD2CURU: error obtaining cUs and rUs'); assert False
    #
    return cUs, rUs
def XYZ2UUaVUa(mainSet, xs, ys, zs, options={}): # 202106141800
    #
    ''' comments:
    .- input mainSet is a dictionary (including at least 'xc', 'yc', 'zc', 'eu', 'ev' and 'ef')
    .- input xs, ys and zs are float-ndarrays of the same length
    .- output uUas and vUas are float-ndarrays of the same length as xs, ys and zs
    .- output possRightSideOfCamera is a list of integers or None (if not options['returnPositionsRightSideOfCamera'])
    '''
    #
    # complete options
    try:
        keys, defaultValues = ['returnPositionsRightSideOfCamera'], [False]
        options = CompleteADictionary(options, keys, defaultValues)
    except:
        print('*** XYZ2UUaVUa: error completing options')
    #
    # obtain uUas, vUas and possRightSideOfCamera
    try:
        # obtain uUas and vUas
        xas, yas, zas = xs - mainSet['xc'], ys - mainSet['yc'], zs - mainSet['zc']
        nus = xas * mainSet['eu'][0] + yas * mainSet['eu'][1] + zas * mainSet['eu'][2]
        nvs = xas * mainSet['ev'][0] + yas * mainSet['ev'][1] + zas * mainSet['ev'][2]
        dns = xas * mainSet['ef'][0] + yas * mainSet['ef'][1] + zas * mainSet['ef'][2]
        dns = ClipWithSign(dns, 1.e-8, 1.e+8)
        uUas = nus / dns
        vUas = nvs / dns
        #
        # obtain possRightSideOfCamera
        if options['returnPositionsRightSideOfCamera']:
            possRightSideOfCamera = XYZ2PositionsRightSideOfCamera(mainSet, xs, ys, zs)
        else:
            possRightSideOfCamera = None
    except:
        if not (set(['xc', 'yc', 'zc', 'eu', 'ev', 'ef']) <= set(mainSet.keys())):
            print('*** XYZ2UUaVUa: check keys of mainSet'); assert False
        elif not (all([isinstance(item, (np.ndarray)) for item in [xs, ys, zs]]) and len(xs) == len(ys) == len(zs)):
            print('*** XYZ2UUaVUa: check formats and lengths of xs, ys and zs'); assert False
        else:
            print('*** XYZ2UUaVUa: error obtaining uUas and vUas'); assert False
    #
    return uUas, vUas, possRightSideOfCamera
def UUaVUa2XYZ(mainSet, planes, uUas, vUas, options={}): # can be expensive 202106141800
    #
    ''' comments:
    .- input mainSet is a dictionary (including at least 'eu', 'ev', 'ef', 'pc')
    .- input planes is a dictionary (including at least 'pxs', 'pys', 'pzs' and 'pts')
      .- input planes['pxs'/'pys'/'pzs'/'pts'] is a float (single plane) or a float-ndarray of the same length as uUas and vUas (different planes for each point)
    .- input uUas and vUas are float-ndarrays of the same length
    .- output xs, ys, zs are float-ndarrays of the same length as uUas and vUas
    .- output possRightSideOfCamera is a list of integers or None (if not options['returnPositionsRightSideOfCamera'])
    '''
    #
    # complete options
    try:
        keys, defaultValues = ['returnPositionsRightSideOfCamera'], [False]
        options = CompleteADictionary(options, keys, defaultValues)
    except:
        print('*** UUaVUa2XYZ: error completing options'); assert False
        
    #
    # obtain and check xs, ys and zs
    try:
        # obtain the matrices of the system
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
        #
        # obtain the solution
        # A11 A12 A13  x  b1
        # A21 A22 A23  y  b2
        # A31 A32 A33  z  b3
        auxs = A11s * (A22s * A33s - A23s * A32s) + A12s * (A23s * A31s - A21s * A33s) + A13s * (A21s * A32s - A22s * A31s)
        auxs = ClipWithSign(auxs, 1.e-8, 1.e+8)
        xs = (bb1s * (A22s * A33s - A23s * A32s) + A12s * (A23s * bb3s - bb2s * A33s) + A13s * (bb2s * A32s - A22s * bb3s)) / auxs
        ys = (A11s * (bb2s * A33s - A23s * bb3s) + bb1s * (A23s * A31s - A21s * A33s) + A13s * (A21s * bb3s - bb2s * A31s)) / auxs
        zs = (A11s * (A22s * bb3s - bb2s * A32s) + A12s * (bb2s * A31s - A21s * bb3s) + bb1s * (A21s * A32s - A22s * A31s)) / auxs
        #
        # check the solution
        poss = np.where(np.max(np.asarray([np.abs(xs), np.abs(ys), np.abs(zs)]), axis=0) < 1.e+8)[0]
        if isinstance(planes['pxs'], (np.ndarray)):
            auxs = planes['pxs'][poss] * xs[poss] + planes['pys'][poss] * ys[poss] + planes['pzs'][poss] * zs[poss] + planes['pts'][poss]
        else:
            auxs = planes['pxs'] * xs[poss] + planes['pys'] * ys[poss] + planes['pzs'] * zs[poss] + planes['pts']
        if not np.allclose(auxs, np.zeros(len(poss))): # plane
            print('*** UUaVUa2XYZ: error checking the solution (1)'); assert False
        uUasR, vUasR = XYZ2UUaVUa(mainSet, xs[poss], ys[poss], zs[poss], options={})[0:2]
        if not (np.allclose(uUasR, uUas[poss]) and np.allclose(vUasR, vUas[poss])): # inversion
            print('*** UUaVUa2XYZ: error checking the solution (2)'); assert False
    except:
        if not (all([isinstance(item, (np.ndarray)) for item in [uUas, vUas]]) and len(uUas) == len(vUas)):
            print('*** UUaVUa2XYZ: check formats and lengths of uUas and vUas'); assert False
        elif not (set(['eu', 'ev', 'ef', 'pc']) <= set(mainSet.keys())):
            print('*** UUaVUa2XYZ: check keys of mainSet'); assert False
        elif not (set(['pxs', 'pys', 'pzs', 'pts']) <= set(planes.keys())):
            print('*** UUaVUa2XYZ: check keys of planes'); assert False
        elif isinstance(planes['pxs'], (np.ndarray)):
            if not all([isinstance(planes[item], (np.ndarray)) for item in ['pxs', 'pys', 'pzs', 'pts']]):
                print('*** UUaVUa2XYZ: check formats of pxs, pys, pzs, pts'); assert False
            elif not all([len(planes[item]) == len(uUas) for item in ['pxs', 'pys', 'pzs', 'pts']]):
                print('*** UUaVUa2XYZ: check lengths of pxs, pys, pzs, pts'); assert False
        else:
            print('*** UUaVUa2XYZ: error obtaining and checking xs, ys and zs'); assert False
    #
    # obtain possRightSideOfCamera
    try:
        if options['returnPositionsRightSideOfCamera']:
            possRightSideOfCamera = XYZ2PositionsRightSideOfCamera(mainSet, xs, ys, zs)
        else:
            possRightSideOfCamera = None
    except:
        print('*** UUaVUa2XYZ: error obtaining possRightSideOfCamera'); assert False
    #
    return xs, ys, zs, possRightSideOfCamera
def XYZ2CDRD(mainSet, xs, ys, zs, options={}): # 202106141800
    #
    ''' comments:
    .- input mainSet is a dictionary (including at least 'nc' and 'nr')
    .- input xs, ys and zs are float-ndarrays of the same length
    .- output cDs and rDs are float-ndarrays of the same length as xs, ys and zs
    .- output possGood is an integer-list or None (if not options['returnGoodPositions'])
    '''
    #
    # complete options
    try:
        keys, defaultValues = ['imgMargins', 'returnGoodPositions'], [{'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}, False]
        options = CompleteADictionary(options, keys, defaultValues)
        ''' comments:
        .- options['imgMargins'] is a dictionary (see CompleteImgMargins)
        .- options['returnGoodPositions'] is a boolean
        '''
    except:
        print('*** XYZ2CDRD: error completing options')
    #
    # obtain cDs and rDs
    try:
        optionsTMP = {'returnPositionsRightSideOfCamera':options['returnGoodPositions']}
        uUas, vUas, possGood = XYZ2UUaVUa(mainSet, xs, ys, zs, optionsTMP)
        uDas, vDas = UUaVUa2UDaVDa(mainSet, uUas, vUas)
        cDs, rDs = UaVa2CR(mainSet, uDas, vDas)
        #
        # manage options['returnGoodPositions']
        if options['returnGoodPositions']:
            # cDs, rDs within the image
            if len(possGood) > 0:
                nc, nr, cDsGood, rDsGood = mainSet['nc'], mainSet['nr'], cDs[possGood], rDs[possGood]
                optionsTMP = {'imgMargins':options['imgMargins']}
                possGoodInGood = CR2PositionsWithinImage(nc, nr, cDsGood, rDsGood, optionsTMP)
                possGood = [possGood[item] for item in possGoodInGood]
            # xs, ys are recovered from cs, rs and zs # IMP*
            if len(possGood) > 0:
                xsGood, ysGood, zsGood, csGood, rsGood = xs[possGood], ys[possGood], zs[possGood], cDs[possGood], rDs[possGood]
                xsGoodR, ysGoodR = CDRDZ2XY(mainSet, csGood, rsGood, zsGood, options={})[0:2]
                distances = DistanceFromAPointToAPoint(xsGood, ysGood, xsGoodR, ysGoodR)
                possGoodInGood = np.where(distances < 1.e-5)[0]
                possGood = [possGood[item] for item in possGoodInGood]
        else:
            possGood = None
    except:
        if not (set(['nc', 'nr']) <= set(mainSet.keys())):
            print('*** XYZ2CDRD: check keys of mainSet'); assert False
        else:
            print('*** XYZ2CDRD: error obtaining cDs and rDs'); assert False
    #
    return cDs, rDs, possGood
def CDRDZ2XY(mainSet, cDs, rDs, zs, options={}): # can be expensive 202106141800
    #
    ''' comments:
    .- input mainSet is a dictionary
    .- input cDs, rDs and zs are float-arrays of the same length
    .- output xs and ys are float-arrays of the same length as cDs, rDs and zs or None (if it does not succeed)
    .- output possGood is an integer-list or None (if not options['returnGoodPositions'] or it does not succeed)
    '''
    #
    # complete options
    try:
        keys, defaultValues = ['imgMargins', 'returnGoodPositions'], [{'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}, False]
        options = CompleteADictionary(options, keys, defaultValues)
        ''' comments:
        .- options['imgMargins'] is a dictionary (see CompleteImgMargins)
        .- options['returnGoodPositions'] is a boolean
        '''
    except:
        print('*** CDRDZ2XY: error completing options'); assert False
    #
    # obtain xs, ys and possGood
    try:
        # obtain cUs and rUs
        cUs, rUs = CDRD2CURU(mainSet, cDs, rDs) # can be expensive
        #
        # manage cUs or rUs None
        if any([item is None for item in [cUs, rUs]]):
            return None, None, None
        #
        # obtain uU* and vU*
        uUas, vUas = CR2UaVa(mainSet, cUs, rUs)
        #
        # obtain planes
        if isinstance(zs, (np.ndarray)): # array
            planes = {'pxs':np.zeros(zs.shape), 'pys':np.zeros(zs.shape), 'pzs':np.ones(zs.shape), 'pts': -zs}
        else: # scalar
            planes = {'pxs':0., 'pys':0., 'pzs':1., 'pts': -zs}
        #
        # obtain xs, ys and zs (and perhaps possGood as positions at the right side of the camera)
        optionsTMP = {'returnPositionsRightSideOfCamera':options['returnGoodPositions']}
        xs, ys, zs, possGood = UUaVUa2XYZ(mainSet, planes, uUas, vUas, optionsTMP)
        #
        # obtain possGood
        if options['returnGoodPositions']:
            # obtain positions within the image
            if len(possGood) > 0: # so far possGood are at the right side of the camera
                nc, nr, cDsGood, rDsGood = mainSet['nc'], mainSet['nr'], cDs[possGood], rDs[possGood]
                optionsTMP = {'imgMargins':options['imgMargins']}
                possGoodInGood = CR2PositionsWithinImage(nc, nr, cDsGood, rDsGood, optionsTMP)
                possGood = [possGood[item] for item in possGoodInGood]
        else:
            possGood = None
    except:
        print('*** CDRDZ2XY: error obtaining xs, ys and possGood'); assert False
    #
    return xs, ys, possGood
#
# transformations: horizonLine and horizon pixels
def MainSetAndZ02HorizonLine(dataBasic, mainSet, z0): # 202106141800
    #
    ''' comments:
    .- input dataBasic is a dictionary (including at least 'radiusOfEarth' and 'orderOfTheHorizonPolynomial')
    .- input mainSet is a dictionary (including at least 'eu', 'ev', 'ef', 'zc', 'nc', 'nr', 'oc', 'or', 'sua' and 'sva')
    .- input z0 is a float
    .- output horizonLine is a dictionary (including at least 'nc', 'nr', 'z0', 'c*Uh*' and 'ccDh')
    '''
    #
    # initialize horizonLine with nc, nr and z0
    try:
        horizonLine = {item:mainSet[item] for item in ['nc', 'nr']}
        horizonLine['z0'] = z0
    except:
        if not (set(['nc', 'nr']) <= set(mainSet.keys())):
            print('*** MainSetAndZ02HorizonLine: check keys of mainSet ("nc" and "nr")'); assert False
        else:
            print('*** MainSetAndZ02HorizonLine: error initializing horizonLine'); assert False
    #
    # obtain horizonLine undistorted keys: ccUh1 * cUh + crUh1 * rUh + ccUh0 = 0; rUh = -(ccUh1 * cUh + ccUh0) / crUh1
    try:
        TMP = np.sqrt(mainSet['ef'][0] ** 2 + mainSet['ef'][1] ** 2)
        efxp = mainSet['ef'][0] / TMP
        efyp = mainSet['ef'][1] / TMP
        a = mainSet['zc'] - z0
        b = np.sqrt(2. * max(1.e-2, a) * dataBasic['radiusOfEarth'])
        au = b * (efxp * mainSet['eu'][0] + efyp * mainSet['eu'][1]) - (a + mainSet['zc']) * mainSet['eu'][2]
        av = b * (efxp * mainSet['ev'][0] + efyp * mainSet['ev'][1]) - (a + mainSet['zc']) * mainSet['ev'][2]
        af = b * (efxp * mainSet['ef'][0] + efyp * mainSet['ef'][1]) - (a + mainSet['zc']) * mainSet['ef'][2]
        #
        ccUh1 = af * mainSet['eu'][2] * mainSet['sua']
        crUh1 = af * mainSet['ev'][2] * mainSet['sva']
        ccUh0 = - au * mainSet['eu'][2] - av * mainSet['ev'][2] - ccUh1 * mainSet['oc'] - crUh1 * mainSet['or']
        #
        TMP = max([np.sqrt(ccUh1 ** 2 + crUh1 ** 2), 1.e-8])
        horizonLine['ccUh1'] = ccUh1 / TMP
        horizonLine['crUh1'] = crUh1 / TMP
        horizonLine['ccUh0'] = ccUh0 / TMP
        horizonLine['crUh1'] = ClipWithSign(horizonLine['crUh1'], 1.e-8, 1.e+8) 
    except:
        if not ('radiusOfEarth' in dataBasic.keys()):
            print('*** MainSetAndZ02HorizonLine: check keys of dataBasic ("radiusOfEarth")'); assert False
        elif not (set(['eu', 'ev', 'ef', 'zc', 'oc', 'or', 'sua', 'sva']) <= set(mainSet.keys())):
            print('*** MainSetAndZ02HorizonLine: check keys of mainSet (1)'); assert False
        else:
            print('*** MainSetAndZ02HorizonLine: error obtaining horizonLine undistorted keys'); assert False
    #
    # obtain horizon line distorted keys: rDh = ccDh[0] + ccDh[1] * cDh + ccDh[2] * cDh ** 2 + ...
    try:
        cUhMin = -0.1 * mainSet['nc']
        cUhMax = +1.1 * mainSet['nc']
        cUhs = np.linspace(cUhMin, cUhMax, 31, endpoint=True)
        rUhs = CUh2RUh(horizonLine, cUhs)
        cDhs, rDhs = CURU2CDRD(mainSet, cUhs, rUhs) # explicit
        A = np.ones((len(cDhs), dataBasic['orderOfTheHorizonPolynomial'] + 1))
        for power in range(1, dataBasic['orderOfTheHorizonPolynomial'] + 1):
            A[:, power] = cDhs ** power
        b = rDhs
        horizonLine['ccDh'] = np.linalg.solve(np.dot(np.transpose(A), A), np.dot(np.transpose(A), b))
        #
        # manage horizon lines with large errors
        if np.max(np.abs(b - np.dot(A, horizonLine['ccDh']))) > 5e-1: # IMP* pixel error WATCH OUT
            horizonLine['ccDh'] = np.zeros(dataBasic['orderOfTheHorizonPolynomial'] + 1)
            horizonLine['ccDh'][0] = 1.e+5 # WATCH OUT
    except:
        if not ('orderOfTheHorizonPolynomial' in dataBasic.keys()):
            print('*** MainSetAndZ02HorizonLine: check keys of dataBasic ("orderOfTheHorizonPolynomial")'); assert False
        elif not ('nc' in mainSet.keys()):
            print('*** MainSetAndZ02HorizonLine: check keys of mainSet ("nc")'); assert False
        else:
            print('*** MainSetAndZ02HorizonLine: error obtaining horizonLine distorted keys'); assert False
    #
    return horizonLine
def CUh2RUh(horizonLine, cUhs): # 202106141800
    #
    ''' comments:
    .- input horizonLine is a dictionary (including at least 'ccUh1', 'crUh1', 'ccUh0' and involving z0)
      .- the horizon line is 'ccUh1' * cUhs + 'crUh1' * ruhs + 'ccUh0' = 0, i.e., ruhs = - ('ccUh0' + 'ccUh1' * cUhs) / 'crUh1'
    .- input cUhs is a float or a float-ndarray
    .- output rUhs is a float or a float-ndarray
    '''
    #
    # obtain ruhs
    try:
        crUh1 = ClipWithSign(horizonLine['crUh1'], 1.e-8, 1.e+8)
        rUhs = - (horizonLine['ccUh0'] + horizonLine['ccUh1'] * cUhs) / crUh1
    except:
        print('*** CUh2RUh: error obtaining ruhs'); assert False
    #
    return rUhs
def CDh2RDh(horizonLine, cDhs, options={}): # 202106141800
    #
    ''' comments:
    .- input horizonLine is a dictionary (including at least 'ccDh' and involving z0)
      .- the horizon line is rDhs = ccDh[0] + ccDh[1] * cDhs + ccDh[2] * cDhs ** 2 + ...
    .- input cDhs is a float-ndarray
    .- output rDhs is a float-ndarray of the same length as cDhs
    .- output possGood is an integer-list or None (if not options['returnGoodPositions'])
    '''
    #
    # complete options
    try:
        keys, defaultValues = ['imgMargins', 'returnGoodPositions'], [{'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}, False]
        options = CompleteADictionary(options, keys, defaultValues)
        ''' comments:
        .- if options['returnGoodPositions'] is False, then output possGood is None
        .- if options['imgMargins'] is None, then no margins are considered for possGood
        '''
    except:
        print('*** CDh2RDh: error completing options'); assert False
    #
    # obtain rDhs
    try:
        # obtain rDhs
        rDhs = horizonLine['ccDh'][0] * np.ones(cDhs.shape)
        for power in range(1, len(horizonLine['ccDh'])):
            rDhs = rDhs + horizonLine['ccDh'][power] * cDhs ** power
        #
        # obtain possGood
        if options['returnGoodPositions']:
            nc, nr, optionsTMP = horizonLine['nc'], horizonLine['nr'], {'imgMargins':options['imgMargins']}
            possGood = CR2PositionsWithinImage(nc, nr, cDhs, rDhs, optionsTMP)
        else:
            possGood = None
    except:
        if not isinstance(cDhs, (np.ndarray)):
            print('*** CDh2RDh: check format of cDhs'); assert False
        else:
            print('*** CDh2RDh: error obtaining rDhs'); assert False
    #
    return rDhs, possGood
#
# variables: unified and selectedUnified
def SelectedUnifiedVariables2UnifiedVariables(dataBasic, selectedUnifiedVariables, nc, nr): # 202106141800
    #
    ''' comments:
    .- input dataBasic is a dictionary (including at least 'selectedUnifiedVariablesKeys')
    .- input selectedUnifiedVariables a float-ndarray
    .- input nc and nr are integers or floats
    .- output unifiedVariables is a 14-float-ndarray (xc, yc, zc, ph, sg, ta, k1a, k2a, p1a, p2a, sca, sra, oc, or)
      .- if not in selectedUnifiedVariables, then k1a, k2a, p1a and p2a are set to 0
      .- if not in selectedUnifiedVariables, then sra is set to sca
      .- if not in selectedUnifiedVariables, then oc and or are respectively set to kc and kr
    '''
    #
    unifiedVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']
    #
    # obtain unifiedVariables
    try:
        # initialize unifiedVariablesDictionary as selectedUnifiedVariablesDictionary
        unifiedVariablesDictionary = Array2Dictionary(dataBasic['selectedUnifiedVariablesKeys'], selectedUnifiedVariables)
        #
        # complete unifiedVariablesDictionary
        for key in [item for item in unifiedVariablesKeys if item not in dataBasic['selectedUnifiedVariablesKeys']]:
            if key in ['k1a', 'k2a', 'p1a', 'p2a']:
                unifiedVariablesDictionary[key] = 0.
            elif key == 'sra':
                unifiedVariablesDictionary[key] = unifiedVariablesDictionary['sca']
            elif key == 'oc':
                unifiedVariablesDictionary[key] = N2K(nc) # kc
            elif key == 'or':
                unifiedVariablesDictionary[key] = N2K(nr) # kr
            else:
                print('*** SelectedUnifiedVariables2UnifiedVariables: error completing unifiedVariablesDictionary'); assert False
        unifiedVariables = Dictionary2Array(unifiedVariablesKeys, unifiedVariablesDictionary)
        #
        # check unifiedVariables
        selectedUnifiedVariablesR = UnifiedVariables2SelectedUnifiedVariables(dataBasic, unifiedVariables)
        if not np.allclose(selectedUnifiedVariablesR, selectedUnifiedVariables):
            print('*** SelectedUnifiedVariables2UnifiedVariables: error checking unifiedVariables'); assert False
    except:
        if not (set(['selectedUnifiedVariablesKeys']) <= set(dataBasic.keys())):
            print('*** SelectedUnifiedVariables2UnifiedVariables: check keys of dataBasic'); assert False
        else:
            print('*** SelectedUnifiedVariables2UnifiedVariables: error obtaining unifiedVariables'); assert False
    #
    return unifiedVariables
def UnifiedVariables2SelectedUnifiedVariables(dataBasic, unifiedVariables): # 202106141800
    #
    ''' comments:
    .- input dataBasic is a dictionary (including at least 'selectedUnifiedVariablesKeys')
    .- input unifiedVariables is a 14-float-ndarray (xc, yc, zc, ph, sg, ta, k1a, k2a, p1a, p2a, sca, sra, oc, or)
    .- output selectedUnifiedVariables is a float-ndarray, subset of unifiedVariables
    '''
    #
    unifiedVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']
    #
    # obtain selectedUnifiedVariables
    try:
        unifiedVariablesDictionary = Array2Dictionary(unifiedVariablesKeys, unifiedVariables)
        selectedUnifiedVariables = Dictionary2Array(dataBasic['selectedUnifiedVariablesKeys'], unifiedVariablesDictionary)
    except:
        if not ('selectedUnifiedVariablesKeys' in dataBasic.keys()):
            print('*** UnifiedVariables2SelectedUnifiedVariables: check keys of dataBasic'); assert False
        else:
            print('*** UnifiedVariables2SelectedUnifiedVariables: error obtaining selectedUnifiedVariables'); assert False
    #
    return selectedUnifiedVariables
#
# variables: extrinsic, intrinsic and unified
def IntrinsicVariablesAndUnifiedVariables2Fd(intrinsicVariables, unifiedVariables): # 202106141800
    #
    ''' comments:
    .- input intrinsicVariables is a 8-float-ndarray (k1, k2, p1, p2, sc, sr, oc, or)
    .- input unifiedVariables is a 14-float-ndarray (xc, yc, zc, ph, sg, ta, k1a, k2a, p1a, p2a, sca, sra, oc, or)
    .- output fd is a float (focal distance)
    '''
    #
    intrinsicVariablesKeys = ['k1', 'k2', 'p1', 'p2', 'sc', 'sr', 'oc', 'or']
    unifiedVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']
    #
    # obtain fd
    try:
        intrinsicVariablesDictionary = Array2Dictionary(intrinsicVariablesKeys, intrinsicVariables)
        unifiedVariablesDictionary = Array2Dictionary(unifiedVariablesKeys, unifiedVariables)
        fd = intrinsicVariablesDictionary['sc'] / unifiedVariablesDictionary['sca']
    except:
        print('*** IntrinsicVariablesAndUnifiedVariables2Fd: error obtaining fd'); assert False
    #
    return fd
def AreIntrinsicVariablesAndUnifiedVariablesCompatible(intrinsicVariables, unifiedVariables): # 202106141800
    #
    ''' comments:
    .- input intrinsicVariables is a 8-float-ndarray (k1, k2, p1, p2, sc, sr, oc, or)
    .- input unifiedVariables is a 14-float-ndarray (xc, yc, zc, ph, sg, ta, k1a, k2a, p1a, p2a, sca, sra, oc, or)
    .- output areCompatible is a boolean
    '''
    #
    intrinsicVariablesKeys = ['k1', 'k2', 'p1', 'p2', 'sc', 'sr', 'oc', 'or']
    unifiedVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']
    #
    # obtain areCompatible
    try:
        # obtain intrinsicVariablesDictionary and unifiedVariablesDictionary
        intrinsicVariablesDictionary = Array2Dictionary(intrinsicVariablesKeys, intrinsicVariables)
        unifiedVariablesDictionary = Array2Dictionary(unifiedVariablesKeys, unifiedVariables)
        #
        # obtain fd
        fd = IntrinsicVariablesAndUnifiedVariables2Fd(intrinsicVariables, unifiedVariables)
        if not fd > 0.:
            print('*** AreIntrinsicVariablesAndUnifiedVariablesCompatible: non-positive focal distance fd'); assert False
        #
        # obtain areCompatible
        aux0s = Dictionary2Array(['k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or'], unifiedVariablesDictionary)
        aux1s = Dictionary2Array(['k1', 'k2', 'p1', 'p2', 'sc', 'sr', 'oc', 'or'], intrinsicVariablesDictionary)
        aux2s = np.asarray([fd ** 2, fd ** 4, fd ** 1, fd ** 1, fd ** -1, fd ** -1,  1.,  1.])
        areCompatible = np.allclose(aux0s, aux1s * aux2s)
    except:
        print('*** AreIntrinsicVariablesAndUnifiedVariablesCompatible: error obtaining areCompatible'); assert False
    #
    return areCompatible

def ExtrinsicVariablesAndIntrinsicVariables2UnifiedVariables(extrinsicVariables, intrinsicVariables): # 202106141800
    #
    ''' comments:
    .- input extrinsicVariables is a 7-float-ndarray (xc, yc, zc, ph, sg, ta, fd)
    .- input intrinsicVariables is a 8-float-ndarray (k1, k2, p1, p2, sc, sr, oc, or)
    .- output unifiedVariables is a 14-float-ndarray (xc, yc, zc, ph, sg, ta, k1a, k2a, p1a, p2a, sca, sra, oc, or)
    '''
    #
    extrinsicVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'fd']
    intrinsicVariablesKeys = ['k1', 'k2', 'p1', 'p2', 'sc', 'sr', 'oc', 'or']
    unifiedVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']
    #
    # obtain unifiedVariables
    try:
        # obtain extrinsicVariablesDictionary and intrinsicVariablesDictionary
        extrinsicVariablesDictionary = Array2Dictionary(extrinsicVariablesKeys, extrinsicVariables)
        intrinsicVariablesDictionary = Array2Dictionary(intrinsicVariablesKeys, intrinsicVariables)
        #
        # obtain unifiedVariablesDictionary
        unifiedVariablesDictionary = {}
        unifiedVariablesDictionary['xc'] = extrinsicVariablesDictionary['xc']
        unifiedVariablesDictionary['yc'] = extrinsicVariablesDictionary['yc']
        unifiedVariablesDictionary['zc'] = extrinsicVariablesDictionary['zc']
        unifiedVariablesDictionary['ph'] = extrinsicVariablesDictionary['ph']
        unifiedVariablesDictionary['sg'] = extrinsicVariablesDictionary['sg']
        unifiedVariablesDictionary['ta'] = extrinsicVariablesDictionary['ta']
        unifiedVariablesDictionary['k1a'] = intrinsicVariablesDictionary['k1'] * extrinsicVariablesDictionary['fd'] ** 2
        unifiedVariablesDictionary['k2a'] = intrinsicVariablesDictionary['k2'] * extrinsicVariablesDictionary['fd'] ** 4
        unifiedVariablesDictionary['p1a'] = intrinsicVariablesDictionary['p1'] * extrinsicVariablesDictionary['fd'] ** 1
        unifiedVariablesDictionary['p2a'] = intrinsicVariablesDictionary['p2'] * extrinsicVariablesDictionary['fd'] ** 1
        unifiedVariablesDictionary['sca'] = intrinsicVariablesDictionary['sc'] * extrinsicVariablesDictionary['fd'] ** -1
        unifiedVariablesDictionary['sra'] = intrinsicVariablesDictionary['sr'] * extrinsicVariablesDictionary['fd'] ** -1
        unifiedVariablesDictionary['oc'] = intrinsicVariablesDictionary['oc']
        unifiedVariablesDictionary['or'] = intrinsicVariablesDictionary['or']
        #
        # obtain unifiedVariables
        unifiedVariables = Dictionary2Array(unifiedVariablesKeys, unifiedVariablesDictionary)
    except:
        print('*** ExtrinsicVariablesAndIntrinsicVariables2UnifiedVariables: error obtaining unifiedVariables'); assert False
    #
    return unifiedVariables
def IntrinsicVariablesAndUnifiedVariables2ExtrinsicVariables(intrinsicVariables, unifiedVariables): # 202106141800
    #
    ''' comments:
    .- input intrinsicVariables is a 8-float-ndarray (k1, k2, p1, p2, sc, sr, oc, or)
    .- input unifiedVariables is a 14-float-ndarray (xc, yc, zc, ph, sg, ta, k1a, k2a, p1a, p2a, sca, sra, oc, or)
    .- output extrinsicVariables is a 7-float-ndarray (xc, yc, zc, ph, sg, ta, fd)
    '''
    #
    extrinsicVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'fd']
    unifiedVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']
    #
    # obtain extrinsicVariables
    try:
        # check compatibility
        if not AreIntrinsicVariablesAndUnifiedVariablesCompatible(intrinsicVariables, unifiedVariables):
            print('*** IntrinsicVariablesAndUnifiedVariables2ExtrinsicVariables: check intrinsicVariables and unifiedVariables'); assert False
        #
        # obtain unifiedVariablesDictionary
        unifiedVariablesDictionary = Array2Dictionary(unifiedVariablesKeys, unifiedVariables)
        #
        # obtain extrinsicVariablesDictionary
        extrinsicVariablesDictionary = {}
        extrinsicVariablesDictionary['xc'] = unifiedVariablesDictionary['xc']
        extrinsicVariablesDictionary['yc'] = unifiedVariablesDictionary['yc']
        extrinsicVariablesDictionary['zc'] = unifiedVariablesDictionary['zc']
        extrinsicVariablesDictionary['ph'] = unifiedVariablesDictionary['ph']
        extrinsicVariablesDictionary['sg'] = unifiedVariablesDictionary['sg']
        extrinsicVariablesDictionary['ta'] = unifiedVariablesDictionary['ta']
        extrinsicVariablesDictionary['fd'] = IntrinsicVariablesAndUnifiedVariables2Fd(intrinsicVariables, unifiedVariables)
        #
        # obtain extrinsicVariables
        extrinsicVariables = Dictionary2Array(extrinsicVariablesKeys, extrinsicVariablesDictionary)
    except:
        print('*** IntrinsicVariablesAndUnifiedVariables2ExtrinsicVariables: error obtaining extrinsicVariables'); assert False
    #
    return extrinsicVariables

#
# calibration: errors
def ErrorC(dataCdcTxt, mainSet): # 202106141800
    #
    ''' comments:
    .- input dataCdcTxt is a dictionary (including at least a 3-float-ndarray 'pc')
    .- input mainSet is a dictionary (including at least a 3-float-ndarray 'pc')
    .- output errorC is a float
    '''
    #
    # obtain errorC
    try:
        errorC = np.sqrt(np.sum((dataCdcTxt['pc'] - mainSet['pc']) ** 2))
    except:
        if not ('pc' in dataCdcTxt.keys() and 'pc' in mainSet.keys()):
            print('*** ErrorC: check keys of dataCdcTxt and mainSet ("pc" is missing)'); assert False
        else:
            print('*** ErrorC: error obtaining errorC'); assert False
    #
    return errorC
def ErrorG(dataCdgTxt, mainSet): # 202106141800
    #
    ''' comments:
    .- input dataCdgTxt is a dictionary (including at least 'xs', 'ys', 'zs', 'cs' and 'rs')
      .- dataCdgTxt['cs'] and dataCdgTxt['rs'] are distorted pixel coordinates
    .- input mainSet is a dictionary
    .- output errorG is a float
    .- uses only explicit functions
    '''
    #
    # obtain errorG
    try:
        cs, rs = XYZ2CDRD(mainSet, dataCdgTxt['xs'], dataCdgTxt['ys'], dataCdgTxt['zs'], options={})[0:2]
        errorG = np.sqrt(np.mean((dataCdgTxt['cs'] - cs) ** 2 + (dataCdgTxt['rs'] - rs) ** 2)) # RMSE
    except:
        if not (set(['xs', 'ys', 'zs', 'cs', 'rs']) <= set(dataCdgTxt.keys())):
            print('*** ErrorG: check keys of dataCdgTxt'); assert False
        elif not (all([isinstance(item, (np.ndarray)) for item in [cs, rs, dataCdgTxt['cs'], dataCdgTxt['rs']]]) and len(cs) == len(rs) == len(dataCdgTxt['cs']) == len(dataCdgTxt['rs'])):
            print('*** ErrorG: check formats and lengths of xs, ys, zs, cs and rs of dataCdgTxt'); assert False
        else:
            print('*** ErrorG: error obtaining errorG'); assert False
    #
    return errorG
def ErrorH(dataCdhTxt, horizonLine): # 202106141800
    #
    ''' comments:
    .- input dataCdhTxt is a dictionary (including at least 'chs' and 'rhs')
      .- dataCdgTxt['chs'] and dataCdgTxt['rhs'] are distorted pixel coordinates
    .- input horizonLine is a dictionary
    .- output errorH is a float
    '''
    #
    # obtain errorH
    try:
        rhs = CDh2RDh(horizonLine, dataCdhTxt['chs'], options={})[0]
        errorH = np.sqrt(np.mean((dataCdhTxt['rhs'] - rhs) ** 2)) # RMSE
    except:
        if not (set(['chs', 'rhs']) <= set(dataCdhTxt.keys())):
            print('*** ErrorH: check keys of dataCdhTxt'); assert False
        elif not (all([isinstance(item, (np.ndarray)) for item in [rhs, dataCdhTxt['rhs']]]) and len(rhs) == len(dataCdhTxt['rhs'])):
            print('*** ErrorH: check formats and lengths of chs and rhs of dataCdhTxt'); assert False
        else:
            print('*** ErrorH: error obtaining errorH'); assert False
    #
    return errorH
def ErrorT(dataForCalibration, mainSet, horizonLine, options={}): # 202106141800
    #
    ''' comments:
    .- input dataForCalibration is a dictionary (including at least 'dataCdcTxt', 'dataCdgTxt', 'dataCdwTxt')
      .- dataForCalibration optional but relevant key: 'dataCdhTxt'
      .- dataForCalibration['dataCdwTxt'] is a dictionary (including at least 'aG')
        .- if 'aC' not in dataForCalibration['dataCdwTxt'] or dataForCalibration['dataCdwTxt']['aC'] == 0., errorC is ignored
        .- if 'aH' not in dataForCalibration['dataCdwTxt'] or dataForCalibration['dataCdwTxt']['aH'] == 0., errorH is ignored
    .- input mainSet is a dictionary
    .- input horizonLine is a dictionary or None
    .- output errorT is a float
    '''
    #
    # complete options
    try:
        keys, defaultValues = ['ignoreErrorC'], [False]
        options = CompleteADictionary(options, keys, defaultValues)
        ''' comments:
        .- if options['ignoreErrorC'], then the errorC is ignored
        '''
    except:
        print('*** ErrorT: error completing options'); assert False
    #
    # obtain errorT
    try:
        # errorC
        if not options['ignoreErrorC'] and 'aC' in dataForCalibration['dataCdwTxt'] and dataForCalibration['dataCdwTxt']['aC'] > 0.: # account for errorC
            errorC = ErrorC(dataForCalibration['dataCdcTxt'], mainSet) / dataForCalibration['dataCdwTxt']['aC']
        else: # ignore errorC
            errorC = 0.
        #
        # errorG
        if not (dataForCalibration['dataCdwTxt']['aG'] > 0.):
            print('*** ErrorT: check aG of dataCdwTxt'); assert False
        errorG = ErrorG(dataForCalibration['dataCdgTxt'], mainSet) / dataForCalibration['dataCdwTxt']['aG']
        #
        # errorH
        if 'dataCdhTxt' in dataForCalibration.keys() and 'aH' in dataForCalibration['dataCdwTxt'] and dataForCalibration['dataCdwTxt']['aH'] > 0.: # account for errorH
            if horizonLine is None:
                print('*** ErrorT: check horizonLine (is None)'); assert False
            errorH = ErrorH(dataForCalibration['dataCdhTxt'], horizonLine) / dataForCalibration['dataCdwTxt']['aH']
        else: # ignore errorH
            errorH = 0.
        #
        # errorT
        errorT = errorC + errorG + errorH
    except:
        if not (set(['dataCdcTxt', 'dataCdgTxt', 'dataCdwTxt']) <= set(dataForCalibration.keys())):
            print('*** ErrorT: check keys of dataForCalibration'); assert False
        if not ('aG' in dataForCalibration['dataCdwTxt'].keys()):
            print('*** ErrorT: check keys of dataForCalibration["dataCdwTxt"]'); assert False
        else:
            print('*** ErrorT: error obtaining errorT'); assert False
    #
    return errorT
#
# calibration: intrinsic given
def RandomScaledExtrinsicVariables(dataBasic, dataForCalibration): # 202106141800
    #
    ''' comments:
    .- input dataBasic is a dictionary (including at least 'referenceValuesDictionary' and 'referenceRangesDictionary')
    .- input dataForCalibration is a dictionary (including at least 'dataCdcTxt')
    .- output scaledExtrinsicVariables is a 7-float-ndarray (xc, yc, zc, ph, sg, ta, fd) scaled
    '''
    #
    extrinsicVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'fd']
    #
    # obtain scaledExtrinsicVariables
    try:
        # obtain random extrinsicVariablesDictionary
        extrinsicVariablesDictionary = {}
        for key in extrinsicVariablesKeys:
            if key in ['xc', 'yc', 'zc']: # dataBasic['referenceValuesDictionary'][key] are None
                value0 = dataForCalibration['dataCdcTxt'][key]
            elif key in ['ph', 'sg', 'ta', 'fd']:
                value0 = dataBasic['referenceValuesDictionary'][key]
            else:
                print('*** RandomScaledExtrinsicVariables: check keys of extrinsicVariablesKeys'); assert False
            extrinsicVariablesDictionary[key] = value0 + 2.*(np.random.random()-0.5) * dataBasic['referenceRangesDictionary'][key]
        #
        extrinsicVariablesDictionary['zc'] = max(extrinsicVariablesDictionary['zc'], dataForCalibration['dataCdcTxt']['zc'] / 2.)
        #
        extrinsicVariables = Dictionary2Array(dataBasic['extrinsicVariablesKeys'], extrinsicVariablesDictionary)
        scaledExtrinsicVariables = VariablesScaling(dataBasic, extrinsicVariables, 'extrinsic', 'scale')
    except:
        print('*** RandomScaledExtrinsicVariables: error obtaining scaledExtrinsicVariables'); assert False
    #
    return scaledExtrinsicVariables
def PerturbateScaledExtrinsicVariables(dataBasic, scaledExtrinsicVariables, nc, nr, options={}): # 202106141800
    #
    ''' comments:
    .- input dataBasic is a dictionary (including at least 'referenceRangesDictionary')
    .- input scaledExtrinsicVariables is a 7-float-ndarray (xc, yc, zc, ph, sg, ta, fd) scaled
    .- input nc and nr are floats or integers
    .- output scaledExtrinsicVariables is a 7-float-ndarray (xc, yc, zc, ph, sg, ta, fd) scaled
    '''
    #
    extrinsicVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'fd']
    #
    # complete options
    try:
        keys, defaultValues = ['perturbationFactor'], [1.]
        options = CompleteADictionary(options, keys, defaultValues)
    except:
        print('*** PerturbateScaledExtrinsicVariables: error completing options')
    #
    # perturbate scaledExtrinsicVariables
    try:
        extrinsicVariables = VariablesScaling(dataBasic, scaledExtrinsicVariables, 'extrinsic', 'unscale')
        extrinsicVariablesDictionary = Array2Dictionary(extrinsicVariablesKeys, extrinsicVariables)
        for key in extrinsicVariablesKeys:
            extrinsicVariablesDictionary[key] = extrinsicVariablesDictionary[key] + options['perturbationFactor'] * 2.*(np.random.random()-0.5) * dataBasic['referenceRangesDictionary'][key]
        extrinsicVariables = Dictionary2Array(extrinsicVariablesKeys, extrinsicVariablesDictionary)
        scaledExtrinsicVariables = VariablesScaling(dataBasic, extrinsicVariables, 'extrinsic', 'scale')
    except:
        if not ('referenceRangesDictionary' in dataBasic.keys()):
            print('*** PerturbateScaledExtrinsicVariables: check keys of dataBasic'); assert False
        else:
            print('*** PerturbateScaledExtrinsicVariables: error perturbating scaledExtrinsicVariables'); assert False
    #
    return scaledExtrinsicVariables
def ScaledExtrinsicVariables2MainSet(dataBasic, dataForCalibration, scaledExtrinsicVariables): # 202106141800
    #
    ''' comments:
    .- input dataBasic is a dictionary
    .- input dataForCalibration is a dictionary (including at least 'dataIntMat', 'nc' and 'nr')
    .- input scaledExtrinsicVariables is a 7-float-ndarray
    .- output mainSet is a dictionary
    '''
    #
    # obtain mainSet
    try:
        extrinsicVariables = VariablesScaling(dataBasic, scaledExtrinsicVariables, 'extrinsic', 'unscale')
        unifiedVariables = ExtrinsicVariablesAndIntrinsicVariables2UnifiedVariables(extrinsicVariables, dataForCalibration['dataIntMat']['intrinsicVariables'])
        mainSet = UnifiedVariables2MainSet(dataForCalibration['nc'], dataForCalibration['nr'], unifiedVariables)
    except:
        if not (set(['dataIntMat', 'nc', 'nr']) <= set(dataForCalibration.keys()) and 'intrinsicVariables' in dataForCalibration['dataIntMat'].keys()):
            print('*** ScaledExtrinsicVariables2MainSet: check keys of dataForCalibration'); assert False
        elif not (isinstance(scaledExtrinsicVariables, (np.ndarray)) and len(scaledExtrinsicVariables == 7)):
            print('*** ScaledExtrinsicVariables2MainSet: check format and length of scaledExtrinsicVariables'); assert False
        else:
            print('*** ScaledExtrinsicVariables2MainSet: error obtaining mainSet'); assert False
    #
    return mainSet
def ScaledExtrinsicVariables2ErrorT(dataBasic, dataForCalibration, scaledExtrinsicVariables, options={}): # 202106141800
    #
    ''' comments:
    .- input dataBasic is a dictionary
    .- input dataForCalibration is a dictionary
    .- input scaledExtrinsicVariables is a 7-float-ndarray
    .- output errorT is a dictionary
    '''
    #
    # complete options
    try:
        keys, defaultValues = ['ignoreErrorC'], [False]
        options = CompleteADictionary(options, keys, defaultValues)
    except:
        print('*** ScaledExtrinsicVariables2ErrorT: error completing options'); assert False
    #
    # obtain errorT
    try:
        mainSet = ScaledExtrinsicVariables2MainSet(dataBasic, dataForCalibration, scaledExtrinsicVariables)
        if 'dataCdhTxt' in dataForCalibration.keys() and 'aH' in dataForCalibration['dataCdwTxt'].keys() and dataForCalibration['dataCdwTxt']['aH'] > 0.:
            if not ('z0' in dataForCalibration.keys()):
                print('*** ScaledExtrinsicVariables2ErrorT: check keys of dataForCalibration ("z0")'); assert False
            horizonLine = MainSetAndZ02HorizonLine(dataBasic, mainSet, dataForCalibration['z0'])
        else:
            horizonLine = None
        errorT = ErrorT(dataForCalibration, mainSet, horizonLine, {'ignoreErrorC':options['ignoreErrorC']})
    except:
        print('*** ScaledExtrinsicVariables2ErrorT: error obtaining errorT'); assert False
    #
    return errorT
#
# calibration: intrinsic not given
def RandomScaledSelectedUnifiedVariables(dataBasic, dataForCalibration): # 202106141800
    #
    ''' comments:
    .- input dataBasic is a dictionary (including at least 'referenceValuesDictionary' and 'referenceRangesDictionary')
    .- input dataForCalibration is a dictionary (including at least 'dataCdcTxt', 'nc' and 'nr')
    .- output scaledSelectedUnifiedVariables is a float-ndarray, a subset of (xc, yc, zc, ph, sg, ta, k1a, k2a, p1a, p2a, sca, sra, oc, or) scaled
    '''
    #
    unifiedVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']
    #
    # obtain scaledSelectedUnifiedVariables
    try:
        # obtain random unifiedVariablesDictionary
        unifiedVariablesDictionary = {}
        for key in unifiedVariablesKeys:
            # load value0 from referenceValuesDictionary
            if key in ['xc', 'yc', 'zc']: # dataBasic['referenceValuesDictionary'][key] are None
                value0 = dataForCalibration['dataCdcTxt'][key]
            elif key in ['ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra']:
                value0 = dataBasic['referenceValuesDictionary'][key]
            elif key in ['oc', 'or']: # dataBasic['referenceValuesDictionary'][key] are None
                value0 = N2K(dataForCalibration['n' + key[-1]])
            else:
                print('*** RandomScaledSelectedUnifiedVariables: check keys of unifiedVariablesKeys'); assert False
            # perturbate with referenceRangesDictionary
            unifiedVariablesDictionary[key] = value0 + 2.*(np.random.random()-0.5) * dataBasic['referenceRangesDictionary'][key]
        #
        unifiedVariablesDictionary['zc'] = max(unifiedVariablesDictionary['zc'], dataForCalibration['dataCdcTxt']['zc'] / 2.)
        #
        # obtain scaledSelectedUnifiedVariables
        unifiedVariables = Dictionary2Array(unifiedVariablesKeys, unifiedVariablesDictionary)
        selectedUnifiedVariables = UnifiedVariables2SelectedUnifiedVariables(dataBasic, unifiedVariables)
        scaledSelectedUnifiedVariables = VariablesScaling(dataBasic, selectedUnifiedVariables, 'selectedUnified', 'scale')
    except:
        print('*** RandomScaledSelectedUnifiedVariables: error obtaining scaledSelectedUnifiedVariables'); assert False
    #
    return scaledSelectedUnifiedVariables
def PerturbateScaledSelectedUnifiedVariables(dataBasic, scaledSelectedUnifiedVariables, nc, nr, options={}): # 202106141800
    #
    ''' comments:
    .- input dataBasic is a dictionary (including at least 'referenceRangesDictionary')
    .- input scaledSelectedUnifiedVariables is a float-ndarray, subset of (xc, yc, zc, ph, sg, ta, k1a, k2a, p1a, p2a, sca, sra, oc, or) scaled
    .- input nc and nr are floats or integers
    .- output scaledSelectedUnifiedVariables is a float-ndarray, a subset of (xc, yc, zc, ph, sg, ta, k1a, k2a, p1a, p2a, sca, sra, oc, or) scaled
    '''
    #
    unifiedVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']
    #
    # complete options
    try:
        keys, defaultValues = ['perturbationFactor'], [1.]
        options = CompleteADictionary(options, keys, defaultValues)
    except:
        print('*** PerturbateScaledSelectedUnifiedVariables: error completing options')
    #
    # perturbate scaledSelectedUnifiedVariables
    try:
        selectedUnifiedVariables = VariablesScaling(dataBasic, scaledSelectedUnifiedVariables, 'selectedUnified', 'unscale')
        unifiedVariables = SelectedUnifiedVariables2UnifiedVariables(dataBasic, selectedUnifiedVariables, nc, nr)
        unifiedVariablesDictionary = Array2Dictionary(unifiedVariablesKeys, unifiedVariables)
        for key in unifiedVariablesKeys:
            unifiedVariablesDictionary[key] = unifiedVariablesDictionary[key] + options['perturbationFactor'] * 2.*(np.random.random()-0.5) * dataBasic['referenceRangesDictionary'][key]
        unifiedVariables = Dictionary2Array(unifiedVariablesKeys, unifiedVariablesDictionary)
        selectedUnifiedVariables = UnifiedVariables2SelectedUnifiedVariables(dataBasic, unifiedVariables)
        scaledSelectedUnifiedVariables = VariablesScaling(dataBasic, selectedUnifiedVariables, 'selectedUnified', 'scale')
    except:
        if not ('referenceRangesDictionary' in dataBasic.keys()):
            print('*** PerturbateScaledSelectedUnifiedVariables: check keys of dataBasic'); assert False
        else:
            print('*** PerturbateScaledSelectedUnifiedVariables: error perturbating scaledSelectedUnifiedVariables'); assert False
    #
    return scaledSelectedUnifiedVariables
def ScaledSelectedUnifiedVariables2MainSet(dataBasic, dataForCalibration, scaledSelectedUnifiedVariables): # 202106141800
    #
    ''' comments:
    .- input dataBasic is a dictionary
    .- input dataForCalibration is a dictionary (including at least 'nc' and 'nr')
    .- input scaledSelectedUnifiedVariables is a float-ndarray
    .- output mainSet is a dictionary
    '''
    #
    # obtain mainSet
    try:
        selectedUnifiedVariables = VariablesScaling(dataBasic, scaledSelectedUnifiedVariables, 'selectedUnified', 'unscale')
        unifiedVariables = SelectedUnifiedVariables2UnifiedVariables(dataBasic, selectedUnifiedVariables, dataForCalibration['nc'], dataForCalibration['nr'])
        mainSet = UnifiedVariables2MainSet(dataForCalibration['nc'], dataForCalibration['nr'], unifiedVariables)
    except:
        if not (set(['nc', 'nr']) <= set(dataForCalibration.keys())):
            print('*** ScaledSelectedUnifiedVariables2MainSet: check keys of dataForCalibration'); assert False
        else:
            print('*** ScaledSelectedUnifiedVariables2MainSet: error obtaining mainSet'); assert False
    #
    return mainSet
def ScaledSelectedUnifiedVariables2ErrorT(dataBasic, dataForCalibration, scaledSelectedUnifiedVariables, options={}): # 202106141800
    #
    ''' comments:
    .- input dataBasic is a dictionary
    .- input dataForCalibration is a dictionary
    .- input scaledSelectedUnifiedVariables is a float-ndarray
    .- output errorT is a float
    '''
    #
    # complete options
    try:
        keys, defaultValues = ['ignoreErrorC'], [False]
        options = CompleteADictionary(options, keys, defaultValues)
    except:
        print('*** ScaledSelectedUnifiedVariables2ErrorT: error completing options'); assert False
    #
    # obtain errorT
    try:
        mainSet = ScaledSelectedUnifiedVariables2MainSet(dataBasic, dataForCalibration, scaledSelectedUnifiedVariables)
        if 'dataCdhTxt' in dataForCalibration.keys() and 'aH' in dataForCalibration['dataCdwTxt'].keys() and dataForCalibration['dataCdwTxt']['aH'] > 0.: # horizonLine is used
            if not ('z0' in dataForCalibration.keys()):
                print('*** ScaledSelectedUnifiedVariables2ErrorT: check keys of dataForCalibration ("z0")'); assert False
            horizonLine = MainSetAndZ02HorizonLine(dataBasic, mainSet, dataForCalibration['z0'])
        else:
            horizonLine = None
        errorT = ErrorT(dataForCalibration, mainSet, horizonLine, {'ignoreErrorC':options['ignoreErrorC']})
    except:
        print('*** ScaledSelectedUnifiedVariables2ErrorT: error obtaining errorT'); assert False
    #
    return errorT
#
# calibration: nonlinear manual calibration
def NonlinearManualCalibration(dataBasic, dataForCalibration, options={}): # 202106141800
    #
    def IsUnifiedVariablesDictionaryOK(unifiedVariablesDictionary): # 202106141800
        #
        ''' comments:
        .- input unifiedVariablesDictionary (including at least 'zc', 'ta', 'sca', 'sra' and 'sg')
        .- output conditions is a boolean
        '''
        # obtain isUnifiedVariablesDictionaryOK
        try:
            isUnifiedVariablesDictionaryOK = all([unifiedVariablesDictionary[item] > 0 for item in ['zc', 'ta', 'sca', 'sra']]) and np.abs(unifiedVariablesDictionary['sg']) < np.pi/2.
        except:
            if not (set(['zc', 'ta', 'sca', 'sra', 'sg']) <= set(unifiedVariablesDictionary.keys())):
                print('*** NonlinearManualCalibration; IsUnifiedVariablesDictionaryOK: check keys of unifiedVariablesDictionary'); assert False
            else:
                print('*** NonlinearManualCalibration; IsUnifiedVariablesDictionaryOK: error obtaining isUnifiedVariablesDictionaryOK'); assert False
        #
        return isUnifiedVariablesDictionaryOK
    def FunctionToMinimize(scaled_Variables): # 202106141800
        #
        ''' comments:
        .- input scaled_Variables is a float-ndarray (scaledExtrinsicVariables or scaledSelectedUnifiedVariables)
        .- output errorT is a float
        .- uses global variables (e.g., dataBasic and dataForCalibration)
          .- dataForCalibration is a dictionary (including at least dataForCalibration['dataCdwTxt']['ignoreErrorC'])
        '''
        # obtain errorT
        try:
            errorT = Scaled_Variables2ErrorT(dataBasic, dataForCalibration, scaled_Variables, {'ignoreErrorC':dataForCalibration['dataCdwTxt']['ignoreErrorC']})
        except:
            if not ('dataCdwTxt' in dataForCalibration.keys() and 'ignoreErrorC' in dataForCalibration['dataCdwTxt']):
                print('*** NonlinearManualCalibration; FunctionToMinimize: check keys of dataForCalibration'); assert False
            else:
                print('*** NonlinearManualCalibration; FunctionToMinimize: error obtaining errorT'); assert False
        #
        return errorT
    def UnifiedVariables2Scaled_Variables(unifiedVariables): # 202106141800
        #
        ''' comments:
        .- input unifiedVariables is a 14-float-ndarray
        .- output scaled_Variables is a float-ndarray
        .- uses global variables (e.g., dataBasic and dataForCalibration)
          .- if dataForCalibration includes 'dataIntMat', then the given intrinsic calibration is automatically used
        '''
        # obtain scaled_Variables
        try:
            if 'dataIntMat' in dataForCalibration.keys():
                extrinsicVariables = IntrinsicVariablesAndUnifiedVariables2ExtrinsicVariables(dataForCalibration['dataIntMat']['intrinsicVariables'], unifiedVariables)
                scaled_Variables = VariablesScaling(dataBasic, extrinsicVariables, 'extrinsic', 'scale')
            else:
                selectedUnifiedVariables = UnifiedVariables2SelectedUnifiedVariables(dataBasic, unifiedVariables)
                scaled_Variables = VariablesScaling(dataBasic, selectedUnifiedVariables, 'selectedUnified', 'scale')
        except:
            if 'dataIntMat' in dataForCalibration and 'intrinsicVariables' not in dataForCalibration['dataIntMat'].keys():
                print('*** UnifiedVariables2Scaled_Variables: check keys of dataForCalibration["dataIntMat"]'); assert False
            else:
                print('*** UnifiedVariables2Scaled_Variables: error obtaining scaled_Variables'); assert False
        #
        return scaled_Variables
    def MainSet2ErrorT(mainSet): # 202106141800
        #
        ''' comments:
        .- input mainSet is a dictionary (including at least 'unifiedVariables')
        .- output errorT is a float
        '''
        # obtain errorT
        try:
            scaled_Variables = UnifiedVariables2Scaled_Variables(mainSet['unifiedVariables'])
            errorT = FunctionToMinimize(scaled_Variables)
        except:
            if not ('unifiedVariables' in mainSet.keys()):
                print('*** NonlinearManualCalibration; MainSet2ErrorT: check keys of mainSet ("unifiedVariables")'); assert False
            else:
                print('*** NonlinearManualCalibration; MainSet2ErrorT: error obtaining errorT'); assert False
        #
        return errorT
    def ReadAFirstSeed(): # 202106141800
        #
        ''' comments:
        .- output mainSetSeed is a dictionary or None (if there is no dataForCalibration['dataCalTxt']['mainSet'] and no options['mainSetSeed'])
        .- output errorTSeed is a float or None (if there is no dataForCalibration['dataCalTxt']['mainSet'] and no options['mainSetSeed'])
        .- uses global variables (e.g., dataForCalibration and options)
        '''
        # obtain mainSetSeed and errorTSeed
        try:
            # initialize mainSetSeed and errorTSeed
            mainSetSeed, errorTSeed = None, None
            # obtain mainSetSeed and errorTSeed if dataForCalibration['dataCalTxt']['mainSet'] exists and it is not None
            if 'dataCalTxt' in dataForCalibration.keys() and 'mainSet' in dataForCalibration['dataCalTxt'].keys() and dataForCalibration['dataCalTxt']['mainSet'] is not None:
                mainSetSeed = dataForCalibration['dataCalTxt']['mainSet']
                errorTSeed = MainSet2ErrorT(mainSetSeed) # IMP*
            # update mainSetSeed and errorTSeed if options['mainSetSeed'] is not None
            if 'mainSetSeed' in options.keys() and options['mainSetSeed'] is not None:
                mainSetTMP = options['mainSetSeed']
                errorTTMP = MainSet2ErrorT(mainSetTMP) # IMP*
                if errorTSeed is None or errorTTMP < errorTSeed: # update
                    mainSetSeed, errorTSeed = mainSetTMP, errorTTMP
        except:
            print('*** NonlinearManualCalibration; ReadAFirstSeed: error obtaining mainSetSeed and errorTSeed'); assert False
        #
        return mainSetSeed, errorTSeed
    def FindAFirstSeed(): # 202106141800
        #
        ''' comments
        .- output mainSetSeed is a dictionary or None (if it does not succeed)
        .- output errorTSeed is a float or None (if it does not succeed)
        .- uses global variables (e.g., dataBasic and dataForCalibration)
        '''
        # obtain mainSetSeed and errorTSeed
        try:
            seedFound, time0 = False, datetime.datetime.now()
            while (not seedFound and datetime.datetime.now() - time0 < dataForCalibration['dataCdwTxt']['firstSeedTimedelta']):
                scaled_Variables = RandomScaled_Variables(dataBasic, dataForCalibration)
                try: # IMPORTANT to try
                    errorT = FunctionToMinimize(scaled_Variables)
                except: # IMPORTANT not to inform
                    continue
                #
                if errorT < dataForCalibration['dataCdwTxt']['firstSeedErrorT']:
                    # check
                    if not IsUnifiedVariablesDictionaryOK(Scaled_Variables2MainSet(dataBasic, dataForCalibration, scaled_Variables)['unifiedVariablesDictionary']):
                        continue
                    # refine the seed
                    try:
                        scaled_VariablesSeed = optimize.minimize(FunctionToMinimize, scaled_Variables).x
                        mainSetSeed = Scaled_Variables2MainSet(dataBasic, dataForCalibration, scaled_VariablesSeed)
                        errorTSeed = MainSet2ErrorT(mainSetSeed)
                        seedFound = IsUnifiedVariablesDictionaryOK(mainSetSeed['unifiedVariablesDictionary'])
                    except:
                        print('*** NonlinearManualCalibration; FindAFirstSeed: error refining the seed'); pass
            if not seedFound: # failure
                mainSetSeed, errorTSeed = None, None
        except:
            if not ('dataCdwTxt' in dataForCalibration and set(['firstSeedTimedelta', 'firstSeedErrorT']) <= set(dataForCalibration['dataCdwTxt'])):
                print('*** NonlinearManualCalibration; FindAFirstSeed: check keys of dataForCalibration'); assert False
            else:
                print('*** NonlinearManualCalibration; FindAFirstSeed: error obtaining mainSetSeed and errorTSeed'); assert False
        #
        return mainSetSeed, errorTSeed
    #
    ''' comments:
    .- input dataBasic is a dictionary
    .- input dataForCalibration is a dictionary (including at least 'nc', 'nr', 'dataCdcTxt', 'dataCdgTxt' and 'dataCdwTxt')
      .- dataForCalibration optional but relevant keys: 'dataCdhTxt', 'z0'
        .- if 'dataCdhTxt' is in dataForCalibration.keys(), then 'z0' must be in dataForCalibration.keys()
      .- dataForCalibration optional but relevant key: 'dataIntMat'
        .- if 'dataIntMat' in dataForCalibration.keys(), it is automatically used
    .- output mainSet is a dictionary or None (if it does not succeed)
    .- output errorT is a float or None (if it does not succeed)
    .- uses functions for scaled_Variables ('_' = 'Extrinsic' or 'SelectedUnified'):
      .- RandomScaled_Variables = f(dataBasic, dataForCalibration)
      .- PerturbateScaled_Variables = f(dataBasic, scaled_Variables, nc, nr, options={'perturbationFactor'})
      .- Scaled_Variables2MainSet = f(dataBasic, dataForCalibration, scaled_Variables)
      .- Scaled_Variables2ErrorT = f(dataBasic, dataForCalibration, scaled_Variables, options={'ignoreErrorC'})
    .- uses all explicit functions since the error is evaluated in pixels (XYZ2CDRD)
    '''
    #
    # complete options
    try:
        keys, defaultValues = ['mainSetSeed', 'minimumNOfGCPs'], [None, 5]
        options = CompleteADictionary(options, keys, defaultValues)
        ''' comments:
        .- uses the best between options['mainSetSeed'] and dataForCalibration['dataCalTxt']['mainSet']
        '''
    except:
        print('*** NonlinearManualCalibration: error completing options'); assert False
    #
    # check
    if 'dataCdhTxt' in dataForCalibration.keys() and (not 'z0' in dataForCalibration.keys() or dataForCalibration['z0'] is None):
        print('*** NonlinearManualCalibration: check keys of dataForCalibration ("z0" is missing or None)'); assert False
    elif 'dataIntMat' in dataForCalibration.keys() and (not 'intrinsicVariables' in dataForCalibration['dataIntMat'] or dataForCalibration['dataIntMat']['intrinsicVariables'] is None):
        print('*** NonlinearManualCalibration: check keys of dataForCalibration["dataIntMat"] ("intrinsicVariables" is missing or None)'); assert False
    #
    # manage options['minimumNOfGCPs']
    try:
        if len(dataForCalibration['dataCdgTxt']['xs']) < options['minimumNOfGCPs']:
            return None, None
    except:
        if not ('dataCdgTxt' in dataForCalibration.keys() and 'xs' in dataForCalibration['dataCdgTxt'].keys()):
            print('*** NonlinearManualCalibration: check keys of dataForCalibration'); assert False
    #
    # define the functions depending on whether intrinsic calibration is used or not
    if 'dataIntMat' in dataForCalibration.keys(): # use intrinsic ("_" = "Extrinsic")
        RandomScaled_Variables = RandomScaledExtrinsicVariables
        PerturbateScaled_Variables = PerturbateScaledExtrinsicVariables
        Scaled_Variables2MainSet = ScaledExtrinsicVariables2MainSet
        Scaled_Variables2ErrorT = ScaledExtrinsicVariables2ErrorT
    else: # do not use intrinsic ("_" = "SelectedUnified")
        RandomScaled_Variables = RandomScaledSelectedUnifiedVariables
        PerturbateScaled_Variables = PerturbateScaledSelectedUnifiedVariables
        Scaled_Variables2MainSet = ScaledSelectedUnifiedVariables2MainSet
        Scaled_Variables2ErrorT = ScaledSelectedUnifiedVariables2ErrorT
    #
    # obtain mainSetSeed, errorTSeed and scaled_VariablesSeed
    try:
        mainSetSeed, errorTSeed = ReadAFirstSeed()
        if mainSetSeed is not None and errorTSeed <= dataForCalibration['dataCdwTxt']['firstSeedErrorT']:
            #print('... NonlinearManualCalibration: seed available with errorT = {:.3f}'.format(errorTSeed))
            pass
        else:
            #print('... NonlinearManualCalibration: looking for a first seed')
            ignoreErrorCTMP = dataForCalibration['dataCdwTxt']['ignoreErrorC']
            dataForCalibration['dataCdwTxt']['ignoreErrorC'] = True
            mainSetSeed, errorTSeed = FindAFirstSeed()
            dataForCalibration['dataCdwTxt']['ignoreErrorC'] = ignoreErrorCTMP
            if mainSetSeed is None:
                print('*** NonlinearManualCalibration: no first seed found'); assert False
        scaled_VariablesSeed = UnifiedVariables2Scaled_Variables(mainSetSeed['unifiedVariables'])
        mainSet0, errorT0 = mainSetSeed, errorTSeed # for checkings
    except:
        if not ('dataCdwTxt' in dataForCalibration.keys() and set(['firstSeedErrorT', 'ignoreErrorC']) <= set(dataForCalibration['dataCdwTxt'].keys())):
            print('*** NonlinearManualCalibration: check keys of dataForCalibration ("dataCdwTxt")'); assert False
        else:
            print('*** NonlinearManualCalibration: error obtaining mainSetSeed, errorTSeed and scaled_VariablesSeed'); assert False
    #
    # obtain nOfSeeds and perturbationFactor for MonteCarlo
    try:
        if errorTSeed > dataForCalibration['dataCdwTxt']['MonteCarloErrorT']:
            nOfSeeds, perturbationFactor = dataForCalibration['dataCdwTxt']['MonteCarloNOfSeeds'], dataForCalibration['dataCdwTxt']['MonteCarloPerturbationFactor']
        else:
            nOfSeeds, perturbationFactor = int(np.ceil(0.2 * dataForCalibration['dataCdwTxt']['MonteCarloNOfSeeds'])), 0.2 * dataForCalibration['dataCdwTxt']['MonteCarloPerturbationFactor']
    except:
        if not ('dataCdwTxt' in dataForCalibration.keys() and set(['MonteCarloErrorT', 'MonteCarloNOfSeeds', 'MonteCarloPerturbationFactor']) <= set(dataForCalibration['dataCdwTxt'].keys())):
            print('*** NonlinearManualCalibration: check keys of dataForCalibration ("dataCdwTxt")'); assert False
        else:
            print('*** NonlinearManualCalibration: error obtaining MonteCarloNOfSeeds and perturbationFactor'); assert False
    #
    # apply MonteCarlo
    try:
        counter, scaled_VariablesO, errorTO = 0, scaled_VariablesSeed, errorTSeed
        while counter < nOfSeeds:
            optionsTMP = {'perturbationFactor':perturbationFactor}
            scaled_VariablesP = PerturbateScaled_Variables(dataBasic, scaled_VariablesO, dataForCalibration['nc'], dataForCalibration['nr'], optionsTMP)
            errorTP = FunctionToMinimize(scaled_VariablesP)
            if errorTP > dataForCalibration['dataCdwTxt']['firstSeedErrorT']:
                continue
            scaled_VariablesL = optimize.minimize(FunctionToMinimize, scaled_VariablesP).x
            errorTL = FunctionToMinimize(scaled_VariablesL)
            if errorTL < errorTO:
                mainSetL = Scaled_Variables2MainSet(dataBasic, dataForCalibration, scaled_VariablesL)
                # control: initialize isGood
                isGood = True
                # control: intrinsic compatibilities
                if isGood and 'dataIntMat' in dataForCalibration.keys():
                    intrinsicVariables, unifiedVariables = dataForCalibration['dataIntMat']['intrinsicVariables'], mainSetL['unifiedVariables']
                    isGood = AreIntrinsicVariablesAndUnifiedVariablesCompatible(intrinsicVariables, unifiedVariables)
                # control: isUnifiedVariablesDictionaryOK
                if isGood:
                    isGood = IsUnifiedVariablesDictionaryOK(mainSetL['unifiedVariablesDictionary'])
                # control: right side of the camera WATCH OUT
                if isGood:
                    xs, ys, zs = dataForCalibration['dataCdgTxt']['xs'], dataForCalibration['dataCdgTxt']['ys'], dataForCalibration['dataCdgTxt']['zs']
                    isGood = len(XYZ2PositionsRightSideOfCamera(mainSetL, xs, ys, zs)) == len(xs) == len(ys) == len(zs)
                # update
                if isGood:
                    #print('  ... improvement in iteration {:3}, errorT = {:9.3f}'.format(counter+1, errorTO))
                    scaled_VariablesO, errorTO = copy.deepcopy(scaled_VariablesL), copy.deepcopy(errorTL)
            counter = counter + 1
        #
        # obtain scaled_Variables, errorT and mainSet
        scaled_Variables, errorT = scaled_VariablesO, errorTO
        mainSet = Scaled_Variables2MainSet(dataBasic, dataForCalibration, scaled_Variables)
        #
        # check whether we improve what we already had (IMPORTANT)
        if errorT0 <= errorT * (1. + 1.e-8):
            #print('  ... the provided calibration or initial seed was not significantly improved')
            mainSet, errorT = mainSet0, errorT0
    except:
        if not (set(['nc', 'nr', 'dataCdgTxt', 'dataCdwTxt']) <= set(dataForCalibration.keys())):
            print('*** NonlinearManualCalibration: check keys of dataForCalibration'); assert False
        elif not (set(['xs', 'ys', 'zs']) <= set(dataForCalibration['dataCdgTxt'])):
            print('*** NonlinearManualCalibration: check keys of dataForCalibration["dataCdgTxt"]'); assert False
        elif not ('firstSeedErrorT' in dataForCalibration['dataCdwTxt'].keys()):
            print('*** NonlinearManualCalibration: check keys of dataForCalibration["dataCdwTxt"]'); assert False
        elif 'dataIntMat' in dataForCalibration.keys() and not ('intrinsicVariables' in dataForCalibration['dataIntMat']):
            print('*** NonlinearManualCalibration: check keys of dataForCalibration["dataIntMat"]'); assert False
        elif not ('unifiedVariables' in mainSetL.keys()):
            print('*** NonlinearManualCalibration: check keys of mainSetL'); assert False
        else:
            print('*** NonlinearManualCalibration: error applying MonteCarlo'); assert False
    #
    return mainSet, errorT
def NonlinearManualCalibrationFromGCPs(xs, ys, zs, cs, rs, nc, nr, options={}): # 202106141800
    #
    ''' comments:
    .- input xs, ys, zs, cs and rs are float-ndarrays of the same length (GCPs)
    .- input nc and nr are integers or floats
    '''
    #
    # complete options
    try:
        keys, defaultValues = ['mainSetSeed', 'chs', 'rhs', 'z0', 'xc', 'yc', 'zc'], None
        options = CompleteADictionary(options, keys, defaultValues)
        keys, defaultValues = ['selectedUnifiedVariablesKeys'], [['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'sca']]
        options = CompleteADictionary(options, keys, defaultValues)
        keys, defaultValues = ['firstSeedErrorT', 'firstSeedTimedeltaInSeconds', 'MonteCarloErrorT', 'MonteCarloNOfSeeds', 'MonteCarloPerturbationFactor'], [500., 30, 10., 100, 1.]
        options = CompleteADictionary(options, keys, defaultValues)
    except:
        print('*** NonlinearManualCalibrationFromGCPs: error completing options'); assert False
    #
    # obtain dataBasic
    try:
        dataBasic = LoadDataBasic0({'selectedUnifiedVariablesKeys':options['selectedUnifiedVariablesKeys']})
    except:
        print('*** NonlinearManualCalibrationFromGCPs: error obtaining dataBasic'); assert False
    #
    # obtain dataForCalibration
    try:
        # initialize dataForCalibration
        dataForCalibration = {'nc':nc, 'nr':nr}
        #
        # obtain and load initial dataCdwTxt (ignoreErrorC is set True)
        dataCdwTxt = {item:options[item] for item in ['firstSeedErrorT', 'firstSeedTimedeltaInSeconds', 'MonteCarloErrorT', 'MonteCarloNOfSeeds', 'MonteCarloPerturbationFactor']}
        dataCdwTxt['aG'], dataCdwTxt['firstSeedTimedelta'], dataCdwTxt['ignoreErrorC'] = 1., datetime.timedelta(seconds = dataCdwTxt['firstSeedTimedeltaInSeconds']), True
        dataForCalibration['dataCdwTxt'] = dataCdwTxt
        #
        # obtain and load dataCdcTxt and update dataBasic (this is only to find the first seed)
        if any([options[item] is None for item in ['xc', 'yc', 'zc']]):
            dataCdcTxt = {'xc':np.mean(xs), 'yc':np.mean(ys), 'zc':np.mean(zs)+10.}
            dataBasic['referenceRangesDictionary']['xc'] = 5. * np.std(xs)
            dataBasic['referenceRangesDictionary']['yc'] = 5. * np.std(ys)
            dataBasic['referenceRangesDictionary']['zc'] = 200.
        else:
            dataCdcTxt = {'xc':options['xc'], 'yc':options['yc'], 'zc':options['zc']}
        dataCdcTxt['pc'] = np.asarray([dataCdcTxt['xc'], dataCdcTxt['yc'], dataCdcTxt['zc']])
        dataForCalibration['dataCdcTxt'] = dataCdcTxt
        #
        # obtain and load dataCdgTxt
        dataCdgTxt = {'xs':xs, 'ys':ys, 'zs':zs, 'cs':cs, 'rs':rs, 'camera':'-', 'date':'-'}
        dataForCalibration['dataCdgTxt'] = dataCdgTxt
        #
        # obtain and load dataCdhTxt and update dataCdwTxt
        if all([options[item] is not None for item in ['chs', 'rhs']]):
            dataCdhTxt = {item:options[item] for item in ['chs', 'rhs']}
            dataForCalibration['dataCdhTxt'] = dataCdhTxt
            dataForCalibration['z0'] = options['z0']
            dataCdwTxt['aH'] = 1.
            dataForCalibration['dataCdwTxt'] = dataCdwTxt
    except:
        print('*** NonlinearManualCalibrationFromGCPs: error obtaining dataForCalibration'); assert False
    #
    if options['mainSetSeed'] is not None:
        optionsTMP = {item:options[item] for item in ['mainSetSeed']}
    else:
        optionsTMP = {}
    mainSet, errorT = NonlinearManualCalibration(dataBasic, dataForCalibration, optionsTMP)
    #
    return mainSet, errorT
#
''' -------------------------------------------------------------------------------------- '''
''' --- specific UCalib functions -------------------------------------------------------- '''
''' -------------------------------------------------------------------------------------- '''
# 
def p190227ErrorsG(xs, ys, zs, cs, rs, mainSet): # 202106141800
    #
    ''' comments:
    .- input xs, ys, zs, cs and rs are float-ndarrays of the same non-null length
    .- input mainSet is a dictionary
    .- output errorsG is a float-ndarray of the same length as xs
    '''
    #
    # obtain errorsG
    try:
        csR, rsR = XYZ2CDRD(mainSet, xs, ys, zs, options={})[0:2]
        errorsG = np.sqrt((csR - cs) ** 2 + (rsR - rs) ** 2)
    except:
        print('*** p190227ErrorsG: error obtaining errorsG'); assert False
    #
    return errorsG
def p190227ErrorG(xs, ys, zs, cs, rs, mainSet): # 202106141800
    #
    ''' comments:
    .- input xs, ys, zs, cs and rs are float-ndarrays of the same non-null length
    .- input mainSet is a dictionary
    .- output errorG is a float
    '''
    #
    # obtain errorG
    try:
        errorsG = p190227ErrorsG(xs, ys, zs, cs, rs, mainSet)
        errorG = np.sqrt(np.mean(errorsG ** 2))
    except:
        print('*** p190227ErrorG: error obtaining errorG'); assert False
    #
    return errorG
def p190227ErrorsH(chs, rhs, horizonLine): # 202106141800
    #
    ''' comments:
    .- input chs and rhs are float-ndarrays of the same length
    .- input horizonLine is a dictionary
    .- output errorsH is a float-ndarray of the same length as chs
    '''
    #
    # obtain errorsH
    try:
        if len(chs) == 0:
            errorsH = np.asarray([])
        else:
            rhsR = CDh2RDh(horizonLine, chs, options={})[0]
            errorsH = np.sqrt((rhsR - rhs) ** 2)
    except:
        print('*** p190227ErrorsH: error obtaining errorsH'); assert False
    #
    return errorsH
def p190227ErrorH(chs, rhs, horizonLine): # 202106141800
    #
    ''' comments:
    .- input chs and rhs are float-ndarrays of the same length
    .- input horizonLine is a dictionary
    .- output errorH is a float
    '''
    #
    # obtain errorH
    try:
        if len(chs) == 0:
            errorH = 0.
        else:
            errorsH = p190227ErrorsH(chs, rhs, horizonLine)
            errorH = np.sqrt(np.mean(errorsH ** 2))
    except:
        print('*** p190227ErrorH: error obtaining errorH'); assert False
    #
    return errorH

def p190227FunctionToMinimize0(scaledVariables0, theArgs): # 202106141800
    #
    ''' comments:
    .- input scaledVariables0 is a 3-float-ndarray (the angles ph, sg and ta)
    .- input theArgs is a dictionary
    .- output errorT is a float
    '''
    #
    # read theArgs
    try:
        dataBasic, scaledVariables1 = theArgs['dataBasic'], theArgs['scaledVariables1']
        nc, nr, kc, kr = theArgs['nc'], theArgs['nr'], theArgs['kc'], theArgs['kr']
        xs, ys, zs, cs, rs = [theArgs[item] for item in ['xs', 'ys', 'zs', 'cs', 'rs']]
        thereIsH, chs, rhs, z0 = [theArgs[item] for item in ['thereIsH', 'chs', 'rhs', 'z0']]
    except:
        print('*** p190227FunctionToMinimize0: error reading theArgs'); assert False
    #
    # obtain errorT
    try:
        scaledUnifiedVariables = np.zeros(14)
        scaledUnifiedVariables[0] = scaledVariables1[0] # xc - constant
        scaledUnifiedVariables[1] = scaledVariables1[1] # yc - constant
        scaledUnifiedVariables[2] = scaledVariables1[2] # zc - constant
        scaledUnifiedVariables[3] = scaledVariables0[0] # ph
        scaledUnifiedVariables[4] = scaledVariables0[1] # sg
        scaledUnifiedVariables[5] = scaledVariables0[2] # ta
        scaledUnifiedVariables[6] = scaledVariables1[3] # k1a - constant
        scaledUnifiedVariables[7] = 0. # k2a
        scaledUnifiedVariables[8] = 0. # p1a
        scaledUnifiedVariables[9] = 0. # p2a
        scaledUnifiedVariables[10] = scaledVariables1[4] # sca - constant
        scaledUnifiedVariables[11] = scaledVariables1[4] # sra - constant
        scaledUnifiedVariables[12] = kc / dataBasic['scalesDictionary']['oc'] # oca
        scaledUnifiedVariables[13] = kr / dataBasic['scalesDictionary']['or'] # ora
        #
        unifiedVariables = VariablesScaling(dataBasic, scaledUnifiedVariables, 'unified', 'unscale')
        mainSet = UnifiedVariables2MainSet(nc, nr, unifiedVariables)
        errorG = p190227ErrorG(xs, ys, zs, cs, rs, mainSet)
        if thereIsH:
            try:
                horizonLine = MainSetAndZ02HorizonLine(dataBasic, mainSet, z0)
                errorH = p190227ErrorH(chs, rhs, horizonLine)
            except:
                errorH = 10000.
        else:
            errorH = 0.
        errorT = errorG + errorH
    except:
        print('*** p190227FunctionToMinimize0: error obtaining errorT'); assert False
    #
    return errorT
def p190227FunctionToMinimize1(scaledVariables1, theArgs): # 202106141800
    #
    ''' comments:
    .- input scaledVariables1 is a 5-float-ndarray (xc, yc, zc, k1a and sca)
    .- input theArgs is a dictionary
    .- output errorT is a float
    '''
    #
    # read theArgs
    try:
        dataBasic, scaledVariables0ForCodes = theArgs['dataBasic'], theArgs['scaledVariables0ForCodes']
        nc, nr, kc, kr = theArgs['nc'], theArgs['nr'], theArgs['kc'], theArgs['kr']
        xsForCodes, ysForCodes, zsForCodes, csForCodes, rsForCodes = theArgs['xsForCodes'], theArgs['ysForCodes'], theArgs['zsForCodes'], theArgs['csForCodes'], theArgs['rsForCodes']
        chsForCodes, rhsForCodes, z0ForCodes, thereIsHForCodes = theArgs['chsForCodes'], theArgs['rhsForCodes'], theArgs['z0ForCodes'], theArgs['thereIsHForCodes']
    except:
        print('*** p190227FunctionToMinimize1: error reading theArgs'); assert False
    #
    # obtain errorT
    try:
        errorsGAll, errorsHAll = [np.asarray([]) for item in range(2)]
        for fnCode in csForCodes.keys():
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
            #
            unifiedVariables = VariablesScaling(dataBasic, scaledUnifiedVariables, 'unified', 'unscale')
            mainSet = UnifiedVariables2MainSet(nc, nr, unifiedVariables)
            errorsGAll = np.concatenate((errorsGAll, p190227ErrorsG(xsForCodes[fnCode], ysForCodes[fnCode], zsForCodes[fnCode], csForCodes[fnCode], rsForCodes[fnCode], mainSet)))
            if thereIsHForCodes[fnCode]:
                try:
                    horizonLine = MainSetAndZ02HorizonLine(dataBasic, mainSet, z0ForCodes[fnCode])
                    errorsHAll = np.concatenate((errorsHAll, p190227ErrorsH(chsForCodes[fnCode], rhsForCodes[fnCode], horizonLine)))
                except:
                    errorsHAll = np.concatenate((errorsHAll, 10000. * np.ones(chsForCodes[fnCode])))
            #
        if len(errorsHAll) == 0:
            errorT = np.sqrt(np.mean(errorsGAll ** 2))
        else:
            errorT = np.sqrt(np.mean(errorsGAll ** 2)) + np.sqrt(np.mean(errorsHAll ** 2)) # WATCH OUT
    except:
        print('*** p190227FunctionToMinimize1: error obtaining errorT'); assert False
    #
    return errorT
def p190227ORBForBasis(pathFolder, options={}): # 202106141800
    #
    ''' comments:
    .- input pathFolder is a string
    '''
    #
    # complete options
    try:
        keys, defaultValues = ['nOfFeatures'], [10000]
        options = CompleteADictionary(options, keys, defaultValues)
    except:
        print('*** p190227ORBForBasis: error completing options'); assert False
    #
    # obtain fnsBasisImages, ncs, nrs, kpss and dess
    try:
        # initialize fnsBasisImages, ncs, nrs, kpss and dess
        fnsBasisImages, ncs, nrs, kpss, dess = [[] for item in range(5)]
        #
        # load fnsBasisImages, ncs, nrs, kpss and dess
        for fnBasisImage in sorted([item for item in os.listdir(pathFolder) if '.' in item and item[item.rfind('.')+1:] in ['jpeg', 'jpg', 'png']]):
            #
            # obtain ORB information
            optionsTMP = {'nOfFeatures':options['nOfFeatures']}
            nc, nr, kps, des, ctrl = ORBKeypoints(pathFolder + os.sep + fnBasisImage, optionsTMP)
            if not ctrl:
                continue
            fnsBasisImages.append(fnBasisImage); ncs.append(nc); nrs.append(nr); kpss.append(kps); dess.append(des)
    except:
        print('*** p190227ORBForBasis: error obtaining fnsBasisImages, ncs, nrs, kpss and dess'); assert False
    #
    return fnsBasisImages, ncs, nrs, kpss, dess
def p190227FindHomographyHa01(xs0, ys0, xs1, ys1): # 202106141800
    #
    ''' comments:
    .- input xs0, ys0, xs1 and ys1 are float-ndarrays of the same length
    .- output Ha is a 3x3-float-ndarray or None (if it does not succeed)
      .- Ha allows to transform from 0 to 1
    '''
    #
    # obtain Ha
    try:
        if len(xs0) == len(ys0) == len(xs1) == len(ys1) >= 4:
            nOfPoints = len(xs0)
            A, b = np.zeros((2 * nOfPoints, 8)), np.zeros(2 * nOfPoints)
            for pos in range(nOfPoints):
                pos0, pos1 = 2 * pos, 2 * pos + 1
                A[pos0, 0], A[pos0, 1], A[pos0, 2], A[pos0, 6], A[pos0, 7], b[pos0] = xs0[pos], ys0[pos], 1., -xs0[pos] * xs1[pos], -ys0[pos] * xs1[pos], xs1[pos]
                A[pos1, 3], A[pos1, 4], A[pos1, 5], A[pos1, 6], A[pos1, 7], b[pos1] = xs0[pos], ys0[pos], 1., -xs0[pos] * ys1[pos], -ys0[pos] * ys1[pos], ys1[pos]
            try:
                sol = np.linalg.solve(np.dot(np.transpose(A), A), np.dot(np.transpose(A), b))
                Ha = np.ones((3, 3))
                Ha[0, 0:3], Ha[1, 0:3], Ha[2, 0:2] = sol[0:3], sol[3:6], sol[6:8]
            except: # aligned points
                Ha = None
        else: # insufficient number of points
            Ha = None
    except:
        print('*** p190227FindHomographyHa01: error obtaining Ha'); assert False
    #
    return Ha
def p190227FindGoodPositionsForHomographyHa01ViaRANSAC(xs0, ys0, xs1, ys1, parametersRANSAC): # 202106141800
    #
    ''' comments:
    .- input xs0, ys0, xs1 and ys1 are float-ndarrays of the same length
    .- input parametersRANSAC is a dictionary (including at least 'p', 'e', 's' and 'errorC')
    .- output goodPositions is a integer-list or None (if it does not succeed)
    .- output dsGoodPositions is a float-ndarray of the same lenor None (if it does not succeed)
    '''
    #
    # obtain goodPositions and dsGoodPositions
    try:
        if len(xs0) == len(ys0) == len(xs1) == len(ys1) >= 4:
            nOfPoints = len(xs0)
            N = int(np.log(1. - parametersRANSAC['p']) / np.log(1. - (1. - parametersRANSAC['e']) ** parametersRANSAC['s'])) + 1
            #
            goodPositions = []
            for iN in range(N):
                poss4 = random.sample(range(0, nOfPoints), 4)
                Ha = p190227FindHomographyHa01(xs0[poss4], ys0[poss4], xs1[poss4], ys1[poss4])
                if Ha is None: # aligned points
                    continue
                dens = Ha[2, 0] * xs0 + Ha[2, 1] * ys0 + Ha[2, 2]
                if np.min(np.abs(dens)) < 1.e-6:
                    continue
                xs1R = (Ha[0, 0] * xs0 + Ha[0, 1] * ys0 + Ha[0, 2]) / dens
                ys1R = (Ha[1, 0] * xs0 + Ha[1, 1] * ys0 + Ha[1, 2]) / dens
                ds = np.sqrt((xs1R - xs1) ** 2 + (ys1R - ys1) ** 2)
                goodPositionsH = np.where(ds < parametersRANSAC['errorC'])[0]
                if len(goodPositionsH) > len(goodPositions):
                    goodPositions, dsGoodPositions = copy.deepcopy(goodPositionsH), ds[goodPositionsH]
        else:
            goodPositions, dsGoodPositions = None, None
    except:
        print('*** p190227FindGoodPositionsForHomographyHa01ViaRANSAC: error obtaining goodPositions and dsGoodPositions')
    #
    return goodPositions, dsGoodPositions



def p190227ErrorForAngles(x, theArgs): # 202106141800
    #
    ''' comments:
    .- input x is a 3-float-ndarray
    .- input theArgs is a dictionary
    '''
    #
    # read theArgs
    try:
        mainSet0 = theArgs['mainSet0']
        uUas0F, vUas0F = theArgs['uUas0F'], theArgs['vUas0F']
        uUas1F, vUas1F = theArgs['uUas1F'], theArgs['vUas1F']
    except:
        print('*** p190227ErrorForAngles: error reading theArgs'); assert False
    #
    # obtain f
    try:
        R0 = EulerianAngles2R(x[0], x[1], x[2])
        H = np.dot(mainSet0['R'], np.transpose(R0)) # sends R0 to mainSet0['R'] --coded 1 in the paper--
        dens = H[2, 0] * uUas0F + H[2, 1] * vUas0F + H[2, 2]
        uUas1FR = (H[0, 0] * uUas0F + H[0, 1] * vUas0F + H[0, 2]) / dens
        vUas1FR = (H[1, 0] * uUas0F + H[1, 1] * vUas0F + H[1, 2]) / dens
        f = np.sqrt(np.mean((uUas1FR - uUas1F) ** 2 + (vUas1FR - vUas1F) ** 2)) / mainSet0['sca']
    except:
        print('*** p190227ErrorForAngles: error obtaining f'); assert False
    #
    return f
#
# UCalib 2 main functions
def CalibrationOfBasis(pathFolderBasis, errorTCritical): # 202106141800
    #
    selectedUnifiedVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'sca']
    dataBasic = LoadDataBasic0({'selectedUnifiedVariablesKeys':selectedUnifiedVariablesKeys})
    assert np.abs(dataBasic['scalesDictionary']['sca'] - dataBasic['scalesDictionary']['sra']) < 1.e-11
    #
    # obtain filenames of the basis images in pathFolderBasis
    fnsImages = sorted([item for item in os.listdir(pathFolderBasis) if '.' in item and item[item.rfind('.')+1:] in ['jpeg', 'jpg', 'png']])
    fnsCodes = [item[0:item.rfind('.')] for item in fnsImages]
    #
    # load cdg, cdh and zms files
    csForCodes, rsForCodes, xsForCodes, ysForCodes, zsForCodes, chsForCodes, rhsForCodes, z0ForCodes, thereIsHForCodes = [{} for item in range(9)]
    for posFnImage, fnImage in enumerate(fnsImages):
        #
        fnCode = fnsCodes[posFnImage]
        #
        # load ncsForCodes and nrsForCodes
        nr, nc = cv2.imread(pathFolderBasis + os.sep + fnImage).shape[0:2]
        if posFnImage == 0:
            kc, kr = N2K(nc), N2K(nr)
            nc0, nr0 = nc, nr
        else:
            if not (nc == nc0 and nr == nr0):
                print('*** check that all the basis images have the same size'); assert False
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
        if os.path.exists(pathFolderBasis + os.sep + fnCdhTxt) and not os.path.exists(pathFolderBasis + os.sep + fnZmsTxt):
            print('*** error looking for file {:}'.format(fnZmsTxt)); assert False
        if os.path.exists(pathFolderBasis + os.sep + fnZmsTxt) and not os.path.exists(pathFolderBasis + os.sep + fnCdhTxt):
            print('*** error looking for file {:}'.format(fnCdhTxt)); assert False
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
    print('... files loaded successfully!')
    #
    # obtain one initial calibration
    errorTMin = 1.e+6
    for fnCode in fnsCodes:
        #
        optionsTMP = {'chs':None, 'rhs':None, 'z0':None, 'xc':None, 'yc':None, 'zc':None, 'selectedUnifiedVariablesKeys':selectedUnifiedVariablesKeys, 'MonteCarloNOfSeeds':10}
        mainSet, errorT = NonlinearManualCalibrationFromGCPs(xsForCodes[fnCode], ysForCodes[fnCode], zsForCodes[fnCode], csForCodes[fnCode], rsForCodes[fnCode], nc, nr, optionsTMP)
        if errorT < errorTMin:
            xc, yc, zc, ph, sg, ta, k1a, sca = [mainSet[item] for item in ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'sca']]
            errorTMin = 1. * errorT
        if errorTMin < 2.:
            break
    #
    # initialize scaledVariables0ForCodes (Code dependent: ph, sg, ta)
    scaledVariables0TMP = np.zeros(3)
    scaledVariables0TMP[0] = ph / dataBasic['scalesDictionary']['ph']
    scaledVariables0TMP[1] = sg / dataBasic['scalesDictionary']['sg']
    scaledVariables0TMP[2] = ta / dataBasic['scalesDictionary']['ta']
    scaledVariables0ForCodes = {}
    for fnCode in fnsCodes:
        scaledVariables0ForCodes[fnCode] = scaledVariables0TMP
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
    # calibrations forcing constant camera position and intrinsic parameter
    counter, errorT0ForCodes = 0, {}
    while counter < 10000:
        #
        print('... iteration {:2}'.format(counter+1))
        #
        # recompute Code dependent (scaledVariables0: ph, sg, ta) given Code independent (scaledVariables1: xc, yc, zc, k1a, sca)
        for fnCode in fnsCodes:
            theArgs = {}
            theArgs['dataBasic'], theArgs['scaledVariables1'] = dataBasic, scaledVariables1
            theArgs['nc'], theArgs['nr'], theArgs['kc'], theArgs['kr'] = nc, nr, kc, kr
            theArgs['xs'], theArgs['ys'], theArgs['zs'], theArgs['cs'], theArgs['rs'] = xsForCodes[fnCode], ysForCodes[fnCode], zsForCodes[fnCode], csForCodes[fnCode], rsForCodes[fnCode]
            theArgs['chs'], theArgs['rhs'], theArgs['z0'], theArgs['thereIsH'] = chsForCodes[fnCode], rhsForCodes[fnCode], z0ForCodes[fnCode], thereIsHForCodes[fnCode]
            #
            scaledVariables0 = scaledVariables0ForCodes[fnCode]
            errorT0 = p190227FunctionToMinimize0(scaledVariables0, theArgs)
            #
            scaledVariables0 = optimize.minimize(p190227FunctionToMinimize0, scaledVariables0, args = (theArgs)).x
            errorT0 = p190227FunctionToMinimize0(scaledVariables0, theArgs)
            scaledVariables0ForCodes[fnCode], errorT0ForCodes[fnCode] = scaledVariables0, errorT0
            print('  ... {:}: errorT {:8.2f}'.format(fnCode, errorT0))
        #
        if (counter > 3 and np.sqrt(np.mean(np.asarray([errorT0ForCodes[fnCode] for fnCode in fnsCodes]) ** 2)) < 1.) or counter > 40:
            break
        #
        # recompute Code independent (scaledVariables1: xc, yc, zc, k1a, sca) given Code dependent (scaledVariables0: ph, sg, ta)
        theArgs = {}
        theArgs['dataBasic'], theArgs['scaledVariables0ForCodes'] = dataBasic, scaledVariables0ForCodes
        theArgs['nc'], theArgs['nr'], theArgs['kc'], theArgs['kr'] = nc, nr, kc, kr
        theArgs['xsForCodes'], theArgs['ysForCodes'], theArgs['zsForCodes'], theArgs['csForCodes'], theArgs['rsForCodes'] = xsForCodes, ysForCodes, zsForCodes, csForCodes, rsForCodes
        theArgs['chsForCodes'], theArgs['rhsForCodes'], theArgs['z0ForCodes'], theArgs['thereIsHForCodes'] = chsForCodes, rhsForCodes, z0ForCodes, thereIsHForCodes
        #
        scaledVariables1 = optimize.minimize(p190227FunctionToMinimize1, scaledVariables1, args = (theArgs)).x
        errorT1 = p190227FunctionToMinimize1(scaledVariables1, theArgs)
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
    print('... final calibrations obtained!')
    #
    for fnCode in fnsCodes:
        scaledUnifiedVariables = np.zeros(14)
        scaledUnifiedVariables[0] = scaledVariables1[0] # xc - constant
        scaledUnifiedVariables[1] = scaledVariables1[1] # yc - constant
        scaledUnifiedVariables[2] = scaledVariables1[2] # zc - constant
        scaledUnifiedVariables[3] = scaledVariables0ForCodes[fnCode][0] # ph
        scaledUnifiedVariables[4] = scaledVariables0[1] # sg
        scaledUnifiedVariables[5] = scaledVariables0[2] # ta
        scaledUnifiedVariables[6] = scaledVariables1[3] # k1a - constant
        scaledUnifiedVariables[7] = 0. # k2a
        scaledUnifiedVariables[8] = 0. # p1a
        scaledUnifiedVariables[9] = 0. # p2a
        scaledUnifiedVariables[10] = scaledVariables1[4] # sca - constant
        scaledUnifiedVariables[11] = scaledVariables1[4] # sra - constant
        scaledUnifiedVariables[12] = kc / dataBasic['scalesDictionary']['oc'] # oca
        scaledUnifiedVariables[13] = kr / dataBasic['scalesDictionary']['or'] # ora
        #
        unifiedVariables = VariablesScaling(dataBasic, scaledUnifiedVariables, 'unified', 'unscale')
        mainSet = UnifiedVariables2MainSet(nc, nr, unifiedVariables)
        errorsG = p190227ErrorsG(xsForCodes[fnCode], ysForCodes[fnCode], zsForCodes[fnCode], csForCodes[fnCode], rsForCodes[fnCode], mainSet)
        for pos in range(len(errorsG)):
            if errorsG[pos] > errorTCritical: # IMP*
                print('*** for image {:}, the error of pixel in line {:} is {:5.1f} > critical error = {:5.1f}: consider to check or remove it'.format(fnCode, pos+1, errorsG[pos], errorTCritical))
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
        unifiedVariables = VariablesScaling(dataBasic, scaledUnifiedVariables, 'unified', 'unscale')
        mainSet = UnifiedVariables2MainSet(nc, nr, unifiedVariables)
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
    return None
def AutoCalibrationOfImages(pathFolderBasis, pathFolderImagesToAutoCalibrate, nOfFeaturesORB, fC, KC): # 202106141800
    #
    unifiedVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']
    #
    # perform ORB for the basis in pathFolderBasis
    fnsBasisImages, ncs, nrs, kpss, dess = p190227ORBForBasis(pathFolderBasis, {'nOfFeatures':nOfFeaturesORB})
    #
    # load mainSets and Hs for the basis (actually, for those successful from ORB)
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
        unifiedVariables = Dictionary2Array(unifiedVariablesKeys, unifiedVariablesDictionary)
        mainSets.append(UnifiedVariables2MainSet(nc, nr, unifiedVariables))
        Hs.append(np.dot(mainSets[0]['R'], np.transpose(mainSets[posFnBasisImage]['R'])))
    #
    # autocalibrate images from pathFolderImagesToAutoCalibrate
    fnsImagesToAutoCalibrate = sorted([item for item in os.listdir(pathFolderImagesToAutoCalibrate) if item[item.rfind('.')+1:] in ['jpg', 'png', 'jpeg']])
    for fnImageToAutoCalibrate in fnsImagesToAutoCalibrate:
        #
        print('... autocalibration of {:}'.format(fnImageToAutoCalibrate))
        #
        nc0, nr0, kps0, des0, ctrl0 = ORBKeypoints(pathFolderImagesToAutoCalibrate + os.sep + fnImageToAutoCalibrate, {'nOfFeatures':nOfFeaturesORB})
        if not ctrl0:
            print('  ... not calibratable automatically (ORB failure)'); continue
        if not (nc0 == nc and nr0 == nr):
            print('  ... image with wrong shape!'); assert False
        #
        # find pairs with basis images in the space (u, v)
        uUas0F, vUas0F = [np.asarray([]) for item in range(2)]
        uUas1F, vUas1F = [np.asarray([]) for item in range(2)]
        for posFnBasisImage, fnBasisImage in enumerate(fnsBasisImages):
            # find matches
            cDs0, rDs0, cDsB, rDsB, ersB = ORBMatches(kps0, des0, kpss[posFnBasisImage], dess[posFnBasisImage], options={})
            if len(cDs0) == 0:
                continue
            # update 0, the image to calibrate (recall that intrinsic is constant, we use mainSets[0])
            uDas0, vDas0 = CR2UaVa(mainSets[0], cDs0, rDs0)[0:2]
            uUas0, vUas0 = UDaVDa2UUaVUa(mainSets[0], uDas0, vDas0)
            uUas0F, vUas0F = np.concatenate((uUas0F, uUas0)), np.concatenate((vUas0F, vUas0))
            # update 1, in the first image of the basis (recall that intrinsic is constant, we use mainSets[0])
            uDasB, vDasB = CR2UaVa(mainSets[0], cDsB, rDsB)[0:2]
            uUasB, vUasB = UDaVDa2UUaVUa(mainSets[0], uDasB, vDasB)
            dens = Hs[posFnBasisImage][2, 0] * uUasB + Hs[posFnBasisImage][2, 1] * vUasB + Hs[posFnBasisImage][2, 2]
            uUas1 = (Hs[posFnBasisImage][0, 0] * uUasB + Hs[posFnBasisImage][0, 1] * vUasB + Hs[posFnBasisImage][0, 2]) / dens
            vUas1 = (Hs[posFnBasisImage][1, 0] * uUasB + Hs[posFnBasisImage][1, 1] * vUasB + Hs[posFnBasisImage][1, 2]) / dens
            uUas1F, vUas1F = np.concatenate((uUas1F, uUas1)), np.concatenate((vUas1F, vUas1))
        #
        if len(uUas0F) < min([4, KC]):
            print('  ... not calibratable automatically (K <= {:})'.format(len(uUas0F))); continue
        #
        # apply RANSAC (in pixels to control the error in parametersRANSAC)
        cUs0F, rUs0F = UaVa2CR(mainSets[0], uUas0F, vUas0F)[0:2]
        cUs1F, rUs1F = UaVa2CR(mainSets[0], uUas1F, vUas1F)[0:2]
        parametersRANSAC = {'e':0.8, 's':4, 'p':0.999999, 'errorC':2.} # should be fine (on the safe side)
        goodPositions, dsGoodPositions = p190227FindGoodPositionsForHomographyHa01ViaRANSAC(cUs0F, rUs0F, cUs1F, rUs1F, parametersRANSAC)
        if any([item is None for item in [goodPositions, dsGoodPositions]]) or len(goodPositions) < KC:
            print('  ... not calibratable automatically (K <= {:})'.format(len(goodPositions))); continue
        uUas0F, vUas0F = uUas0F[goodPositions], vUas0F[goodPositions]
        uUas1F, vUas1F = uUas1F[goodPositions], vUas1F[goodPositions]
        cUs0F, rUs0F = cUs0F[goodPositions], rUs0F[goodPositions]
        cUs1F, rUs1F = cUs1F[goodPositions], rUs1F[goodPositions]
        #
        # grid selection through distorted pixels
        uDas0F, vDas0F = UUaVUa2UDaVDa(mainSets[0], uUas0F, vUas0F)
        uDas1F, vDas1F = UUaVUa2UDaVDa(mainSets[0], uUas1F, vUas1F)
        cDs0F, rDs0F = UaVa2CR(mainSets[0], uDas0F, vDas0F)[0:2]
        cDs1F, rDs1F = UaVa2CR(mainSets[0], uDas1F, vDas1F)[0:2]
        possGrid = SelectPixelsInGrid(10, nc, nr, cDs0F, rDs0F, dsGoodPositions)[0]
        K = len(possGrid)
        #
        if K < KC:
            print('  ... not calibratable automatically (K = {:})'.format(K)); continue
        #
        # find the angles
        x0 = np.asarray([mainSets[0]['ph'], mainSets[0]['sg'], mainSets[0]['ta']])
        theArgs = {}
        theArgs['mainSet0'] = mainSets[0]
        theArgs['uUas0F'], theArgs['vUas0F'] = uUas0F[possGrid], vUas0F[possGrid]
        theArgs['uUas1F'], theArgs['vUas1F'] = uUas1F[possGrid], vUas1F[possGrid]
        xN = optimize.minimize(p190227ErrorForAngles, x0, args = (theArgs)).x
        f = p190227ErrorForAngles(xN, theArgs)
        if f > fC:
            print('  ... not calibratable automatically (f = {:4.1e})'.format(f)); continue
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
    return None

