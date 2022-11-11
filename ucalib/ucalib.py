#'''
# Created on 2021 by Gonzalo Simarro and Daniel Calvete
#'''
#
import cv2
import numpy as np
import os
from scipy import optimize
import shutil
import sys
#
#
#
import ulises_ucalib as ulises
#
def CalibrationOfBasisImages(pathFldBasis, errorTCritical, model, givenVariablesDict, verbosePlot): # 202211081147 (last read 2022-11-08)
    #
    # manage model
    pK = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'sca']
    qK = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'sca']
    fK = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']
    model2SelectedVariablesKeys = {'parabolic':pK, 'quartic':qK, 'full':fK}
    if model not in model2SelectedVariablesKeys.keys():
        print('*** Invalid calibration model ({:}): choose among {:}'.format(model, list(model2SelectedVariablesKeys.keys()))); sys.exit()
    if not set(givenVariablesDict.keys()) <= set(model2SelectedVariablesKeys[model]):
        print('*** Invalid forced varibles ({:}): not in the calibration model'.format(list(givenVariablesDict.keys()))); sys.exit()
    freeVariablesKeys = [item for item in model2SelectedVariablesKeys[model] if item not in givenVariablesDict.keys()]
    #
    # obtain calibrations
    extsImg = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']
    fnsImgs = sorted([item for item in os.listdir(pathFldBasis) if '.' in item and os.path.splitext(item)[1][1:] in extsImg])
    for posFnImg, fnImg in enumerate(fnsImgs):
        print('... calibration of {:}'.format(fnImg), end='', flush=True)
        #
        # load image information and dataBasic
        if posFnImg == 0:
            nr, nc = cv2.imread(os.path.join(pathFldBasis, fnImg)).shape[0:2]
            dataBasic = ulises.LoadDataBasic0(options={'nc':nc, 'nr':nr, 'selectedVariablesKeys':model2SelectedVariablesKeys[model]})
        else:
            assert cv2.imread(os.path.join(pathFldBasis, fnImg)).shape[0:2] == (nr, nc)
        #
        # load GCPs
        pathCdgTxt = os.path.join(pathFldBasis, '{:}cdg.txt'.format(os.path.splitext(fnImg)[0]))
        cs, rs, xs, ys, zs = ulises.ReadCdgTxt(pathCdgTxt, options={'readCodes':False, 'readOnlyGood':True})[0:5]
        #
        # load horizon points
        pathCdhTxt = os.path.join(pathFldBasis, '{:}cdh.txt'.format(os.path.splitext(fnImg)[0]))
        if os.path.exists(pathCdhTxt):
            chs, rhs = ulises.ReadCdhTxt(pathCdhTxt, options={'readOnlyGood':True})
        else:
            chs, rhs = [np.asarray([]) for item in range(2)]
        #
        # load dataForCal (aG, aH, mainSetSeeds are in dataForCal)
        dataForCal = {'nc':nc, 'nr':nr, 'cs':cs, 'rs':rs, 'xs':xs, 'ys':ys, 'zs':zs, 'aG':1., 'mainSetSeeds':[]} # IMP*
        if len(chs) == len(rhs) > 0:
            dataForCal['chs'], dataForCal['rhs'], dataForCal['aH'] = chs, rhs, 1. # IMP*
        for fn in [item for item in os.listdir(pathFldBasis) if 'cal' in item and item.endswith('txt')]:
            allVariablesH, ncH, nrH = ulises.ReadCalTxt(os.path.join(pathFldBasis, fn))[0:3]
            dataForCal['mainSetSeeds'].append(ulises.AllVariables2MainSet(allVariablesH, ncH, nrH, options={})) # IMP*
        #
        # obtain calibration
        mainSet, errorT = ulises.NonlinearManualCalibration(dataBasic, dataForCal, freeVariablesKeys, givenVariablesDict, options={})
        #
        # inform and write
        if errorT <= 1. * errorTCritical:
            print(' success')
            #
            # check potential bad GCPs
            csR, rsR = ulises.XYZ2CDRD(mainSet, xs, ys, zs, options={})[0:2]
            errorsG = np.sqrt((csR - cs) ** 2 + (rsR - rs) ** 2)
            for pos in range(len(errorsG)):
                if errorsG[pos] > errorTCritical:
                    print('*** the error of GCP at x = {:8.2f}, y = {:8.2f} and z = {:8.2f} is {:5.1f} > critical error = {:5.1f}: consider to check or remove it'.format(xs[pos], ys[pos], zs[pos], errorsG[pos], errorTCritical))
            #
            # write pathCal0Txt
            pathCal0Txt = os.path.join(pathFldBasis, '{:}cal0.txt'.format(os.path.splitext(fnImg)[0]))
            ulises.WriteCalTxt(pathCal0Txt, mainSet['allVariables'], mainSet['nc'], mainSet['nr'], errorT)
            #
            # plot results
            if verbosePlot:
                pathFldTMP = os.path.join(os.path.split(pathFldBasis)[0], 'TMP')
                ulises.MakeFolder(pathFldTMP)
                pathTMP0, pathTMP1 = os.path.join(pathFldBasis, fnImg), os.path.join(pathFldTMP, fnImg.replace('.', 'cal0_check.'))
                ulises.PlotMainSet(pathTMP0, mainSet, cs, rs, xs, ys, zs, chs, rhs, pathTMP1)
        else:
            print(' failed (error = {:6.1f})'.format(errorT))
            print('*** re-run and, if it keeps failing, check quality of the GCP or try another calibration model ***')
    #
    return None
#
def CalibrationOfBasisImagesConstantXYZAndIntrinsic(pathFldBasis, model, givenVariablesDict, verbosePlot): # 202211101439 (last read 2022-11-08)
    #
    # manage model
    pK = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'sca']
    qK = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'sca']
    fK = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']
    model2SelectedVariablesKeys = {'parabolic':pK, 'quartic':qK, 'full':fK}
    if model not in model2SelectedVariablesKeys.keys():
        print('*** Invalid calibration model ({:}): choose among {:}'.format(model, list(model2SelectedVariablesKeys.keys()))); sys.exit()
    if not set(givenVariablesDict.keys()) <= set(model2SelectedVariablesKeys[model]):
        print('*** Invalid forced varibles ({:}): not in the calibration model'.format(list(givenVariablesDict.keys()))); sys.exit()
    #
    # load basis information
    extsImg = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']
    potentialFnsImgs = sorted([item for item in os.listdir(pathFldBasis) if '.' in item and os.path.splitext(item)[1][1:] in extsImg])
    ncs, nrs, css, rss, xss, yss, zss, chss, rhss, allVariabless, mainSets, errorTs, fnsImgs = [[] for item in range(13)]
    for posFnImg, fnImg in enumerate(potentialFnsImgs):
        #
        # manage posFnImg == 0 and load dataBasic
        if posFnImg == 0:
            nr, nc = cv2.imread(os.path.join(pathFldBasis, fnImg)).shape[0:2]
            dataBasic = ulises.LoadDataBasic0(options={'nc':nc, 'nr':nr, 'selectedVariablesKeys':model2SelectedVariablesKeys[model]})
        else:
            assert cv2.imread(os.path.join(pathFldBasis, fnImg)).shape[0:2] == (nr, nc)
        #
        # check pathCal0Txt
        pathCal0Txt = os.path.join(pathFldBasis, '{:}cal0.txt'.format(os.path.splitext(fnImg)[0]))
        if not os.path.exists(pathCal0Txt):
            print('... image {:} ignored: no initial calibration available'.format(fnImg)); continue
        #
        # update ncs and nrs
        ncs.append(nc); nrs.append(nr)
        #
        # update css, rss, xss, yss and zss
        pathCdgTxt = os.path.join(pathFldBasis, '{:}cdg.txt'.format(os.path.splitext(fnImg)[0]))
        cs, rs, xs, ys, zs = ulises.ReadCdgTxt(pathCdgTxt, options={'readCodes':False, 'readOnlyGood':True})[0:5]
        css.append(cs); rss.append(rs); xss.append(xs); yss.append(ys); zss.append(zs)
        #
        # update chss and rhss
        pathCdhTxt = os.path.join(pathFldBasis, '{:}cdh.txt'.format(os.path.splitext(fnImg)[0]))
        if os.path.exists(pathCdhTxt):
            chs, rhs = ulises.ReadCdhTxt(pathCdhTxt, options={'readOnlyGood':True})
        else:
            chs, rhs = [np.asarray([]) for item in range(2)]
        chss.append(chs); rhss.append(rhs)
        #
        # update allVariabless, mainSets, errorTs and fnsImgs
        allVariables, nc, nr, errorT = ulises.ReadCalTxt(pathCal0Txt)
        mainSet = ulises.AllVariables2MainSet(allVariables, nc, nr, options={})
        allVariabless.append(allVariables); mainSets.append(mainSet); errorTs.append(errorT); fnsImgs.append(fnImg)
    #
    # obtain calibrations and write pathCalTxts forcing unique freeU variables
    if len(fnsImgs) == 0:
        print('*** no initial calibrations available'); sys.exit()
    elif len(fnsImgs) == 1:
        pathCal0Txt = os.path.join(pathFldBasis, '{:}cal0.txt'.format(os.path.splitext(fnsImgs[0])[0]))
        pathCalTxt = os.path.join(pathFldBasis, '{:}cal.txt'.format(os.path.splitext(fnsImgs[0])[0]))
        shutil.copyfile(pathCal0Txt, pathCalTxt)
    else:
        freeDVariablesKeys = ['ph', 'sg', 'ta'] # IMP*
        freeUVariablesKeys = [item for item in model2SelectedVariablesKeys[model] if item not in freeDVariablesKeys + list(givenVariablesDict.keys())]
        dataForCals = {'ncs':ncs, 'nrs':nrs, 'css':css, 'rss':rss, 'xss':xss, 'yss':yss, 'zss':zss, 'aG':1., 'chss':chss, 'rhss':rhss, 'aH':1., 'mainSetSeeds':mainSets}
        mainSets, errorTs = ulises.NonlinearManualCalibrationOfSeveralImages(dataBasic, dataForCals, freeDVariablesKeys, freeUVariablesKeys, givenVariablesDict, options={})
        for pos in range(len(fnsImgs)):
            pathTMP = os.path.join(pathFldBasis, '{:}cal.txt'.format(os.path.splitext(fnsImgs[pos])[0]))
            ulises.WriteCalTxt(pathTMP, mainSets[pos]['allVariables'], mainSets[pos]['nc'], mainSets[pos]['nr'], errorTs[pos])
    #
    # plot results
    if verbosePlot:
        pathFldTMP = os.path.join(os.path.split(pathFldBasis)[0], 'TMP')
        ulises.MakeFolder(pathFldTMP)
        for pos in range(len(fnsImgs)):
            pathTMP0, pathTMP1 = os.path.join(pathFldBasis, fnsImgs[pos]), os.path.join(pathFldTMP, fnsImgs[pos].replace('.', 'cal_check.'))
            ulises.PlotMainSet(pathTMP0, mainSets[pos], css[pos], rss[pos], xss[pos], yss[pos], zss[pos], chss[pos], rhss[pos], pathTMP1)
    #
    return None
#
def AutoCalibrationOfImages(pathFldBasis, pathFldImages, nOfFeaturesORB, fC, KC, overwrite, verbosePlot): #202211101441 (last read 2022-11-08 except details)
    #
    # manage verbosePlot
    if verbosePlot:
        pathFldTMP = os.path.join(os.path.split(pathFldImages)[0], 'TMP')
        ulises.MakeFolder(pathFldTMP)
        xs, ys, zs = [np.asarray([]) for item in range(3)]
        for fnCdgTxt in [item for item in os.listdir(pathFldBasis) if 'cdg' in item and item.endswith('txt')]:
            xsH, ysH, zsH = ulises.ReadCdgTxt(os.path.join(pathFldBasis, fnCdgTxt), options={})[2:5]
            xs, ys, zs = np.concatenate((xs, xsH)), np.concatenate((ys, ysH)), np.concatenate((zs, zsH))
        cs, rs = [-999. * np.ones(xs.shape) for item in range(2)]
    #
    # perform ORB for basis
    fnsBasisImgs, ncs, nrs, kpss, dess = ulises.ORBKeypointsForAllImagesInAFolder(pathFldBasis, options={'nOfFeatures':nOfFeaturesORB})
    #
    # load mainSets and Hs for basis
    mainSets, Hs = [[] for item in range(2)]
    for posFnBasisImg, fnBasisImg in enumerate(fnsBasisImgs):
        pathCalTxt = os.path.join(pathFldBasis, '{:}cal.txt'.format(os.path.splitext(fnBasisImg)[0]))
        if os.path.exists(pathCalTxt):
            allVariables, nc, nr = ulises.ReadCalTxt(pathCalTxt)[0:3]
            mainSets.append(ulises.AllVariables2MainSet(allVariables, nc, nr, options={}))
            Hs.append(np.dot(mainSets[0]['R'], np.transpose(mainSets[posFnBasisImg]['R']))) # posFnBasisImg -> posFnBasisImg=0 (first image of the basis)
        else:
            mainSets.append(None)
            Hs.append(None)
    #
    # obtain calibrations and write pathCalTxts
    extsImg = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']
    fnsImgs = sorted([item for item in os.listdir(pathFldImages) if '.' in item and os.path.splitext(item)[1][1:] in extsImg])
    for fnImg in fnsImgs:
        #
        print('... autocalibration of {:}'.format(fnImg), end='', flush=True)
        #
        # obtain pathCalTxt and check if already exists
        pathCalTxt = os.path.join(pathFldImages, '{:}cal.txt'.format(os.path.splitext(fnImg)[0]))
        if os.path.exists(pathCalTxt) and not overwrite:
            print(' already calibrated'); continue
        #
        # perform ORB
        nc0, nr0, kps0, des0, ctrl0 = ulises.ORBKeypoints(os.path.join(pathFldImages, fnImg), {'nOfFeatures':nOfFeaturesORB})
        if not ctrl0:
            print(' failed (ORB failure)'); continue
        if not (nc0 == nc and nr0 == nr):
            print(' failed (image with wrong shape!)'); continue
        #
        # find pairs with basis images in the space (u, v), F = Full (all basis images)
        uUas0F, vUas0F = [np.asarray([]) for item in range(2)] # image to calibrate
        uUas1F, vUas1F = [np.asarray([]) for item in range(2)] # first image of the basis
        for posFnBasisImg, fnBasisImg in enumerate(fnsBasisImgs):
            #
            # manage not calibrated basis images
            if any([item is None for item in [mainSets[posFnBasisImg], Hs[posFnBasisImg]]]):
                continue
            #
            # find matches
            cDs0, rDs0, cDsB, rDsB, ersB = ulises.ORBMatches(kps0, des0, kpss[posFnBasisImg], dess[posFnBasisImg], options={'erMaximum':30., 'nOfStd':2.0})
            if len(cDs0) == 0: 
                continue
            #
            # update 0, the image to calibrate (recall that intrinsic is constant, we use mainSets[0])
            uDas0, vDas0 = ulises.CR2UaVa(mainSets[0], cDs0, rDs0)[0:2]
            uUas0, vUas0 = ulises.UDaVDa2UUaVUa(mainSets[0], uDas0, vDas0)
            uUas0F, vUas0F = np.concatenate((uUas0F, uUas0)), np.concatenate((vUas0F, vUas0))
            #
            # update 1, in the first image of the basis (recall that intrinsic is constant, we use mainSets[0])
            uDasB, vDasB = ulises.CR2UaVa(mainSets[0], cDsB, rDsB)[0:2]
            uUasB, vUasB = ulises.UDaVDa2UUaVUa(mainSets[0], uDasB, vDasB)
            uUas1, vUas1 = ulises.ApplyHomographyHa01(Hs[posFnBasisImg], uUasB, vUasB)
            uUas1F, vUas1F = np.concatenate((uUas1F, uUas1)), np.concatenate((vUas1F, vUas1))
            #
        if len(uUas0F) < min([4, KC]):
            print(' failed (K = {:}) after ORB'.format(len(uUas0F))); continue
        #
        # apply RANSAC in the pixel domain (parRANSAC['errorC'] in pixels)
        cUs0F, rUs0F = ulises.UaVa2CR(mainSets[0], uUas0F, vUas0F)[0:2] # image to calibrate
        cUs1F, rUs1F = ulises.UaVa2CR(mainSets[0], uUas1F, vUas1F)[0:2] # first image of the basis
        parRANSAC = {'e':0.8, 's':4, 'p':0.999999, 'errorC':2.} # should be fine (on the safe side)
        Ha01, possGood = ulises.FindHomographyHa01ViaRANSAC(cUs0F, rUs0F, cUs1F, rUs1F, parRANSAC)
        if Ha01 is None or possGood is None or len(possGood) < KC:
            print(' failed (K < KC) after RANSAC'); continue
        uUas0F, vUas0F, cUs0F, rUs0F = [item[possGood] for item in [uUas0F, vUas0F, cUs0F, rUs0F]] # image to calibrate
        uUas1F, vUas1F, cUs1F, rUs1F = [item[possGood] for item in [uUas1F, vUas1F, cUs1F, rUs1F]] # first image of the basis
        #
        # apply grid selection (after computing error-distances)
        cUs1FR, rUs1FR = ulises.ApplyHomographyHa01(Ha01, cUs0F, rUs0F)
        dsGoodPositions = np.sqrt((cUs1FR - cUs1F) ** 2 + (rUs1FR - rUs1F) ** 2)
        uDas0F, vDas0F = ulises.UUaVUa2UDaVDa(mainSets[0], uUas0F, vUas0F) # image to calibrate
        uDas1F, vDas1F = ulises.UUaVUa2UDaVDa(mainSets[0], uUas1F, vUas1F) # first image of the basis
        cDs0F, rDs0F = ulises.UaVa2CR(mainSets[0], uDas0F, vDas0F)[0:2] # image to calibrate
        cDs1F, rDs1F = ulises.UaVa2CR(mainSets[0], uDas1F, vDas1F)[0:2] # first image of the basis
        possGrid = ulises.SelectPixelsInGrid(10, nc, nr, cDs1F, rDs1F, dsGoodPositions)[0]
        K = len(possGrid)
        if K < KC:
            print(' failed (K = {:}) after grid selection'.format(K)); continue
        uUas0F, vUas0F = [item[possGrid] for item in [uUas0F, vUas0F]] # image to calibrate
        uUas1F, vUas1F = [item[possGrid] for item in [uUas1F, vUas1F]] # first image of the basis
        #
        # obtain the eulerian angles
        x0 = np.asarray([mainSets[0]['ph'], mainSets[0]['sg'], mainSets[0]['ta']]) # seed image to calibrate
        theArgs = {'R1':mainSets[0]['R'], 'sca':mainSets[0]['sca'], 'uUas0':uUas0F, 'vUas0':vUas0F, 'uUas1':uUas1F, 'vUas1':vUas1F}
        xN = optimize.minimize(ulises.PixelsErrorOfRotationalHomographyUsingUVUas, x0, args = (theArgs)).x
        f = ulises.PixelsErrorOfRotationalHomographyUsingUVUas(xN, theArgs)
        if f > fC:
            print(' failed (f = {:4.1e})'.format(f)); continue
        #
        # write pathCalTxt
        print(' success')
        fileout = open(pathCalTxt, 'w')
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
        #
        # plot results
        if verbosePlot:
            allVariables, nc, nr = ulises.ReadCalTxt(pathCalTxt)[0:3]
            mainSet = ulises.AllVariables2MainSet(allVariables, nc, nr, options={})
            pathTMP0, pathTMP1 = os.path.join(pathFldImages, fnImg), os.path.join(pathFldTMP, fnImg.replace('.', 'cal_check.'))
            ulises.PlotMainSet(pathTMP0, mainSet, cs, rs, xs, ys, zs, None, None, pathTMP1)
    #
    return None
#
def PlanviewsFromImages(pathFldImages, pathFldPlanviews, z0, ppm, overwrite, verbosePlot): # 202211101430 (last read 2022-11-08)
    #
    # obtain the planview domain from the cloud of points
    if not os.path.exists(pathFldPlanviews):
        print('*** folder {:} not found'.format(pathFldPlanviews)); sys.exit()
    if not os.path.exists(os.path.join(pathFldPlanviews, 'xy_planview.txt')):
        print('*** file xy_planview.txt not found in {:}'.format(pathFldPlanviews)); sys.exit()
    rawData = np.asarray(ulises.ReadRectangleFromTxt(os.path.join(pathFldPlanviews, 'xy_planview.txt'), options={'c1':2, 'valueType':'float'}))
    xsCloud, ysCloud = rawData[:, 0], rawData[:, 1]
    xcs, ycs = ulises.CloudOfPoints2Rectangle(xsCloud, ysCloud, options={})[0:2]
    angle, xUL, yUL = np.angle((xcs[1]-xcs[0])+1j*(ycs[1]-ycs[0])), xcs[0], ycs[0]
    xyLengthInR = np.sqrt((xcs[2]-xcs[1])**2+(ycs[2]-ycs[1])**2)
    xyLengthInC = np.sqrt((xcs[1]-xcs[0])**2+(ycs[1]-ycs[0])**2)
    dataPdfTxt = ulises.LoadDataPdfTxt(options={'xUL':xUL, 'yUL':yUL, 'angle':angle, 'xyLengthInC':xyLengthInC, 'xyLengthInR':xyLengthInR, 'ppm':ppm})
    csCloud, rsCloud = ulises.PlanXY2CR(xsCloud, ysCloud, dataPdfTxt['xUL'], dataPdfTxt['yUL'], dataPdfTxt['angle'], dataPdfTxt['ppm'], options={})[0:2]
    #
    # write the planview domain
    fileout = open(os.path.join(pathFldPlanviews, 'crxyz_planview.txt'), 'w')
    for pos in range(4):
        c, r, x, y = dataPdfTxt['csC'][pos], dataPdfTxt['rsC'][pos], dataPdfTxt['xsC'][pos], dataPdfTxt['ysC'][pos]
        fileout.write('{:6.0f} {:6.0f} {:8.2f} {:8.2f} {:8.2f}\t c, r, x, y and z\n'.format(c, r, x, y, z0))
    fileout.close()
    #
    # obtain and write planviews
    extsImg = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']
    fnsImgs = sorted([item for item in os.listdir(pathFldImages) if '.' in item and os.path.splitext(item)[1][1:] in extsImg])
    for fnImg in fnsImgs:
        #
        # inform
        print('... planview of {:}'.format(fnImg), end='', flush=True)
        #
        # obtain pathPlw and check if already exists
        pathPlw = os.path.join(pathFldPlanviews, fnImg.replace('.', 'plw.'))
        if os.path.exists(pathPlw) and not overwrite:
            print(' already exists'); continue
        #
        # load calibration and obtain and write planview
        pathCalTxt = os.path.join(pathFldImages, '{:}cal.txt'.format(os.path.splitext(fnImg)[0]))
        if os.path.exists(pathCalTxt):
            #
            # load calibration
            allVariables, nc, nr, errorT = ulises.ReadCalTxt(pathCalTxt)
            mainSet = ulises.AllVariables2MainSet(allVariables, nc, nr, options={})
            #
            # obtain and write planview
            plwPC = ulises.PlanviewPrecomputations({'01':mainSet}, dataPdfTxt, z0)
            img = cv2.imread(os.path.join(pathFldImages, fnImg))
            imgPlanview = ulises.CreatePlanview(plwPC, {'01':img})
            cv2.imwrite(pathPlw, imgPlanview)
            print(' success')
            #
            # plot results
            if verbosePlot:
                pathFldTMP = os.path.join(os.path.split(pathFldPlanviews)[0], 'TMP')
                ulises.MakeFolder(pathFldTMP)
                #
                imgTMP = ulises.DisplayCRInImage(imgPlanview, csCloud, rsCloud, options={'colors':[[0, 255, 255]], 'size':10})
                cv2.imwrite(os.path.join(pathFldTMP, fnImg.replace('.', 'plw_check.')), imgTMP)
                #
                cs, rs = ulises.XYZ2CDRD(mainSet, xsCloud, ysCloud, z0)[0:2]
                imgTMP = ulises.DisplayCRInImage(img, cs, rs, options={'colors':[[0, 255, 255]], 'size':10})
                cv2.imwrite(os.path.join(pathFldTMP, fnImg.replace('.', '_checkplw.')), imgTMP)
        else:
            print(' failed (no calibration file)')
    #
    return None
#
def CheckGCPs(pathFldBasisCheck, errorCritical): # 202211081210 (last read 2022-11-08)
    #
    # set RANSAC parameters
    eRANSAC, pRANSAC, ecRANSAC, NForRANSACMax = 0.8, 0.999999, errorCritical, 50000
    #
    # check GCPs
    extsImg = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']
    fnsImgs = sorted([item for item in os.listdir(pathFldBasisCheck) if '.' in item and os.path.splitext(item)[1][1:] in extsImg])
    for fnImg in fnsImgs:
        #
        # inform
        print('... checking of {:}'.format(fnImg))
        #
        # load image information and dataBasic
        nr, nc = cv2.imread(os.path.join(pathFldBasisCheck, fnImg)).shape[0:2]
        oca, ora = (nc - 1) / 2, (nr - 1) / 2
        #
        # load GCPs
        pathCdgTxt = os.path.join(pathFldBasisCheck, '{:}cdg.txt'.format(os.path.splitext(fnImg)[0]))
        cs, rs, xs, ys, zs = ulises.ReadCdgTxt(pathCdgTxt, options={'readCodes':False, 'readOnlyGood':True})[0:5]
        #
        # obtain good points
        possGood = ulises.RANSACForGCPs(cs, rs, xs, ys, zs, oca, ora, eRANSAC, pRANSAC, ecRANSAC, NForRANSACMax, options={'nOfK1asa2':1000})[0]
        #
        # inform
        if possGood is None:
            print('... too few GCPs to be checked')
        elif len(possGood) < len(cs):
            print('... re-run or consider to ignore the following GCPs')
            for pos in [item for item in range(len(cs)) if item not in possGood]:
                print('... c = {:8.2f}, r = {:8.2f}, x = {:8.2f}, y = {:8.2f}, z = {:8.2f}'.format(cs[pos], rs[pos], xs[pos], ys[pos], zs[pos]))
        else:
            print('... all the GCPs for {:} are OK'.format(fnImg))
    #
    return None
#
