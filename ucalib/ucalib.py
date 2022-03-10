#'''
# Created on 2021 by Gonzalo Simarro and Daniel Calvete
#'''
#
import cv2
import numpy as np
import os
from scipy import optimize
import shutil
#
import ulises_ucalib as ulises
#
def CalibrationOfBasisImages(pathBasis, errorTCritical, model, verbosePlot):
    #
    # manage model
    model2SelectedVariablesKeys = {'parabolic':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'sca'], 'quartic':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'sca'], 'full':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']}
    if model not in model2SelectedVariablesKeys.keys():
        print('*** Invalid calibration model {:}'.format(model))
        print('*** Choose one of the following calibration models: {:}'.format(list(model2SelectedVariablesKeys.keys()))); exit()
    #
    # obtain calibrations
    fnsImages = sorted([item for item in os.listdir(pathBasis) if '.' in item and item[item.rfind('.')+1:] in ['jpeg', 'JPEG', 'jpg', 'JPG', 'png', 'PNG']])
    for posFnImage, fnImage in enumerate(fnsImages):
        #
        print('... calibration of {:}'.format(fnImage), end='', flush=True)
        #
        # load image information and dataBasic
        if posFnImage == 0:
            nr, nc = cv2.imread(pathBasis + os.sep + fnImage).shape[0:2]
            dataBasic = ulises.LoadDataBasic0(options={'nc':nc, 'nr':nr, 'selectedVariablesKeys':model2SelectedVariablesKeys[model]})
        else:
            assert cv2.imread(pathBasis + os.sep + fnImage).shape[0:2] == (nr, nc)
        #
        # load GCPs
        pathCdgTxt = pathBasis + os.sep + fnImage[0:fnImage.rfind('.')] + 'cdg.txt'
        cs, rs, xs, ys, zs = ulises.ReadCdgTxt(pathCdgTxt, options={'readCodes':False, 'readOnlyGood':True})[0:5]
        #
        # load horizon points
        pathCdhTxt = pathBasis + os.sep + fnImage[0:fnImage.rfind('.')] + 'cdh.txt'
        if os.path.exists(pathCdhTxt):
            chs, rhs = ulises.ReadCdhTxt(pathCdhTxt, options={'readOnlyGood':True})
        else:
            chs, rhs = [np.asarray([]) for item in range(2)]
        #
        # load dataForCal and obtain calibration (aG, aH, mainSetSeeds are in dataForCal)
        dataForCal = {'nc':nc, 'nr':nr, 'cs':cs, 'rs':rs, 'xs':xs, 'ys':ys, 'zs':zs, 'aG':1., 'mainSetSeeds':[]} # IMP* to initialize mainSetSeeds
        if len(chs) == len(rhs) > 0:
            dataForCal['chs'], dataForCal['rhs'], dataForCal['aH'] = chs, rhs, 1.
        for filename in [item for item in os.listdir(pathBasis) if 'cal' in item and item[-3:] == 'txt']:
            allVariablesH, ncH, nrH = ulises.ReadCalTxt(pathBasis + os.sep + filename)[0:3]
            dataForCal['mainSetSeeds'].append(ulises.AllVariables2MainSet(allVariablesH, ncH, nrH, options={}))
        subsetVariablesKeys, subCsetVariablesDictionary = model2SelectedVariablesKeys[model], {}
        mainSet, errorT = ulises.NonlinearManualCalibration(dataBasic, dataForCal, subsetVariablesKeys, subCsetVariablesDictionary, options={})
        #
        # inform and write
        if errorT <= 1. * errorTCritical:
            print(' success')
            # check errorsG
            csR, rsR = ulises.XYZ2CDRD(mainSet, xs, ys, zs, options={})[0:2]
            errorsG = np.sqrt((csR - cs) ** 2 + (rsR - rs) ** 2)
            for pos in range(len(errorsG)):
                if errorsG[pos] > errorTCritical:
                    print('*** the error of GCP at x = {:8.2f}, y = {:8.2f} and z = {:8.2f} is {:5.1f} > critical error = {:5.1f}: consider to check or remove it'.format(xs[pos], ys[pos], zs[pos], errorsG[pos], errorTCritical))
            # write pathCal0Txt
            pathCal0Txt = pathBasis + os.sep + fnImage[0:fnImage.rfind('.')] + 'cal0.txt'
            ulises.WriteCalTxt(pathCal0Txt, mainSet['allVariables'], mainSet['nc'], mainSet['nr'], errorT)
            # manage verbosePlot
            if verbosePlot:
                pathTMP = pathBasis + os.sep + '..' + os.sep + 'TMP'
                ulises.MakeFolder(pathTMP)
                ulises.PlotMainSet(pathBasis + os.sep + fnImage, mainSet, cs, rs, xs, ys, zs, chs, rhs, pathTMP + os.sep + fnImage.replace('.', 'cal0_check.'))
        else:
            print(' failed (error = {:6.1f})'.format(errorT))
            print('*** re-run and, if it keeps failing, check quality of the GCP or try another calibration model ***')
    #
    return None
#
def CalibrationOfBasisImagesConstantXYZAndIntrinsic(pathBasis, model, verbosePlot):
    #
    # manage model
    model2SelectedVariablesKeys = {'parabolic':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'sca'], 'quartic':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'sca'], 'full':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']}
    if model not in model2SelectedVariablesKeys.keys():
        print('*** Invalid calibration model {:}'.format(model))
        print('*** Choose one of the following calibration models: {:}'.format(list(model2SelectedVariablesKeys.keys()))); exit()
    #
    # load basis information
    ncs, nrs, css, rss, xss, yss, zss, chss, rhss, allVariabless, mainSets, errorTs, fnsImages = [[] for item in range(13)]
    potentialFnsImages = sorted([item for item in os.listdir(pathBasis) if '.' in item and item[item.rfind('.')+1:] in ['jpeg', 'JPEG', 'jpg', 'JPG', 'png', 'PNG']])
    for posFnImage, fnImage in enumerate(potentialFnsImages):
        #
        # check pathCal0Txt
        pathCal0Txt = pathBasis + os.sep + fnImage[0:fnImage.rfind('.')] + 'cal0.txt'
        if not os.path.exists(pathCal0Txt):
            print('... image {:} ignored: no initial calibration available'.format(fnImage)); continue
        fnsImages.append(fnImage)
        #
        # load image information and dataBasic
        if posFnImage == 0:
            nr, nc = cv2.imread(pathBasis + os.sep + fnImage).shape[0:2]
            dataBasic = ulises.LoadDataBasic0(options={'nc':nc, 'nr':nr, 'selectedVariablesKeys':model2SelectedVariablesKeys[model]})
        else:
            assert cv2.imread(pathBasis + os.sep + fnImage).shape[0:2] == (nr, nc)
        ncs.append(nc); nrs.append(nr)
        #
        # load GCPs
        pathCdgTxt = pathBasis + os.sep + fnImage[0:fnImage.rfind('.')] + 'cdg.txt'
        cs, rs, xs, ys, zs = ulises.ReadCdgTxt(pathCdgTxt, options={'readCodes':False, 'readOnlyGood':True})[0:5]
        css.append(cs); rss.append(rs); xss.append(xs); yss.append(ys); zss.append(zs)
        #
        # load horizon points
        pathCdhTxt = pathBasis + os.sep + fnImage[0:fnImage.rfind('.')] + 'cdh.txt'
        if os.path.exists(pathCdhTxt):
            chs, rhs = ulises.ReadCdhTxt(pathCdhTxt, options={'readOnlyGood':True})
        else:
            chs, rhs = [np.asarray([]) for item in range(2)]
        chss.append(chs); rhss.append(rhs)
        #
        # load allVariables, mainSet and errorT
        allVariables, nc, nr, errorT = ulises.ReadCalTxt(pathCal0Txt)
        mainSet = ulises.AllVariables2MainSet(allVariables, nc, nr, options={})
        allVariabless.append(allVariables); mainSets.append(mainSet); errorTs.append(errorT)
    #
    # obtain calibrations and write pathCalTxts forcing unique xc, yc, zc and intrinsic
    if len(fnsImages) == 0:
        print('*** no initial calibrations available'); exit()
    elif len(fnsImages) == 1:
        pathCal0Txt = pathBasis + os.sep + fnsImages[0][0:fnsImages[0].rfind('.')] + 'cal0.txt'
        pathCalTxt = pathBasis + os.sep + fnsImages[0][0:fnsImages[0].rfind('.')] + 'cal.txt'
        shutil.copyfile(pathCal0Txt, pathCalTxt)
    else:
        subsetVariabless, subsetVariablesKeys = [], ['ph', 'sg', 'ta']
        subCsetVariabless, subCsetVariablesKeys = [], [item for item in model2SelectedVariablesKeys[model] if item not in subsetVariablesKeys]
        for pos in range(len(fnsImages)):
            subsetVariabless.append(ulises.AllVariables2SubsetVariables(dataBasic, allVariabless[pos], subsetVariablesKeys, options={}))
            subCsetVariabless.append(ulises.AllVariables2SubsetVariables(dataBasic, allVariabless[pos], subCsetVariablesKeys, options={}))
        mainSets, errorTs = ulises.NonlinearManualCalibrationForcingUniqueSubCset(dataBasic, ncs, nrs, css, rss, xss, yss, zss, chss, rhss, subsetVariabless, subsetVariablesKeys, subCsetVariabless, subCsetVariablesKeys, options={'aG':1., 'aH':1.}) # IMP* aG and aH
        for pos in range(len(fnsImages)):
            ulises.WriteCalTxt(pathBasis + os.sep + fnsImages[pos][0:fnsImages[pos].rfind('.')] + 'cal.txt', mainSets[pos]['allVariables'], mainSets[pos]['nc'], mainSets[pos]['nr'], errorTs[pos])
    #
    # manage verbosePlot
    if verbosePlot:
        pathTMP = pathBasis + os.sep + '..' + os.sep + 'TMP'
        ulises.MakeFolder(pathTMP)
        for pos in range(len(fnsImages)):
            ulises.PlotMainSet(pathBasis + os.sep + fnsImages[pos], mainSets[pos], css[pos], rss[pos], xss[pos], yss[pos], zss[pos], chss[pos], rhss[pos], pathTMP + os.sep + fnsImages[pos].replace('.', 'cal_check.'))
    #
    return None
#
def AutoCalibrationOfImages(pathBasis, pathImages, nOfFeaturesORB, fC, KC, verbosePlot):
    #
    # manage verbosePlot
    if verbosePlot:
        pathTMP = pathImages + os.sep + '..' + os.sep + 'TMP'
        ulises.MakeFolder(pathTMP)
        xs, ys, zs = [np.asarray([]) for item in range(3)]
        for fnCdgTxt in [item for item in os.listdir(pathBasis) if 'cdg' in item and item[-3:] == 'txt']:
            xsH, ysH, zsH = ulises.ReadCdgTxt(pathBasis + os.sep + fnCdgTxt, options={})[2:5]
            xs, ys, zs = np.concatenate((xs, xsH)), np.concatenate((ys, ysH)), np.concatenate((zs, zsH))
        cs, rs = [-999. * np.ones(xs.shape) for item in range(2)]
    #
    # perform ORB for basis
    fnsBasisImages, ncs, nrs, kpss, dess = ulises.ORBKeypointsForAllImagesInAFolder(pathBasis, options={'nOfFeatures':nOfFeaturesORB})
    #
    # load mainSets and Hs for basis (for those successful from ORB)
    mainSets, Hs = [[] for item in range(2)]
    for posFnBasisImage, fnBasisImage in enumerate(fnsBasisImages):
        pathCalTxt = pathBasis + os.sep + fnBasisImage[0:fnBasisImage.rfind('.')] + 'cal.txt'
        allVariables, nc, nr = ulises.ReadCalTxt(pathCalTxt)[0:3]
        mainSets.append(ulises.AllVariables2MainSet(allVariables, nc, nr, options={}))
        Hs.append(np.dot(mainSets[0]['R'], np.transpose(mainSets[posFnBasisImage]['R']))) # posFnBasisImage -> posFnBasisImage=0 (first image of the basis)
    #
    # obtain calibrations and write pathCalTxts
    fnsImages = sorted([item for item in os.listdir(pathImages) if '.' in item and item[item.rfind('.')+1:] in ['jpeg', 'JPEG', 'jpg', 'JPG', 'png', 'PNG']])
    for fnImage in fnsImages:
        #
        print('... autocalibration of {:}'.format(fnImage), end='', flush=True)
        #
        # obtain pathCalTxt and check if already exists
        pathCalTxt = pathImages + os.sep + fnImage[0:fnImage.rfind('.')] + 'cal.txt'
        if os.path.exists(pathCalTxt):
            print(' already calibrated'); continue
        #
        # perform ORB
        nc0, nr0, kps0, des0, ctrl0 = ulises.ORBKeypoints(pathImages + os.sep + fnImage, {'nOfFeatures':nOfFeaturesORB})
        if not ctrl0:
            print(' failed (ORB failure)'); continue
        if not (nc0 == nc and nr0 == nr):
            print(' failed (image with wrong shape!)'); continue
        #
        # find pairs with basis images in the space (u, v), F = Full (all basis images)
        uUas0F, vUas0F = [np.asarray([]) for item in range(2)] # image to calibrate
        uUas1F, vUas1F = [np.asarray([]) for item in range(2)] # first image of the basis
        for posFnBasisImage, fnBasisImage in enumerate(fnsBasisImages):
            # find matches
            cDs0, rDs0, cDsB, rDsB, ersB = ulises.ORBMatches(kps0, des0, kpss[posFnBasisImage], dess[posFnBasisImage], options={'erMaximum':30., 'nOfStd':2.0})
            if len(cDs0) == 0:
                continue
            # update 0, the image to calibrate (recall that intrinsic is constant, we use mainSets[0])
            uDas0, vDas0 = ulises.CR2UaVa(mainSets[0], cDs0, rDs0)[0:2]
            uUas0, vUas0 = ulises.UDaVDa2UUaVUa(mainSets[0], uDas0, vDas0)
            uUas0F, vUas0F = np.concatenate((uUas0F, uUas0)), np.concatenate((vUas0F, vUas0))
            # update 1, in the first image of the basis (recall that intrinsic is constant, we use mainSets[0])
            uDasB, vDasB = ulises.CR2UaVa(mainSets[0], cDsB, rDsB)[0:2]
            uUasB, vUasB = ulises.UDaVDa2UUaVUa(mainSets[0], uDasB, vDasB)
            uUas1, vUas1 = ulises.ApplyHomographyHa01(Hs[posFnBasisImage], uUasB, vUasB)
            uUas1F, vUas1F = np.concatenate((uUas1F, uUas1)), np.concatenate((vUas1F, vUas1))
        if len(uUas0F) < min([4, KC]):
            print(' failed (K = {:}) after ORB'.format(len(uUas0F))); continue
        #
        # apply RANSAC in the pixel domain (parametersRANSAC['errorC'] in pixels)
        cUs0F, rUs0F = ulises.UaVa2CR(mainSets[0], uUas0F, vUas0F)[0:2] # image to calibrate
        cUs1F, rUs1F = ulises.UaVa2CR(mainSets[0], uUas1F, vUas1F)[0:2] # first image of the basis
        parametersRANSAC = {'e':0.8, 's':4, 'p':0.999999, 'errorC':2.} # should be fine (on the safe side)
        Ha01, possGood = ulises.FindHomographyHa01ViaRANSAC(cUs0F, rUs0F, cUs1F, rUs1F, parametersRANSAC)
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
        # manage verbosePlot
        if verbosePlot:
            allVariables, nc, nr = ulises.ReadCalTxt(pathCalTxt)[0:3]
            mainSet = ulises.AllVariables2MainSet(allVariables, nc, nr, options={})
            ulises.PlotMainSet(pathImages + os.sep + fnImage, mainSet, cs, rs, xs, ys, zs, None, None, pathTMP + os.sep + fnImage.replace('.', 'cal_check.'))
    #
    return None
#
def PlanviewsFromImages(pathImages, pathPlanviews, z0, ppm, verbosePlot):
    #
    # obtain the planview domain from the cloud of points
    if not os.path.exists(pathPlanviews):
        print('*** folder {:} not found'.format(pathPlanviews)); exit()
    if not os.path.exists(pathPlanviews + os.sep + 'xy_planview.txt'):
        print('*** file xy_planview.txt not found in {:}'.format(pathPlanviews)); exit()
    rawData = np.asarray(ulises.ReadRectangleFromTxt(pathPlanviews + os.sep + 'xy_planview.txt', options={'c1':2, 'valueType':'float'}))
    xsCloud, ysCloud = rawData[:, 0], rawData[:, 1]
    angle, xUL, yUL, H, W = ulises.Cloud2Rectangle(xsCloud, ysCloud)
    dataPdfTxt = ulises.LoadDataPdfTxt(options={'xUpperLeft':xUL, 'yUpperLeft':yUL, 'angle':angle, 'xYLengthInC':W, 'xYLengthInR':H, 'ppm':ppm})
    csCloud, rsCloud = ulises.XY2CR(dataPdfTxt, xsCloud, ysCloud)[0:2] # only useful if verbosePlot
    #
    # write the planview domain
    fileout = open(pathPlanviews + os.sep + 'crxyz_planview.txt', 'w')
    for pos in range(4):
        fileout.write('{:6.0f} {:6.0f} {:8.2f} {:8.2f} {:8.2f}\t c, r, x, y and z\n'.format(dataPdfTxt['csC'][pos], dataPdfTxt['rsC'][pos], dataPdfTxt['xsC'][pos], dataPdfTxt['ysC'][pos], z0))
    fileout.close()
    #
    # obtain and write planviews
    fnsImages = sorted([item for item in os.listdir(pathImages) if '.' in item and item[item.rfind('.')+1:] in ['jpeg', 'JPEG', 'jpg', 'JPG', 'png', 'PNG']])
    for fnImage in fnsImages:
        #
        print('... planview of {:}'.format(fnImage), end='', flush=True)
        #
        # obtain pathPlw and check if already exists
        pathPlw = pathPlanviews + os.sep + fnImage.replace('.', 'plw.')
        if os.path.exists(pathPlw):
            print(' already exists'); continue
        #
        # load calibration and obtain and write planview
        pathCalTxt = pathImages + os.sep + fnImage[0:fnImage.rfind('.')] + 'cal.txt'
        if os.path.exists(pathCalTxt):
            # load calibration
            allVariables, nc, nr, errorT = ulises.ReadCalTxt(pathCalTxt)
            mainSet = ulises.AllVariables2MainSet(allVariables, nc, nr, options={})
            # obtain and write planview
            imgPlanview = ulises.CreatePlanview(ulises.PlanviewPrecomputations({'01':mainSet}, dataPdfTxt, z0), {'01':cv2.imread(pathImages + os.sep + fnImage)})
            cv2.imwrite(pathPlw, imgPlanview)
            print(' success')
            # manage verbosePlot
            if verbosePlot:
                pathTMP = pathPlanviews + os.sep + '..' + os.sep + 'TMP'
                ulises.MakeFolder(pathTMP)
                imgTMP = ulises.DisplayCRInImage(imgPlanview, csCloud, rsCloud, options={'colors':[[0, 255, 255]], 'size':10})
                cv2.imwrite(pathTMP + os.sep + fnImage.replace('.', 'plw_check.'), imgTMP)
                cs, rs = ulises.XYZ2CDRD(mainSet, xsCloud, ysCloud, z0)[0:2]
                img = cv2.imread(pathImages + os.sep + fnImage)
                imgTMP = ulises.DisplayCRInImage(img, cs, rs, options={'colors':[[0, 255, 255]], 'size':10})
                cv2.imwrite(pathTMP + os.sep + fnImage.replace('.', '_checkplw.'), imgTMP)
        else:
            print(' failed')
    #
    return None
#
def CheckGCPs(pathBasisCheck, errorCritical):
    #
    eRANSAC, pRANSAC, ecRANSAC, NForRANSACMax = 0.8, 0.999999, errorCritical, 50000
    #
    # check GCPs
    fnsImages = sorted([item for item in os.listdir(pathBasisCheck) if '.' in item and item[item.rfind('.')+1:] in ['jpeg', 'JPEG', 'jpg', 'JPG', 'png', 'PNG']])
    for posFnImage, fnImage in enumerate(fnsImages):
        #
        print('... checking of {:}'.format(fnImage))
        #
        # load image information and dataBasic
        nr, nc = cv2.imread(pathBasisCheck + os.sep + fnImage).shape[0:2]
        oca, ora = (nc-1)/2, (nr-1)/2
        #
        # load GCPs
        pathCdgTxt = pathBasisCheck + os.sep + fnImage[0:fnImage.rfind('.')] + 'cdg.txt'
        cs, rs, xs, ys, zs = ulises.ReadCdgTxt(pathCdgTxt, options={'readCodes':False, 'readOnlyGood':True})[0:5]
        #
        possGood = ulises.RANSACForGCPs(cs, rs, xs, ys, zs, oca, ora, eRANSAC, pRANSAC, ecRANSAC, NForRANSACMax, options={'nOfK1asa2':1000})[0]
        #
        # inform
        if possGood is None:
            print('... too few GCPs to be checked')
        elif len(possGood) < len(cs):
            print('... re-run or consider to ignore the following GCPs')
            for pos in [item for item in range(len(cs)) if item not in possGood]:
                c, r, x, y, z = [item[pos] for item in [cs, rs, xs, ys, zs]]
                print('... c = {:8.2f}, r = {:8.2f}, x = {:8.2f}, y = {:8.2f}, z = {:8.2f}'.format(c, r, x, y, z))
        else:
            print('... all the GCPs for {:} are OK'.format(fnImage))
    #
    return None

