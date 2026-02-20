import os
import glob
import argparse
import numpy as np
import cv2
import ast
import scipy.stats

from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
from skimage.io import imread, imshow
from skimage.transform import resize
from matplotlib import pyplot as plt
from PIL import Image

from collections import Counter


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    sd = np.std(a)
    return [m, m-h, m+h, sd]

# input: the paths to two images and the list of results to append to
# output: the list of results with the agreement metric appended
def agreement(temp_test, temp_retest, val_listAgre, val_listFN, val_listFP, ignore_check=False):
    if ignore_check:
        img1 = imread(temp_test, as_gray=True)
        img1_shape = img1.shape
        img2 = imread(temp_retest, as_gray=True)
        img2 = resize(img2, (img1_shape[0], img1_shape[1]), anti_aliasing=False)
        
        # uncomment if you want to include all pixels except 0 in the agreement metric
        # img1 = np.ndarray.flatten(img1 / 255) > 0.0
        img2 = np.ndarray.flatten(img2 / 255) > 0.0
        
        # uncomment if you want to include only pixels with a value of 255 in the agreement metric
        img1 = np.ndarray.flatten(np.where(img1==255, True, False))
        # img2 = np.ndarray.flatten(np.where(img2==1.0, True, False))
        

        # print(Counter(img1*255))
        # img = Image.fromarray((img1 * 255).reshape(img1_shape[0], img1_shape[1]), 'L')
        # img.show()
        # stop

        black_area = np.logical_and(img1, img2)
        red_area = np.logical_xor(black_area, img1)
        green_area = np.logical_xor(black_area, img2)
        total_area = np.count_nonzero(red_area) + np.count_nonzero(green_area) + np.count_nonzero(black_area)
        reference_area = np.count_nonzero(red_area) + np.count_nonzero(black_area)
        test_area = np.count_nonzero(green_area) + np.count_nonzero(black_area)

        try:
            fn = ((total_area-test_area))/reference_area
        except:
            fn = 0
        try:
            fp = ((total_area-reference_area))/test_area
        except:
            fp = 0
        
        
        agreement = 1-((fn+fp)/2)
        if fp==0 and fn==0:
            agreement = 0
        return val_listAgre.append(agreement), val_listFN.append(fn), val_listFP.append(fp)
    
    if os.path.split(temp_test)[-1].split('.')[0].split('-')[1] == os.path.split(temp_retest)[-1].split('.')[0].split('-')[1]:
        img1 = imread(temp_test)
        img1_shape = img1.shape
        img2 = imread(temp_retest)
        img2 = resize(img2, (img1_shape[0], img1_shape[1]), anti_aliasing=False)
        # uncomment if you want to include all pixels except 0 in the agreement metric
        # img1 = np.ndarray.flatten(img1 / 255) > 0.0
        img2 = np.ndarray.flatten(img2 / 255) > 0.0
        
        # uncomment if you want to include only pixels with a value of 255 in the agreement metric
        img1 = np.ndarray.flatten(np.where(img1==255, True, False))
        # img2 = np.ndarray.flatten(np.where(img2==255, True, False))
        
        black_area = np.logical_and(img1, img2)
        red_area = np.logical_xor(black_area, img1)
        green_area = np.logical_xor(black_area, img2)
        total_area = np.count_nonzero(red_area) + np.count_nonzero(green_area) + np.count_nonzero(black_area)
        reference_area = np.count_nonzero(red_area) + np.count_nonzero(black_area)
        test_area = np.count_nonzero(green_area) + np.count_nonzero(black_area)

        fn = ((total_area-test_area))/reference_area
        fp = ((total_area-reference_area))/test_area

        agreement = 1-(fn+fp)/2
        # print('fn: ', fn)
        # print('fp: ', fp)
        # print('agreement: ', agreement)
        return val_listAgre.append(agreement), val_listFN.append(fn), val_listFP.append(fp)
    else:
        print('WARNING: test and retest images do not match')


def eval(args):

    isConfInter = args.conf_intervals

    # check if args.ofr_gt_masks and args.mask_root file locations exist
    if not os.path.exists(args.ofr_gt_masks):
        print('ERROR: OFR ground truth masks directory does not exist')
        print(args.ofr_gt_masks)
        return
    if not os.path.exists(args.mask_root):
        print('ERROR: OFR ground truth masks directory does not exist')
        print(args.mask_root)
        return


    prediction_files = sorted(glob.glob(os.path.join(args.mask_root, '*')), key=str.casefold)


    # split pred files into test and retest globs, compare the test and retest imanges to calculate  adding the metric to their own list of info for each methods of OC
    # pred_test, pred_retest = [], []
    # for pred_itm in prediction_files:
    #     if "Rd" in os.path.split(pred_itm)[-1]:
    #         pred_retest.append(pred_itm)
    #     elif "Td" in os.path.split(pred_itm)[-1]:
    #         pred_test.append(pred_itm)

    #filters per patient
    # ap12,ap40,ap100,ap200,iso100,iso200,iso300,iso400,tscan100,ofc,ofr = [],[],[],[],[],[],[],[],[],[],[]
    # i = 1
    # while [s for s in pred_test if str(i) in s.split('\\')[-1].split('.')[0].split('-')[0].split('Td')[0]]:
    #     # filters test and retest
    #     temp_test = [s for s in pred_test if str(i) in s.split('\\')[-1].split('.')[0].split('-')[0].split('Td')[0]]
    #     temp_retest = [s for s in pred_retest if str(i) in s.split('\\')[-1].split('.')[0].split('-')[0].split('Rd')[0]]

    #     #compares test and retest
    #     # print('AP12A')    
    #     agreement(temp_test[0], temp_retest[0], ap12)
    #     # print('AP12P')
    #     agreement(temp_test[1], temp_retest[1], ap12)
    #     # print('AP40A')
    #     agreement(temp_test[2], temp_retest[2], ap40)
    #     # print('AP40P')
    #     agreement(temp_test[3], temp_retest[3], ap40)
    #     # print('AP100A')
    #     agreement(temp_test[4], temp_retest[4], ap100)
    #     # print('AP100P')
    #     agreement(temp_test[5], temp_retest[5], ap100)
    #     # print('AP200A')
    #     agreement(temp_test[6], temp_retest[6], ap200)
    #     # print('AP200P')
    #     agreement(temp_test[7], temp_retest[7], ap200)
    #     # print('ISO100')
    #     agreement(temp_test[8], temp_retest[8], iso100)
    #     # print('ISO200')
    #     agreement(temp_test[9], temp_retest[9], iso200)
    #     # print('ISO300')
    #     agreement(temp_test[10], temp_retest[10], iso300)
    #     # print('ISO400')
    #     agreement(temp_test[11], temp_retest[11], iso400)
    #     # print('OFC40')
    #     agreement(temp_test[12], temp_retest[12], ofc)
    #     # print('OFR200')
    #     agreement(temp_test[13], temp_retest[13], ofr)
    #     # print('TSCAN no sen')
    #     agreement(temp_test[14], temp_retest[14], tscan100)
    #     # print('TSCAN sen')
    #     agreement(temp_test[15], temp_retest[15], tscan100)


    #     i += 1
    
    # print('################################# REPRODUCIBILITY #################################')
    # print('AP12: ', np.mean(ap12))
    # print('AP40: ', np.mean(ap40))
    # print('AP100: ', np.mean(ap100))
    # print('AP200: ', np.mean(ap200))
    # print('ISO100: ', np.mean(iso100))
    # print('ISO200: ', np.mean(iso200))
    # print('ISO300: ', np.mean(iso300))
    # print('ISO400: ', np.mean(iso400))
    # print('TSCAN100: ', np.mean(tscan100))
    # print('OFC: ', np.mean(ofc))
    # print('OFR: ', np.mean(ofr))



    print('################################# VALIDITY #################################')

    ofr_files = sorted(glob.glob(os.path.join(args.ofr_gt_masks, '*')), key=str.casefold)

    # take the pred files and compare each one to every OFR iamge of the same patient and test/retest, calculate the metrics and add them to a list of info for each method of OC

    # splits both datasets into test and retest
    pred_test, pred_retest = [], []
    for pred_itm in prediction_files:
        if "Rd" in os.path.split(pred_itm)[-1]:
            pred_retest.append(pred_itm)
        else:
            pred_test.append(pred_itm)

    ofr_test, ofr_retest = [], []
    for ofr_itm in ofr_files:
        if "Rd" in os.path.split(ofr_itm)[-1]:
            ofr_retest.append(ofr_itm)
        else:
            ofr_test.append(ofr_itm)

    # traverses the types of ofr
    for l in ast.literal_eval(args.list_of_ofr):
        #filter ofr types
        ofr_retest_filtered = [s for s in ofr_retest if l in os.path.split(s)[-1].split('.')[0].split('-')[1]]
        ofr_test_filtered = [s for s in ofr_test if l in os.path.split(s)[-1].split('.')[0].split('-')[1]]
        # gets the number of patients in the ofr set
        ofr_patients = [os.path.split(s)[-1].lower().split('.')[0].split('-')[0].split('rd')[0].split('z')[1] for s in ofr_retest_filtered]
        print(ofr_patients)
        # stop

        # print(ofr_test_filtered)
        #traverses the patient for the type of ofr
        ap12,ap40,ap100,ap200,iso100,iso200,iso300,iso400,tscan100,ofc = [],[],[],[],[],[],[],[],[],[]
        ap12FP,ap40FP,ap100FP,ap200FP,iso100FP,iso200FP,iso300FP,iso400FP,tscan100FP,ofcFP = [],[],[],[],[],[],[],[],[],[]
        ap12FN,ap40FN,ap100FN,ap200FN,iso100FN,iso200FN,iso300FN,iso400FN,tscan100FN,ofcFN = [],[],[],[],[],[],[],[],[],[]
        
        for pat in ofr_patients:
            cur_ofr_test_filtered = [s for s in ofr_test_filtered if pat in os.path.split(s)[-1].split('.')[0].split('-')[0].split('Td')[0].split('Tf')[0]]
            cur_ofr_retest_filtered = [s for s in ofr_retest_filtered if pat in os.path.split(s)[-1].split('.')[0].split('-')[0].split('Rd')[0]]
            # print(cur_ofr_test_filtered)
            # print(cur_ofr_retest_filtered)
            
            #filter pred test and retest by patient
            cur_pred_test_filtered = [s for s in pred_test if pat in os.path.split(s)[-1].split('.')[0].split('-')[0].split('Td')[0]]
            cur_pred_retest_filtered = [s for s in pred_retest if pat in os.path.split(s)[-1].split('.')[0].split('-')[0].split('Rd')[0]]
            # print(cur_pred_test_filtered)
            # print(cur_pred_retest_filtered)

            # removes OFR from predicted images
            cur_pred_test_filtered = [s for s in cur_pred_test_filtered if 'ofr' not in os.path.split(s)[-1].lower().split('.')[0].split('-')[1]]
            cur_pred_retest_filtered = [s for s in cur_pred_retest_filtered if 'ofr' not in os.path.split(s)[-1].lower().split('.')[0].split('-')[1]]

            #calculate and append the agreement metrics for each pred type
            # print(cur_ofr_retest_filtered)
            # print(cur_ofr_test_filtered)
            # print(len(cur_ofr_retest_filtered))
            # print(len(cur_ofr_test_filtered))
            # print(len(ap12))
            # print(len(ap12FN))
            # print(len(ap12FP))
            # print('#########')
            agreement(cur_pred_test_filtered[0], cur_ofr_test_filtered[0], ap12, ap12FN, ap12FP, ignore_check=True)
            agreement(cur_pred_retest_filtered[0], cur_ofr_retest_filtered[0], ap12, ap12FN, ap12FP, ignore_check=True)
            agreement(cur_pred_test_filtered[1], cur_ofr_test_filtered[0], ap12, ap12FN, ap12FP, ignore_check=True)
            agreement(cur_pred_retest_filtered[1], cur_ofr_retest_filtered[0], ap12, ap12FN, ap12FP, ignore_check=True)
            agreement(cur_pred_test_filtered[2], cur_ofr_test_filtered[0], ap40, ap40FN, ap40FP, ignore_check=True)
            agreement(cur_pred_retest_filtered[2], cur_ofr_retest_filtered[0], ap40, ap40FN, ap40FP, ignore_check=True)
            agreement(cur_pred_test_filtered[3], cur_ofr_test_filtered[0], ap40, ap40FN, ap40FP, ignore_check=True)
            agreement(cur_pred_retest_filtered[3], cur_ofr_retest_filtered[0], ap40, ap40FN, ap40FP, ignore_check=True)
            agreement(cur_pred_test_filtered[4], cur_ofr_test_filtered[0], ap100, ap100FN, ap100FP, ignore_check=True)
            agreement(cur_pred_retest_filtered[4], cur_ofr_retest_filtered[0], ap100, ap100FN, ap100FP, ignore_check=True)
            agreement(cur_pred_test_filtered[5], cur_ofr_test_filtered[0], ap100, ap100FN, ap100FP, ignore_check=True)
            agreement(cur_pred_retest_filtered[5], cur_ofr_retest_filtered[0], ap100, ap100FN, ap100FP, ignore_check=True)
            agreement(cur_pred_test_filtered[6], cur_ofr_test_filtered[0], ap200, ap200FN, ap200FP, ignore_check=True)
            agreement(cur_pred_retest_filtered[6], cur_ofr_retest_filtered[0], ap200, ap200FN, ap200FP, ignore_check=True)
            agreement(cur_pred_test_filtered[7], cur_ofr_test_filtered[0], ap200, ap200FN, ap200FP, ignore_check=True)
            agreement(cur_pred_retest_filtered[7], cur_ofr_retest_filtered[0], ap200, ap200FN, ap200FP, ignore_check=True)
            if args.ap_only == 'False':
                agreement(cur_pred_test_filtered[8], cur_ofr_test_filtered[0], iso100, iso100FN, iso100FP, ignore_check=True)
                agreement(cur_pred_retest_filtered[8], cur_ofr_retest_filtered[0], iso100, iso100FN, iso100FP, ignore_check=True)
                agreement(cur_pred_test_filtered[9], cur_ofr_test_filtered[0], iso200, iso200FN, iso200FP, ignore_check=True)
                agreement(cur_pred_retest_filtered[9], cur_ofr_retest_filtered[0], iso200, iso200FN, iso200FP, ignore_check=True)
                agreement(cur_pred_test_filtered[10], cur_ofr_test_filtered[0], iso300, iso300FN, iso300FP, ignore_check=True)
                agreement(cur_pred_retest_filtered[10], cur_ofr_retest_filtered[0], iso300, iso300FN, iso300FP, ignore_check=True)
                agreement(cur_pred_test_filtered[11], cur_ofr_test_filtered[0], iso400, iso400FN, iso400FP, ignore_check=True)
                agreement(cur_pred_retest_filtered[11], cur_ofr_retest_filtered[0], iso400, iso400FN, iso400FP, ignore_check=True)
                agreement(cur_pred_test_filtered[12], cur_ofr_test_filtered[0], ofc, ofcFN, ofcFP, ignore_check=True)
                agreement(cur_pred_retest_filtered[12], cur_ofr_retest_filtered[0], ofc, ofcFN, ofcFP, ignore_check=True)
                agreement(cur_pred_test_filtered[13], cur_ofr_test_filtered[0], tscan100, tscan100FN, tscan100FP, ignore_check=True)
                agreement(cur_pred_retest_filtered[13], cur_ofr_retest_filtered[0], tscan100, tscan100FN, tscan100FP, ignore_check=True)
                agreement(cur_pred_test_filtered[14], cur_ofr_test_filtered[0], tscan100, tscan100FN, tscan100FP, ignore_check=True)
                agreement(cur_pred_retest_filtered[14], cur_ofr_retest_filtered[0], tscan100, tscan100FN, tscan100FP, ignore_check=True)


            
        print('############ OFR: ', l, ' ############')
        # print(ap12)
        # print(ap40)
        # print(ap100)
        # print(ap200)
        # print(iso100)
        # print(iso200)
        # print(iso300)
        # print(iso400)
        # print(tscan100)
        # print(ofc)
        # # print(ofr)
        # print('############')
        print('###### Agreement ######')
        # print(ap12)
        if args.ap_only == 'False':
            ofcm = mean_confidence_interval(ofc)
            iso100m = mean_confidence_interval(iso100)
            iso200m = mean_confidence_interval(iso200)
            iso300m = mean_confidence_interval(iso300)
            iso400m = mean_confidence_interval(iso400)
            tscan100m = mean_confidence_interval(tscan100)

            if isConfInter:
                print('OFC: ', ofcm[0], '('+ str(ofcm[1]) + '-'+ str(ofcm[2]), ')')
                print('IOS100: ', iso100m[0], '('+ str(iso100m[1]) + '-'+ str(iso100m[2]), ')')
                print('IOS200: ', iso200m[0], '('+ str(iso200m[1]) + '-'+ str(iso200m[2]), ')')
                print('IOS300: ', iso300m[0], '('+ str(iso300m[1]) + '-'+ str(iso300m[2]), ')')
                print('IOS400: ', iso400m[0], '('+ str(iso400m[1]) + '-'+ str(iso400m[2]), ')')
                print('TSCAN100: ', tscan100m[0], '('+ str(tscan100m[1]) + '-'+ str(tscan100m[2]), ')')
            else:
                print('OFC: ', ofcm[0], '($\pm '+ str(ofcm[3]),'$)')
                print('IOS100: ', iso100m[0], '($\pm '+ str(iso100m[3])+'$)')
                print('IOS200: ', iso200m[0], '($\pm '+ str(iso200m[3])+'$)')
                print('IOS300: ', iso300m[0], '($\pm '+ str(iso300m[3])+'$)')
                print('IOS400: ', iso400m[0], '($\pm '+ str(iso400m[3])+'$)')
                print('TSCAN100: ', tscan100m[0], '($\pm '+ str(tscan100m[3])+'$)')


        ap12m = mean_confidence_interval(ap12)
        ap40m = mean_confidence_interval(ap40)
        ap100m = mean_confidence_interval(ap100)
        ap200m = mean_confidence_interval(ap200)
        if isConfInter:
            print('AP12: ', ap12m[0], '('+ str(ap12m[1])+ '-'+ str(ap12m[2])+ ')')
            print('AP40: ', ap40m[0], '('+ str(ap40m[1]) + '-'+ str(ap40m[2])+ ')')
            print('AP100: ', ap100m[0], '('+ str(ap100m[1]) + '-'+ str(ap100m[2])+ ')')
            print('AP200: ', ap200m[0], '('+ str(ap200m[1]) + '-'+ str(ap200m[2])+ ')')
        else:
            print('AP12: ', ap12m[0], '($\pm '+ str(ap12m[3]),'$)')
            print('AP40: ', ap40m[0], '($\pm '+ str(ap40m[3])+'$)')
            print('AP100: ', ap100m[0], '($\pm '+ str(ap100m[3])+'$)')
            print('AP200: ', ap200m[0], '($\pm '+ str(ap200m[3])+'$)')

        print('ROUND')
        if isConfInter:
            print('AP12: ', ("%.1f" % ap12m[0]), '('+ str(("%.1f" % ap12m[1]))+ '-'+ str(("%.1f" % ap12m[2]))+ ')')
            print('AP40: ', ("%.1f" % ap40m[0]), '('+ str(("%.1f" % ap40m[1])) + '-'+ str(("%.1f" % ap40m[2]))+ ')')
            print('AP100: ', ("%.1f" % ap100m[0]), '('+ str(("%.1f" % ap100m[1])) + '-'+ str(("%.1f" % ap100m[2]))+ ')')
            print('AP200: ', ("%.1f" % ap200m[0]), '('+ str(("%.1f" % ap200m[1])) + '-'+ str(("%.1f" % ap200m[2]))+ ')')
        else:
            print('AP12: ', ("%.3f" % ap12m[0]), '($\pm '+ str("%.3f" % ap12m[3]),'$)')
            print('AP40: ', ("%.3f" % ap40m[0]), '($\pm '+ str("%.3f" % ap40m[3])+'$)')
            print('AP100: ', ("%.3f" % ap100m[0]), '($\pm '+ str("%.3f" % ap100m[3])+'$)')
            print('AP200: ', ("%.3f" % ap200m[0]), '($\pm '+ str("%.3f" % ap200m[3])+'$)')

        print('\n')
        print('\n')
        print('\n')

        # FP and FN are swapped to keep the metrics correct. 
        print('###### FP % ######')
        if args.ap_only == 'False':
            ofcm = mean_confidence_interval(ofcFN)
            iso100m = mean_confidence_interval(iso100FN)
            iso200m = mean_confidence_interval(iso200FN)
            iso300m = mean_confidence_interval(iso300FN)
            iso400m = mean_confidence_interval(iso400FN)
            tscan100m = mean_confidence_interval(tscan100FN)
            if isConfInter:
                print('OFC: ', ofcm[0], '('+ str(ofcm[1]) + '-'+ str(ofcm[2]), ')')
                print('IOS100: ', iso100m[0], '('+ str(iso100m[1]) + '-'+ str(iso100m[2]), ')')
                print('IOS200: ', iso200m[0], '('+ str(iso200m[1]) + '-'+ str(iso200m[2]), ')')
                print('IOS300: ', iso300m[0], '('+ str(iso300m[1]) + '-'+ str(iso300m[2]), ')')
                print('IOS400: ', iso400m[0], '('+ str(iso400m[1]) + '-'+ str(iso400m[2]), ')')
                print('TSCAN100: ', tscan100m[0], '('+ str(tscan100m[1]) + '-'+ str(tscan100m[2]), ')')
            else:
                print('OFC: ', ofcm[0], '($\pm '+ str(ofcm[3]),'$)')
                print('IOS100: ', iso100m[0], '($\pm '+ str(iso100m[3])+'$)')
                print('IOS200: ', iso200m[0], '($\pm '+ str(iso200m[3])+'$)')
                print('IOS300: ', iso300m[0], '($\pm '+ str(iso300m[3])+'$)')
                print('IOS400: ', iso400m[0], '($\pm '+ str(iso400m[3])+'$)')
                print('TSCAN100: ', tscan100m[0], '($\pm '+ str(tscan100m[3])+'$)')

        ap12m = mean_confidence_interval(ap12FN)
        ap40m = mean_confidence_interval(ap40FN)
        ap100m = mean_confidence_interval(ap100FN)
        ap200m = mean_confidence_interval(ap200FN)

        if isConfInter:
            print('AP12: ', ap12m[0], '('+ str(ap12m[1])+ '-'+ str(ap12m[2])+ ')')
            print('AP40: ', ap40m[0], '('+ str(ap40m[1]) + '-'+ str(ap40m[2])+ ')')
            print('AP100: ', ap100m[0], '('+ str(ap100m[1]) + '-'+ str(ap100m[2])+ ')')
            print('AP200: ', ap200m[0], '('+ str(ap200m[1]) + '-'+ str(ap200m[2])+ ')')

            print('ROUND')

            print('AP12: ', ("%.1f" % ap12m[0]), '('+ str(("%.1f" % ap12m[1]))+ '-'+ str(("%.1f" % ap12m[2]))+ ')')
            print('AP40: ', ("%.1f" % ap40m[0]), '('+ str(("%.1f" % ap40m[1])) + '-'+ str(("%.1f" % ap40m[2]))+ ')')
            print('AP100: ', ("%.1f" % ap100m[0]), '('+ str(("%.1f" % ap100m[1])) + '-'+ str(("%.1f" % ap100m[2]))+ ')')
            print('AP200: ', ("%.1f" % ap200m[0]), '('+ str(("%.1f" % ap200m[1])) + '-'+ str(("%.1f" % ap200m[2]))+ ')')
        else:
            print('AP12: ', ap12m[0], '($\pm '+ str(ap12m[3]),'$)')
            print('AP40: ', ap40m[0], '($\pm '+ str(ap40m[3])+'$)')
            print('AP100: ', ap100m[0], '($\pm '+ str(ap100m[3])+'$)')
            print('AP200: ', ap200m[0], '($\pm '+ str(ap200m[3])+'$)')

            print('ROUND')

            print('AP12: ', ("%.3f" % ap12m[0]), '($\pm '+ str("%.3f" % ap12m[3]),'$)')
            print('AP40: ', ("%.3f" % ap40m[0]), '($\pm '+ str("%.3f" % ap40m[3])+'$)')
            print('AP100: ', ("%.3f" % ap100m[0]), '($\pm '+ str("%.3f" % ap100m[3])+'$)')
            print('AP200: ', ("%.3f" % ap200m[0]), '($\pm '+ str("%.3f" % ap200m[3])+'$)')


        print('\n')
        print('\n')
        print('\n')

        # FP and FN are swapped to keep the metrics correct. 
        print('###### FN % ######')
        if args.ap_only == 'False':
            ofcm = mean_confidence_interval(ofcFP)
            iso100m = mean_confidence_interval(iso100FP)
            iso200m = mean_confidence_interval(iso200FP)
            iso300m = mean_confidence_interval(iso300FP)
            iso400m = mean_confidence_interval(iso400FP)
            tscan100m = mean_confidence_interval(tscan100FP)

            if isConfInter:
                print('OFC: ', ofcm[0], '('+ str(ofcm[1]) + '-'+ str(ofcm[2]), ')')
                print('IOS100: ', iso100m[0], '('+ str(iso100m[1]) + '-'+ str(iso100m[2]), ')')
                print('IOS200: ', iso200m[0], '('+ str(iso200m[1]) + '-'+ str(iso200m[2]), ')')
                print('IOS300: ', iso300m[0], '('+ str(iso300m[1]) + '-'+ str(iso300m[2]), ')')
                print('IOS400: ', iso400m[0], '('+ str(iso400m[1]) + '-'+ str(iso400m[2]), ')')
                print('TSCAN100: ', tscan100m[0], '('+ str(tscan100m[1]) + '-'+ str(tscan100m[2]), ')')
            else:
                print('OFC: ', ofcm[0], '($\pm '+ str(ofcm[3]),'$)')
                print('IOS100: ', iso100m[0], '($\pm '+ str(iso100m[3])+'$)')
                print('IOS200: ', iso200m[0], '($\pm '+ str(iso200m[3])+'$)')
                print('IOS300: ', iso300m[0], '($\pm '+ str(iso300m[3])+'$)')
                print('IOS400: ', iso400m[0], '($\pm '+ str(iso400m[3])+'$)')
                print('TSCAN100: ', tscan100m[0], '($\pm '+ str(tscan100m[3])+'$)')

        ap12m = mean_confidence_interval(ap12FP)
        ap40m = mean_confidence_interval(ap40FP)
        ap100m = mean_confidence_interval(ap100FP)
        ap200m = mean_confidence_interval(ap200FP)

        if isConfInter:
            print('AP12: ', ap12m[0], '('+ str(ap12m[1])+ '-'+ str(ap12m[2])+ ')')
            print('AP40: ', ap40m[0], '('+ str(ap40m[1]) + '-'+ str(ap40m[2])+ ')')
            print('AP100: ', ap100m[0], '('+ str(ap100m[1]) + '-'+ str(ap100m[2])+ ')')
            print('AP200: ', ap200m[0], '('+ str(ap200m[1]) + '-'+ str(ap200m[2])+ ')')

            print('ROUND')

            print('AP12: ', ("%.1f" % ap12m[0]), '('+ str(("%.1f" % ap12m[1]))+ '-'+ str(("%.1f" % ap12m[2]))+ ')')
            print('AP40: ', ("%.1f" % ap40m[0]), '('+ str(("%.1f" % ap40m[1])) + '-'+ str(("%.1f" % ap40m[2]))+ ')')
            print('AP100: ', ("%.1f" % ap100m[0]), '('+ str(("%.1f" % ap100m[1])) + '-'+ str(("%.1f" % ap100m[2]))+ ')')
            print('AP200: ', ("%.1f" % ap200m[0]), '('+ str(("%.1f" % ap200m[1])) + '-'+ str(("%.1f" % ap200m[2]))+ ')')
        else:
            print('AP12: ', ap12m[0], '($\pm '+ str(ap12m[3]),'$)')
            print('AP40: ', ap40m[0], '($\pm '+ str(ap40m[3])+'$)')
            print('AP100: ', ap100m[0], '($\pm '+ str(ap100m[3])+'$)')
            print('AP200: ', ap200m[0], '($\pm '+ str(ap200m[3])+'$)')

            print('ROUND')

            print('AP12: ', ("%.3f" % ap12m[0]), '($\pm '+ str("%.3f" % ap12m[3]),'$)')
            print('AP40: ', ("%.3f" % ap40m[0]), '($\pm '+ str("%.3f" % ap40m[3])+'$)')
            print('AP100: ', ("%.3f" % ap100m[0]), '($\pm '+ str("%.3f" % ap100m[3])+'$)')
            print('AP200: ', ("%.3f" % ap200m[0]), '($\pm '+ str("%.3f" % ap200m[3])+'$)')


        
        
        # print('OFR: ', np.mean(ofr))




def get_args():
    parser = argparse.ArgumentParser(
        description="Make predictions on specified dataset"
    )
    parser.add_argument("--mask-root", type=str, required=True)
    parser.add_argument("--invert-mask", type=str, default="False")
    parser.add_argument("--img-size", type=str, default=352)
    parser.add_argument("--ofr-gt-masks", type=str, required=True)
    # list of strings of OFR sensitivities
    parser.add_argument("--list-of-ofr", type=str, required=True)
    parser.add_argument("--ap-only", type=str, default=False)
    parser.add_argument("--conf-intervals", type=str, default=False)

    return parser.parse_args()


def main():
    args = get_args()
    args.invert_mask = True if args.invert_mask == "True" else False
    args.conf_intervals = True if args.conf_intervals == "True" else False
    eval(args)


if __name__ == "__main__":
    main()





































# import os
# import glob
# import argparse
# import numpy as np
# import cv2
# import ast

# from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
# from skimage.io import imread
# from skimage.transform import resize


# # input: the paths to two images and the list of results to append to
# # output: the list of results with the agreement metric appended
# def agreement(temp_test, temp_retest, val_list, ignore_check=False):
#     if ignore_check:
#         img1 = np.ndarray.flatten(imread(temp_test) / 255) > 0.5
#         img2 = np.ndarray.flatten(imread(temp_retest) / 255) > 0.5
#         black_area = np.logical_and(img1, img2)
#         red_area = np.logical_xor(black_area, img1)
#         green_area = np.logical_xor(black_area, img2)
#         total_area = np.count_nonzero(red_area) + np.count_nonzero(green_area) + np.count_nonzero(black_area)
#         reference_area = np.count_nonzero(red_area) + np.count_nonzero(black_area)
#         test_area = np.count_nonzero(green_area) + np.count_nonzero(black_area)

#         fn = ((total_area-test_area)*100)/reference_area
#         fp = ((total_area-reference_area)*100)/test_area

#         agreement = 100-(fn+fp)/2
#         return val_list.append(agreement)
    
#     if temp_test.split('\\')[-1].split('.')[0].split('-')[1] == temp_retest.split('\\')[-1].split('.')[0].split('-')[1]:
#         img1 = np.ndarray.flatten(imread(temp_test) / 255) > 0.5
#         img2 = np.ndarray.flatten(imread(temp_retest) / 255) > 0.5
#         black_area = np.logical_and(img1, img2)
#         red_area = np.logical_xor(black_area, img1)
#         green_area = np.logical_xor(black_area, img2)
#         total_area = np.count_nonzero(red_area) + np.count_nonzero(green_area) + np.count_nonzero(black_area)
#         reference_area = np.count_nonzero(red_area) + np.count_nonzero(black_area)
#         test_area = np.count_nonzero(green_area) + np.count_nonzero(black_area)

#         fn = ((total_area-test_area)*100)/reference_area
#         fp = ((total_area-reference_area)*100)/test_area

#         agreement = 100-(fn+fp)/2
#         # print('fn: ', fn)
#         # print('fp: ', fp)
#         # print('agreement: ', agreement)
#         return val_list.append(agreement)
#     else:
#         print('WARNING: test and retest images do not match')


# def eval(args):


#     prediction_files = sorted(glob.glob(os.path.join(os.getcwd(), args.mask_root, '*')), key=str.casefold)


#     # split pred files into test and retest globs, compare the test and retest imanges to calculate  adding the metric to their own list of info for each methods of OC
#     pred_test, pred_retest = [], []
#     for pred_itm in prediction_files:
#         if "Rd" in pred_itm.split('\\')[-1]:
#             pred_retest.append(pred_itm)
#         elif "Td" in pred_itm.split('\\')[-1]:
#             pred_test.append(pred_itm)

#     #filters per patient
#     ap12,ap40,ap100,ap200,iso100,iso200,iso300,iso400,tscan100,ofc,ofr = [],[],[],[],[],[],[],[],[],[],[]
#     i = 1
#     while [s for s in pred_test if str(i) in s.split('\\')[-1].split('.')[0].split('-')[0].split('Td')[0]]:
#         # print(i)
#         # filters test and retest
#         temp_test = [s for s in pred_test if str(i) in s.split('\\')[-1].split('.')[0].split('-')[0].split('Td')[0]]
#         temp_retest = [s for s in pred_retest if str(i) in s.split('\\')[-1].split('.')[0].split('-')[0].split('Rd')[0]]
#         # print(temp_test)
#         # print(temp_retest)

#         #compares test and retest
#         # for j in range(len(temp_test)):
#         #     temp_retest2 = [s for s in pred_test if str(i) in s.split('\\')[-1].split('.')[0].split('-')[0].split('Td')[0]]   
#         # print('AP12A')    
#         agreement(temp_test[0], temp_retest[0], ap12)
#         # print('AP12P')
#         agreement(temp_test[1], temp_retest[1], ap12)
#         # print('AP40A')
#         agreement(temp_test[2], temp_retest[2], ap40)
#         # print('AP40P')
#         agreement(temp_test[3], temp_retest[3], ap40)
#         # print('AP100A')
#         agreement(temp_test[4], temp_retest[4], ap100)
#         # print('AP100P')
#         agreement(temp_test[5], temp_retest[5], ap100)
#         # print('AP200A')
#         agreement(temp_test[6], temp_retest[6], ap200)
#         # print('AP200P')
#         agreement(temp_test[7], temp_retest[7], ap200)
#         # print('ISO100')
#         agreement(temp_test[8], temp_retest[8], iso100)
#         # print('ISO200')
#         agreement(temp_test[9], temp_retest[9], iso200)
#         # print('ISO300')
#         agreement(temp_test[10], temp_retest[10], iso300)
#         # print('ISO400')
#         agreement(temp_test[11], temp_retest[11], iso400)
#         # print('OFC40')
#         agreement(temp_test[12], temp_retest[12], ofc)
#         # print('OFR200')
#         agreement(temp_test[13], temp_retest[13], ofr)
#         # print('TSCAN no sen')
#         agreement(temp_test[14], temp_retest[14], tscan100)
#         # print('TSCAN sen')
#         agreement(temp_test[15], temp_retest[15], tscan100)






#             # print(temp_test[j])
#             # print(temp_retest[j])
#             # img1 = np.ndarray.flatten(imread(temp_test[j]) / 255) > 0.5
#             # img2 = np.ndarray.flatten(imread(temp_retest[j]) / 255) > 0.5
            
#             # print(len(img1))
#             # print(len(img2))
#             # print('img1: ', np.unique(img1, return_counts=True))
#             # print('img2: ', np.unique(img2, return_counts=True))
#             # black_area = np.logical_and(img1, img2)
#             # print('black area: ', np.unique(black_area, return_counts=True))
#             # logical half-subtractor borrow, same as B-(A∩B)
#             # red_area = np.logical_xor(black_area, img1)
#             # red_area = np.subtract(img2, black_area)
#             # print('red area: ', np.unique(red_area, return_counts=True))

#             # green_area = np.logical_xor(black_area, img2)
#             # print('green area: ', np.unique(green_area, return_counts=True))
            
#             # total_area = np.count_nonzero(red_area) + np.count_nonzero(green_area) + np.count_nonzero(black_area)
#             # reference_area = np.count_nonzero(red_area) + np.count_nonzero(black_area)
#             # test_area = np.count_nonzero(green_area) + np.count_nonzero(black_area)

#             # fn = ((total_area-test_area)*100)/reference_area
#             # fp = ((total_area-reference_area)*100)/test_area

#             # agreement = 100-(fn+fp)/2
#             # print('fn: ', fn)
#             # print('fp: ', fp)
#             # print('agreement: ', agreement)

#             # dice.append(f1_score(gt, pred))
#             # IoU.append(jaccard_score(gt, pred))
#             # precision.append(precision_score(gt, pred))
#             # recall.append(recall_score(gt, pred))

#         i += 1
    
#     # print(ap12)
#     # print(ap40)
#     # print(ap100)
#     # print(ap200)
#     # print(iso100)
#     # print(iso200)
#     # print(iso300)
#     # print(iso400)
#     # print(tscan100)
#     # print(ofc)
#     # print(ofr)
#     # print('############')
#     print('################################# REPRODUCIBILITY #################################')
#     print('AP12: ', np.mean(ap12))
#     print('AP40: ', np.mean(ap40))
#     print('AP100: ', np.mean(ap100))
#     print('AP200: ', np.mean(ap200))
#     print('ISO100: ', np.mean(iso100))
#     print('ISO200: ', np.mean(iso200))
#     print('ISO300: ', np.mean(iso300))
#     print('ISO400: ', np.mean(iso400))
#     print('TSCAN100: ', np.mean(tscan100))
#     print('OFC: ', np.mean(ofc))
#     print('OFR: ', np.mean(ofr))



#     print('################################# VALIDITY #################################')

#     ofr_files = sorted(glob.glob(os.path.join(os.getcwd(), args.ofr_gt_masks, '*')), key=str.casefold)

#     # take the pred files and compare each one to every OFR iamge of the same patient and test/retest, calculate the metrics and add them to a list of info for each method of OC

#     # splits both datasets into test and retest
#     pred_test, pred_retest = [], []
#     for pred_itm in prediction_files:
#         if "Rd" in pred_itm.split('\\')[-1]:
#             pred_retest.append(pred_itm)
#         elif "Td" in pred_itm.split('\\')[-1]:
#             pred_test.append(pred_itm)

#     ofr_test, ofr_retest = [], []
#     for ofr_itm in ofr_files:
#         if "Rd" in ofr_itm.split('\\')[-1]:
#             ofr_retest.append(ofr_itm)
#         elif "Td" in ofr_itm.split('\\')[-1]:
#             ofr_test.append(ofr_itm)

#     # traverses the types of ofr
#     for l in ast.literal_eval(args.list_of_ofr):
#         #filter ofr types
#         ofr_retest_filtered = [s for s in ofr_retest if l in s.split('\\')[-1].split('.')[0].split('-')[1]]
#         ofr_test_filtered = [s for s in ofr_test if l in s.split('\\')[-1].split('.')[0].split('-')[1]]
#         # print(ofr_test_filtered)
#         #traverses the patient for the type of ofr
#         ap12,ap40,ap100,ap200,iso100,iso200,iso300,iso400,tscan100,ofc = [],[],[],[],[],[],[],[],[],[]
#         i = 1
#         while [s for s in ofr_retest_filtered if str(i) in s.split('\\')[-1].split('.')[0].split('-')[0].split('Rd')[0]]:
#             cur_ofr_test_filtered = [s for s in ofr_test_filtered if str(i) in s.split('\\')[-1].split('.')[0].split('-')[0].split('Td')[0]]
#             cur_ofr_retest_filtered = [s for s in ofr_retest_filtered if str(i) in s.split('\\')[-1].split('.')[0].split('-')[0].split('Rd')[0]]
#             # print(cur_ofr_test_filtered)
#             # print(cur_ofr_retest_filtered)
            
#             #filter pred test and retest by patient
#             cur_pred_test_filtered = [s for s in pred_test if str(i) in s.split('\\')[-1].split('.')[0].split('-')[0].split('Td')[0]]
#             cur_pred_retest_filtered = [s for s in pred_retest if str(i) in s.split('\\')[-1].split('.')[0].split('-')[0].split('Rd')[0]]
#             # print(cur_pred_test_filtered)
#             # print(cur_pred_retest_filtered)
#             #calculate and append the agreement metrics for each pred type

#             agreement(cur_pred_test_filtered[0], cur_ofr_test_filtered[0], ap12, ignore_check=True)
#             agreement(cur_pred_retest_filtered[0], cur_ofr_retest_filtered[0], ap12, ignore_check=True)
#             agreement(cur_pred_test_filtered[1], cur_ofr_test_filtered[0], ap12, ignore_check=True)
#             agreement(cur_pred_retest_filtered[1], cur_ofr_retest_filtered[0], ap12, ignore_check=True)
#             agreement(cur_pred_test_filtered[2], cur_ofr_test_filtered[0], ap40, ignore_check=True)
#             agreement(cur_pred_retest_filtered[2], cur_ofr_retest_filtered[0], ap40, ignore_check=True)
#             agreement(cur_pred_test_filtered[3], cur_ofr_test_filtered[0], ap40, ignore_check=True)
#             agreement(cur_pred_retest_filtered[3], cur_ofr_retest_filtered[0], ap40, ignore_check=True)
#             agreement(cur_pred_test_filtered[4], cur_ofr_test_filtered[0], ap100, ignore_check=True)
#             agreement(cur_pred_retest_filtered[4], cur_ofr_retest_filtered[0], ap100, ignore_check=True)
#             agreement(cur_pred_test_filtered[5], cur_ofr_test_filtered[0], ap100, ignore_check=True)
#             agreement(cur_pred_retest_filtered[5], cur_ofr_retest_filtered[0], ap100, ignore_check=True)
#             agreement(cur_pred_test_filtered[6], cur_ofr_test_filtered[0], ap200, ignore_check=True)
#             agreement(cur_pred_retest_filtered[6], cur_ofr_retest_filtered[0], ap200, ignore_check=True)
#             agreement(cur_pred_test_filtered[7], cur_ofr_test_filtered[0], ap200, ignore_check=True)
#             agreement(cur_pred_retest_filtered[7], cur_ofr_retest_filtered[0], ap200, ignore_check=True)
#             agreement(cur_pred_test_filtered[8], cur_ofr_test_filtered[0], iso100, ignore_check=True)
#             agreement(cur_pred_retest_filtered[8], cur_ofr_retest_filtered[0], iso100, ignore_check=True)
#             agreement(cur_pred_test_filtered[9], cur_ofr_test_filtered[0], iso200, ignore_check=True)
#             agreement(cur_pred_retest_filtered[9], cur_ofr_retest_filtered[0], iso200, ignore_check=True)
#             agreement(cur_pred_test_filtered[10], cur_ofr_test_filtered[0], iso300, ignore_check=True)
#             agreement(cur_pred_retest_filtered[10], cur_ofr_retest_filtered[0], iso300, ignore_check=True)
#             agreement(cur_pred_test_filtered[11], cur_ofr_test_filtered[0], iso400, ignore_check=True)
#             agreement(cur_pred_retest_filtered[11], cur_ofr_retest_filtered[0], iso400, ignore_check=True)
#             agreement(cur_pred_test_filtered[12], cur_ofr_test_filtered[0], ofc, ignore_check=True)
#             agreement(cur_pred_retest_filtered[12], cur_ofr_retest_filtered[0], ofc, ignore_check=True)
#             agreement(cur_pred_test_filtered[14], cur_ofr_test_filtered[0], tscan100, ignore_check=True)
#             agreement(cur_pred_retest_filtered[14], cur_ofr_retest_filtered[0], tscan100, ignore_check=True)
#             agreement(cur_pred_test_filtered[15], cur_ofr_test_filtered[0], tscan100, ignore_check=True)
#             agreement(cur_pred_retest_filtered[15], cur_ofr_retest_filtered[0], tscan100, ignore_check=True)


#             i += 1

#         print('############ OFR: ', l, ' ############')
#         # print(ap12)
#         # print(ap40)
#         # print(ap100)
#         # print(ap200)
#         # print(iso100)
#         # print(iso200)
#         # print(iso300)
#         # print(iso400)
#         # print(tscan100)
#         # print(ofc)
#         # # print(ofr)
#         # print('############')
#         print('AP12: ', np.mean(ap12))
#         print('AP40: ', np.mean(ap40))
#         print('AP100: ', np.mean(ap100))
#         print('AP200: ', np.mean(ap200))
#         print('ISO100: ', np.mean(iso100))
#         print('ISO200: ', np.mean(iso200))
#         print('ISO300: ', np.mean(iso300))
#         print('ISO400: ', np.mean(iso400))
#         print('TSCAN100: ', np.mean(tscan100))
#         print('OFC: ', np.mean(ofc))
#         # print('OFR: ', np.mean(ofr))




# def get_args():
#     parser = argparse.ArgumentParser(
#         description="Make predictions on specified dataset"
#     )
#     parser.add_argument("--mask-root", type=str, required=True)
#     parser.add_argument("--img-size", type=str, default=352)
#     parser.add_argument("--ofr-gt-masks", type=str, required=True)
#     # list of strings of OFR sensitivities
#     parser.add_argument("--list-of-ofr", type=str, required=True)

#     return parser.parse_args()


# def main():
#     args = get_args()
#     eval(args)


# if __name__ == "__main__":
#     main()

