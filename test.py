import argparse
from evaluate import eval
from eval_helpers import *
import json
import matplotlib.pyplot as plt
import numpy
import os


def parseArgs():

    parser = argparse.ArgumentParser(description="Test results with different parameters")
    parser.add_argument("--split", type=str, 
                        help="Choose part of the dataset to test on")
    parser.add_argument("--test_k", action="store_true",
                        help="Test with max test_k_max output objects")
    parser.add_argument("--test_k_max", type=int, default=11,
                        help="The maximum value to test")
    parser.add_argument("--test_score", action="store_true",
                        help="Test with different scores")
    parser.add_argument("--test_score_step", type=float, default=0.05,
                        help="Step for test_score")
    parser.add_argument("--test_score_min", type=int, default=0,
                        help="Takes at least test_score_min predictions despite score")

    parser.add_argument("--path_annos", type=str,
                        help="Path to annotations folder")
    parser.add_argument("--path_res", type=str,
                        help="Path to results") 

    return parser.parse_args()

def convert_eval_format(test_score=False, score_=0.0, min_for_score=0, multi=False, test_k=False, k_=11):
    dct_image_id = {}
    for image in results:
        #image['annorect'].sort(reverse=True, key=lambda x: x['score'][0])
        if multi and image['image'][0]['name'] not in multiInds:
            continue
        if not multi and image['image'][0]['name'] not in singleInds:
            continue
        detections = image['annorect']
        final_detections = []
        counter = 0
        if test_score:
            for detection in detections:
                if detection['score'][0] >= score_:
                    final_detections.append(detection)
                    counter+=1
                else:
                    break
            while (counter < min_for_score):
                final_detections.append(detections[counter])
                counter+=1
        elif test_k:
            final_detections = detections[:min(len(detections), k_)]
        else:
            final_detections = detections

        dct_image_id[image['image'][0]['name']] = final_detections

    final_lst = []
    for key in dct_image_id:
        final_lst.append({'image' : [{'name' : key}], 'annorect' : dct_image_id[key]})
    if multi:
        print("multi", len(final_lst))
    else:
        print("single", len(final_lst))
    return final_lst

def run_eval(test=False, test_score=False, score=0.0, min_for_score=0, multi=False, test_k=False, k_=11):
    return eval(gtFramesSingle, convert_eval_format(score_=score,
                          test_score=test_score, min_for_score=min_for_score, multi=False, test_k=test_k, k_=k_), 
                          gtFramesMulti, convert_eval_format(score_=score,
                          test_score=test_score, min_for_score=min_for_score, multi=True, test_k=test_k, k_=k_))


def test_k(K):
    print("Evaluation per num K of preds:")
    AP = {}
    PCKh = {}
    for k in range(1, K+1):
        print('current k: ', k)
        ap, pckh = run_eval(test=True, test_k=True, k_=k)
        for name in ap:
            if name not in AP:
                AP[name] = []
            AP[name].append(ap[name])
        for name in pckh:
            if name not in PCKh:
                PCKh[name] = []
            PCKh[name].append(pckh[name])

    x = range(1, K + 1)
    y_AP = list(AP)
    y_PCKh = list(PCKh)
    
    plt.figure(figsize=(24,12), dpi= 80)
    plt.suptitle("Test for K, AP")
    
    for i in range(len(y_AP)):
        plt.subplot(3, 3, i+1)
        plt.plot(x, AP[y_AP[i]], label=y_AP[i])
        plt.title(y_AP[i])
        #plt.xlabel('score')
        #plt.ylabel('result')
        #plt.legend()
        plt.grid()
    plt.savefig('img_K_AP.png')


    plt.figure(figsize=(24,12), dpi= 80)
    plt.suptitle("Test for K, PCKh")

    for i in range(len(PCKh)):
        plt.subplot(4, 5, i+1)
        plt.plot(x, PCKh[y_PCKh[i]], label=y_PCKh[i])
        plt.title(y_PCKh[i])
        #plt.xlabel('score')
        #plt.ylabel('result')
        #plt.legend()
        plt.grid()
    plt.savefig('img_K_PCKh.png')

    plt.figure()
    #plt.suptitle("Model for different score thresholds")
    plt.plot(x, PCKh['total'], label='PCKh', marker='o', markersize=3)
    plt.plot(x, AP['total'], label='AP', marker='o', markersize=3)
    plt.title('total')
    plt.xlabel('K predictions')
    plt.ylabel('metric value')
    plt.xticks(numpy.arange(1, K+1, 1))
    plt.grid()
    plt.legend()
    #for i in range(len(x)):
    #  #plt.axvline(x[i], ymin=0, ymax=PCKh['total'][i], linestyle='--', color='green')
    #  plt.plot([ x[i], x[i]], [0, PCKh['total'][i]], linestyle='--', color='green', alpha=0.6)
    plt.savefig('img_K_total.png')


def test_score(step=0.05, minNum=0):
    print("Evaluation per score of preds:")
    AP = {}
    PCKh = {}
    for score in numpy.arange(0., 1+step, step):
        print('current score: ', score)
        ap, pckh = run_eval(test=True, test_score=True,min_for_score=minNum, score=score)
        for name in ap:
            if name not in AP:
                AP[name] = []
            AP[name].append(ap[name])
        for name in pckh:
            if name not in PCKh:
                PCKh[name] = []
            PCKh[name].append(pckh[name])

    x = numpy.arange(0, step + 1, step)
    y_AP = list(AP)
    y_PCKh = list(PCKh)
    
    plt.figure(figsize=(24,12), dpi= 80)
    plt.suptitle("Test for score, AP")
    
    for i in range(len(y_AP)):
        plt.subplot(3, 3, i+1)
        plt.plot(x, AP[y_AP[i]], label=y_AP[i])
        plt.title(y_AP[i])
        #plt.xlabel('score')
        #plt.ylabel('result')
        #plt.legend()
        plt.grid()
    plt.savefig('img_score_AP.png')

    plt.figure(figsize=(24,12), dpi= 80)
    plt.suptitle("Test for score, PCKh")

    for i in range(len(PCKh)):
        plt.subplot(4, 5, i+1)
        plt.plot(x, PCKh[y_PCKh[i]], label=y_PCKh[i])
        plt.title(y_PCKh[i])
        #plt.xlabel('score')
        #plt.ylabel('result')
        #plt.legend()
        plt.grid()
    plt.savefig('img_score_PCKh.png')

    plt.figure()
    #plt.suptitle("Test for score, total")
    plt.plot(x, PCKh['total'], label='PCKh', marker='o', markersize=3)
    plt.plot(x, AP['total'], label='AP', marker='o', markersize=3)
    plt.title('total')
    plt.xlabel('min score')
    plt.xticks(numpy.arange(0, 1.1, 0.1))
    plt.ylabel('metric value')
    plt.grid()
    plt.legend()
    #for i in range(len(x)):
    #  #plt.axvline(x[i], ymin=0, ymax=PCKh['total'][i], linestyle='--', color='green')
    #  plt.plot([ x[i], x[i]], [0, PCKh['total'][i]], linestyle='--', color='green', alpha=0.6)
    plt.savefig('img_score_PCKh_total.png')


args = parseArgs()
results = json.load(open(args.path_res, 'r'))

gtFramesSingle = loadGTFrames(args.path_annos, '{}_single.json'.format(args.split))
gtFramesMulti = loadGTFrames(args.path_annos, '{}_multi.json'.format(args.split))

singleInds = []
for frame in gtFramesSingle:
    singleInds.append(frame['image'][0]['name'])

multiInds = []
for frame in gtFramesMulti:
    multiInds.append(frame['image'][0]['name'])

if args.test_score == True:
    test_score(args.test_score_step, args.test_score_min)
if args.test_k == True:
    test_k(args.test_k_max)

#plt.show()

