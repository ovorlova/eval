import json
import os
import sys
import numpy as np
import argparse

from evaluateAP import evaluateAP
from evaluateTracking import evaluateTracking
from evaluatePCKh import evaluatePCKh
from tensorboardX import SummaryWriter

import eval_helpers
from eval_helpers import *

def parseArgs():

    parser = argparse.ArgumentParser(description="Evaluation of Pose Estimation and Tracking (PoseTrack)")
    parser.add_argument("-g", "--groundTruth",required=False,type=str,help="Directory containing ground truth annotatations per sequence in json format")
    parser.add_argument("-p", "--predictions",required=False,type=str,help="Directory containing predictions per sequence in json format")
    parser.add_argument("-e", "--evalPoseEstimation",required=False,action="store_true",help="Evaluation of per-frame  multi-person pose estimation using AP metric")
    parser.add_argument("-t", "--evalPoseTracking",required=False,action="store_true",help="Evaluation of video-based  multi-person pose tracking using MOT metrics")
    parser.add_argument("-s","--saveEvalPerSequence",required=False,action="store_true",help="Save evaluation results per sequence",default=False)
    parser.add_argument("-o", "--outputDir",required=False,type=str,help="Output directory to save the results",default="./out")
    return parser.parse_args()


def pseudo_eval(gtFramesSingle, prFramesAll, gtMulti, prMulti):
  print("# gt frames  :", len(gtFramesAll))
  print("# pred frames:", len(prFramesAll))

  print("Evaluation of per-frame multi-person pose estimation... (multi preds)")
  apMultiAll,preAll,recAll = evaluateAP(gtMulti, prMulti)
  apMulti = turnToAPDict(apMultiAll)
  printResults(apMulti)
  
  print('Evaluation of PCKh@0.5...')
  prFrames = assign(gtFramesSingle, prFramesAll, 0.5)
  pckh = evaluatePCKh(gtFramesSingle, prFrames)
  pckh = turnToPCKhDict(pckh)
  printResults(pckh)
  return apMulti, pckh


def eval(gtFramesAll, gtFramesSingle, prFramesAll):
  
  print("# gt frames  :", len(gtFramesAll))
  print("# pred frames:", len(prFramesAll))

  print("Evaluation of per-frame multi-person pose estimation...")
  apAll,preAll,recAll = evaluateAP(gtFramesAll,prFramesAll)
  ap = turnToAPDict(apAll)
  printResults(ap)
  
  print('Evaluation of PCKh@0.5...')
  prFrames = assign(gtFramesSingle, prFramesAll, 0.5)
  pckh = evaluatePCKh(gtFramesSingle, prFrames)
  pckh = turnToPCKhDict(pckh)
  printResults(pckh)
  return ap, pckh

def main():

    args = parseArgs()
    print(args)
    argv = ['',args.groundTruth,args.predictions]

    print("Loading data")
    gtFramesAll,prFramesAll = eval_helpers.load_data_dir(argv)

    print("# gt frames  :", len(gtFramesAll))
    print("# pred frames:", len(prFramesAll))

    if (not os.path.exists(args.outputDir)):
        os.makedirs(args.outputDir)
        
    if (args.evalPoseEstimation):
        #####################################################
        # evaluate per-frame multi-person pose estimation (AP)

        # compute AP
        #print("Evaluation of per-frame multi-person pose estimation")
        #apAll,preAll,recAll = evaluateAP(gtFramesAll,prFramesAll,args.outputDir,True,args.saveEvalPerSequence)
       # pckh = evaluatePCKh(gtFramesAll, prFramesAll)
       # print(pckh)
        # print AP
       # print("Average Precision (AP) metric:")
        #eval_helpers.printTable(apAll)

      prFrames = assign(gtFramesAll, prFramesAll, 0.5)
      pckh = evaluatePCKh(gtFramesAll, prFrames)
      print(pckh)

    if (args.evalPoseTracking):
        #####################################################
        # evaluate multi-person pose tracking in video (MOTA)
        
        # compute MOTA
        print("Evaluation of video-based  multi-person pose tracking") 
        metricsAll = evaluateTracking(gtFramesAll,prFramesAll,args.outputDir,True,args.saveEvalPerSequence)

        metrics = np.zeros([Joint().count + 4,1])
        for i in range(Joint().count+1):
            metrics[i,0] = metricsAll['mota'][0,i]
        metrics[Joint().count+1,0] = metricsAll['motp'][0,Joint().count]
        metrics[Joint().count+2,0] = metricsAll['pre'][0,Joint().count]
        metrics[Joint().count+3,0] = metricsAll['rec'][0,Joint().count]

        # print AP
        print("Multiple Object Tracking (MOT) metrics:")
        eval_helpers.printTable(metrics,motHeader=True)

#if __name__ == "__main__":
#   main()



