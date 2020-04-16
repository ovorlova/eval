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

def assign(gtFrames, prFrames, distThresh):
    assert (len(gtFrames) == len(prFrames))

    nJoints = Joint().count
    # part detection scores
    scoresAll = {}
    # positive / negative labels
    labelsAll = {}
    # number of annotated GT joints per image
    nGTall = np.zeros([nJoints, len(gtFrames)])
    for pidx in range(nJoints):
        scoresAll[pidx] = {}
        labelsAll[pidx] = {}
        for imgidx in range(len(gtFrames)):
            scoresAll[pidx][imgidx] = np.zeros([0, 0], dtype=np.float32)
            labelsAll[pidx][imgidx] = np.zeros([0, 0], dtype=np.int8)

    # GT track IDs
    trackidxGT = []

    # prediction track IDs
    trackidxPr = []

    # number of GT poses
    nGTPeople = np.zeros((len(gtFrames), 1))
    # number of predicted poses
    nPrPeople = np.zeros((len(gtFrames), 1))

    # container to save info for computing MOT metrics
    motAll = {}

    for imgidx in range(len(gtFrames)):
        # distance between predicted and GT joints
        dist = np.full((len(prFrames[imgidx]["annorect"]), len(gtFrames[imgidx]["annorect"]), nJoints), np.inf)
        # score of the predicted joint
        score = np.full((len(prFrames[imgidx]["annorect"]), nJoints), np.nan)
        # body joint prediction exist
        hasPr = np.zeros((len(prFrames[imgidx]["annorect"]), nJoints), dtype=bool)
        # body joint is annotated
        hasGT = np.zeros((len(gtFrames[imgidx]["annorect"]), nJoints), dtype=bool)

        trackidxGT = []
        trackidxPr = []
        idxsPr = []
        for ridxPr in range(len(prFrames[imgidx]["annorect"])):
            if (("annopoints" in prFrames[imgidx]["annorect"][ridxPr].keys()) and
                ("point" in prFrames[imgidx]["annorect"][ridxPr]["annopoints"][0].keys())):
                idxsPr += [ridxPr];
        prFrames[imgidx]["annorect"] = [prFrames[imgidx]["annorect"][ridx] for ridx in idxsPr]

        nPrPeople[imgidx, 0] = len(prFrames[imgidx]["annorect"])
        nGTPeople[imgidx, 0] = len(gtFrames[imgidx]["annorect"])
        # iterate over GT poses
        for ridxGT in range(len(gtFrames[imgidx]["annorect"])):
            # GT pose
            rectGT = gtFrames[imgidx]["annorect"][ridxGT]
            if ("track_id" in rectGT.keys()):
                trackidxGT += [rectGT["track_id"][0]]
            pointsGT = []
            if len(rectGT["annopoints"]) > 0:
                pointsGT = rectGT["annopoints"][0]["point"]
            # iterate over all possible body joints
            for i in range(nJoints):
                # GT joint in LSP format
                ppGT = getPointGTbyID(pointsGT, i)
                if len(ppGT) > 0:
                    hasGT[ridxGT, i] = True

        # iterate over predicted poses
        for ridxPr in range(len(prFrames[imgidx]["annorect"])):
            # predicted pose
            rectPr = prFrames[imgidx]["annorect"][ridxPr]
            if ("track_id" in rectPr.keys()):
                trackidxPr += [rectPr["track_id"][0]]
            pointsPr = rectPr["annopoints"][0]["point"]
            for i in range(nJoints):
                # predicted joint in LSP format
                ppPr = getPointGTbyID(pointsPr, i)
                if len(ppPr) > 0:
                    if not ("score" in ppPr.keys()):
                        # use minimum score if predicted score is missing
                        if (imgidx == 0):
                            print('WARNING: prediction score is missing. Setting fallback score={}'.format(MIN_SCORE))
                        score[ridxPr, i] = MIN_SCORE
                    else:
                        score[ridxPr, i] = ppPr["score"][0]
                    hasPr[ridxPr, i] = True

        if len(prFrames[imgidx]["annorect"]) and len(gtFrames[imgidx]["annorect"]):
            # predictions and GT are present
            # iterate over GT poses
            for ridxGT in range(len(gtFrames[imgidx]["annorect"])):
                # GT pose
                rectGT = gtFrames[imgidx]["annorect"][ridxGT]
                # compute reference distance as head size
                if "x1" not in rectGT:
                  continue
                headSize = getHeadSize(rectGT["x1"][0], rectGT["y1"][0],
                                                    rectGT["x2"][0], rectGT["y2"][0])
                pointsGT = []
                if len(rectGT["annopoints"]) > 0:
                    pointsGT = rectGT["annopoints"][0]["point"]
                # iterate over predicted poses
                for ridxPr in range(len(prFrames[imgidx]["annorect"])):
                    # predicted pose
                    rectPr = prFrames[imgidx]["annorect"][ridxPr]
                    pointsPr = rectPr["annopoints"][0]["point"]

                    # iterate over all possible body joints
                    for i in range(nJoints):
                        # GT joint
                        ppGT = getPointGTbyID(pointsGT, i)
                        # predicted joint
                        ppPr = getPointGTbyID(pointsPr, i)
                        # compute distance between predicted and GT joint locations
                        if hasPr[ridxPr, i] and hasGT[ridxGT, i]:
                            pointGT = [ppGT["x"][0], ppGT["y"][0]]
                            pointPr = [ppPr["x"][0], ppPr["y"][0]]
                            dist[ridxPr, ridxGT, i] = np.linalg.norm(np.subtract(pointGT, pointPr)) / headSize

            dist = np.array(dist)
            hasGT = np.array(hasGT)

            # number of annotated joints
            nGTp = np.sum(hasGT, axis=1)
            match = dist <= distThresh
            pck = 1.0 * np.sum(match, axis=2)
            for i in range(hasPr.shape[0]):
                for j in range(hasGT.shape[0]):
                    if nGTp[j] > 0:
                        pck[i, j] = pck[i, j] / nGTp[j]

            # preserve best GT match only
            idx = np.argmax(pck, axis=1)
            val = np.max(pck, axis=1)
            for ridxPr in range(pck.shape[0]):
                for ridxGT in range(pck.shape[1]):
                    if (ridxGT != idx[ridxPr]):
                        pck[ridxPr, ridxGT] = 0
            prToGT = np.argmax(pck, axis=0)
            val = np.max(pck, axis=0)
            prToGT[val == 0] = -1


            # assign predicted poses to GT poses
            for ridxPr in range(hasPr.shape[0]):
                if (ridxPr in prToGT):  # pose matches to GT
                    # GT pose that matches the predicted pose
                    ridxGT = np.argwhere(prToGT == ridxPr)
                    assert(ridxGT.size == 1)
                    ridxGT = ridxGT[0,0]
                    s = score[ridxPr, :]
                    m = np.squeeze(match[ridxPr, ridxGT, :])
                    hp = hasPr[ridxPr, :]
                    for i in range(len(hp)):
                        if (hp[i]):
                            scoresAll[i][imgidx] = np.append(scoresAll[i][imgidx], s[i])
                            labelsAll[i][imgidx] = np.append(labelsAll[i][imgidx], m[i])
            newPr = []
            for i in range(len(prToGT)):
                newPr.append(prFrames[imgidx]["annorect"][prToGT[i]])
            prFrames[imgidx]["annorect"] = newPr


    return prFrames

def eval(gtFramesAll, gtFramesSingle, prFramesAll):
  
  print("# gt frames  :", len(gtFramesAll))
  print("# pred frames:", len(prFramesAll))

  print("Evaluation of per-frame multi-person pose estimation...")
  apAll,preAll,recAll = evaluateAP(gtFramesAll,prFramesAll)

  names = ['head', 'shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle', 'total']
  ap = {}
  for i in range(len(names)):
    ap[names[i]] = apAll[i][0]
  print(ap)
  print('Evaluation of PCKh@0.5...')
  prFrames = assign(gtFramesSingle, prFramesAll, 0.5)
  pckh = evaluatePCKh(gtFramesSingle, prFrames)
  print(pckh)
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



