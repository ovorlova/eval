import json
import os
import sys
import numpy as np

from evaluateAP import evaluateAP
from evaluatePCKh import evaluatePCKh
from tensorboardX import SummaryWriter

import eval_helpers
from eval_helpers import *


def eval(gtFramesSingle, prFramesAll, gtMulti, prMulti):

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

