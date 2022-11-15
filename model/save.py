import os
import torch
import shutil

def save_checkpoint(state, is_best, save_path, filename, timestamp=''):
  filename = os.path.join(save_path, filename)
  torch.save(state, filename)
  if is_best:
    bestname = os.path.join(save_path, 'model_best_{0}.pth.tar'.format(timestamp))
    shutil.copyfile(filename, bestname)