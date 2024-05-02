from functools import wraps
import time
import torch
import random
import string
import numpy as np
import itertools
import os

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
HOMEDIR = os.path.abspath(os.path.expanduser("~"))
TMP = '/tmp/sopt'

def chunkify(l, n):
  for i in range(0, len(l), n):
    yield l[i:i + n]

def flatten(*args):
  ret = []
  for arg in args:
    if isinstance(arg,list):
      ret.extend(flatten(*arg))
    else:
      ret.append(arg)
  return ret


def timeit(func):
  @wraps(func)
  def timeit_wrapper(*args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
    return result
  return timeit_wrapper

nvmlInit_called = False

def report_cuda_size():
  global nvmlInit_called
  if torch.cuda.is_available():
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    if not nvmlInit_called:
      nvmlInit()
      nvmlInit_called = True
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'cuda total    : {info.total // 1024 // 1024}MB')
    print(f'cuda free     : {info.free // 1024 // 1024}MB')
    print(f'cuda used     : {info.used // 1024 // 1024}MB')

def report_model_size(model):
  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
  params = sum([np.prod(p.size()) for p in model_parameters])
  print(f"num params {params // 1024 // 1024}M {params // 1024}K ")

def randstring(n):
  return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))