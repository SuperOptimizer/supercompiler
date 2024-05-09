import os
import torch
import tqdm
import tarfile
import random
import math
from x_transformers import XTransformer
from torch.utils.data import Dataset, DataLoader
import multiprocessing
import base64
import msamp
from msamp import deepspeed

if not torch.cuda.is_available():
  raise Exception("cuda not available")
torch.set_default_device('cuda')
#torch.multiprocessing.set_start_method('spawn')

DEVICES = 2
if DEVICES > 1:
  STRATEGY = 'deepspeed_stage_3_offload'
else:
  STRATEGY = ""

DEVICE = 'cuda'
ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
NUM_TOKENS = 65000+5
ENC_SEQ_LEN = 2048
DEC_SEQ_LEN = 2048
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
CHECKPOINT_EVERY = 1000
GENERATE_EVERY = 10000
NUM_EPOCHS = 10000

ENCSTART, ENCEND, DECSTART, DECEND, PAD = [65000, 65001, 65002, 65003, 65004]

# Tokenization functions
def tokenize(data: bytes):
  return list(data)

def detokenize(tokens: [int]):
  ret = []
  for t in tokens:
    if ENCSTART <= t <= PAD:
      pass
    else:
      ret.append(t)
  return bytes(ret)

import sentencepiece as spm

# Load the SentencePiece tokenizers
enc_sp = spm.SentencePieceProcessor(model_file='encoder.model')
dec_sp = spm.SentencePieceProcessor(model_file='decoder.model')

# Tokenization functions
def spm_tokenize(data: bytes, is_encoder=True):
  if is_encoder:
    return enc_sp.encode(base64.b64encode(data).decode('ascii'), out_type=int)
  else:
    return dec_sp.encode(base64.b64encode(data).decode('ascii'), out_type=int)

def spm_detokenize(tokens: [int], is_encoder=True):
  if is_encoder:
    return enc_sp.decode(tokens)
  else:
    return dec_sp.decode(tokens)

class XTransformerDataset(torch.utils.data.IterableDataset):
  def __init__(self, rootdir):
    super().__init__()
    self.rootdir = rootdir
    self.tars = os.listdir(f'{rootdir}/data/')
    random.shuffle(self.tars)

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:  # single-process data loading
      tar_list = self.tars
    else:  # in a worker process
      # split workload
      per_worker = int(math.ceil(len(self.tars) / float(worker_info.num_workers)))
      worker_id = worker_info.id
      start = worker_id * per_worker
      end = min(start + per_worker, len(self.tars))
      tar_list = self.tars[start:end]

    for tar in tar_list:
      if not tar.endswith('.tar.gz'):
        continue

      with tarfile.open(f"{self.rootdir}/data/{tar}", 'r:gz') as tar_file:
        entries = []
        print(f"loading {self.rootdir}/data/{tar}")
        for member in tar_file.getmembers():
          if not member.name.endswith('.o'):
            continue

          if member.name.endswith('.opt.cpp.o'):
            file_obj = tar_file.extractfile(member)
            opt_content = file_obj.read()
          elif member.name.endswith('.unopt.cpp.o'):
            file_obj = tar_file.extractfile(member)
            unopt_content = file_obj.read()
            unopt_tokens = spm_tokenize(unopt_content, True)
            opt_tokens = spm_tokenize(opt_content, False)
            if len(unopt_tokens) < ENC_SEQ_LEN and len(opt_tokens) < DEC_SEQ_LEN:
              entries.append((unopt_tokens, opt_tokens))

        # Sort entries by the sum of unopt and opt token sizes
        entries.sort(key=lambda x: len(x[0]) + len(x[1]), reverse=True)

        # Fill up the context with as many entries as possible
        while entries:
          unopt_batch = [ENCSTART]
          opt_batch = [DECSTART]
          mask = [True]
          curr_enc_len = 1
          curr_dec_len = 1

          while entries:
            unopt_tokens, opt_tokens = entries[-1]

            # Check if adding the current entry would overflow either context
            if curr_enc_len + len(unopt_tokens) + 2 > ENC_SEQ_LEN or curr_dec_len + len(
              opt_tokens) + 2 > DEC_SEQ_LEN:
              break

            entries.pop()
            unopt_batch.extend(unopt_tokens + [ENCEND, ENCSTART])
            opt_batch.extend(opt_tokens + [DECEND, DECSTART])
            mask.extend([True] * (len(unopt_tokens) + 2))
            curr_enc_len += len(unopt_tokens) + 2
            curr_dec_len += len(opt_tokens) + 2

          unopt_batch = unopt_batch[:-1] + [ENCEND]
          opt_batch = opt_batch[:-1] + [DECEND]
          mask = mask[:-1]

          # Pad the sequences
          unopt_batch.extend([PAD] * (ENC_SEQ_LEN - len(unopt_batch)))
          opt_batch.extend([PAD] * (DEC_SEQ_LEN - len(opt_batch)))
          mask.extend([False] * (ENC_SEQ_LEN - len(mask)))

          yield torch.tensor(unopt_batch, dtype=torch.long), torch.tensor(opt_batch,
                                          dtype=torch.long), torch.tensor(
            mask, dtype=torch.bool)


class XTransformerModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.model = XTransformer(
      dim=1024,
      pad_value=PAD,
      tie_token_emb=True,
      enc_attn_flash=True,
      dec_attn_flash=True,
      return_tgt_loss=True,
      enc_num_tokens=NUM_TOKENS,
      enc_depth=10,
      enc_heads=10,
      enc_max_seq_len=ENC_SEQ_LEN,
      dec_num_tokens=NUM_TOKENS,
      dec_depth=10,
      dec_heads=10,
      dec_max_seq_len=DEC_SEQ_LEN,
    )

  def forward(self, src, tgt, mask):
    return self.model(src, tgt, mask=mask)

  def generate(self, src, start_tokens, seq_len, mask):
    return self.model.generate(src, start_tokens, seq_len, mask=mask)


def train(model, dataloader, optimizer, num_epochs):
  model.train()
  for epoch in range(num_epochs):
    i = 0
    for batch in dataloader:
      print(f"batch {i}")
      i+=1
      src, tgt, mask = batch
      src, tgt, mask = src.to(DEVICE), tgt.to(DEVICE), mask.to(DEVICE)

      optimizer.zero_grad()
      loss = model(src, tgt, mask)
      loss.backward()
      optimizer.step()


    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


def main():
  dataset = XTransformerDataset(ROOTDIR)
  dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)

  model = XTransformerModel().cuda()
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

  model, optimizer = msamp.initialize(model, optimizer, opt_level="O2")

  train(model, dataloader, optimizer, NUM_EPOCHS)


if __name__ == '__main__':
  torch.set_float32_matmul_precision('medium')
  main()