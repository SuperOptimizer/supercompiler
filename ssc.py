from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import os
import sys
import ast
import csv
import gzip
import torch
import random
import tqdm
from x_transformers import XTransformer
import tarfile


from util import report_cuda_size, timeit, report_model_size, chunkify

ENC_SEQ_LEN = 4096
DEC_SEQ_LEN = 4096
BOS = 256
EOS = 257
MASK = 258
DECSTART = 259
ENCSTART = 260
PAD = 261
VOCAB_SIZE=262


BATCH_SIZE=1
WORLD_SIZE = 1
LEARNING_RATE = 1e-4
NUM_BATCHES = 10000
GENERATE_EVERY=1000

DEVICE='cuda'

ROOTDIR = os.path.abspath(os.path.dirname(__file__))
CHECKPOINT = f"{ROOTDIR}/checkpoint.pt"

def tokenize(inp: bytes):
  return list(inp)

def detokenize(tokens: [int]):
  ret = []
  for t in tokens:
    if t < 256:
      ret.append(t)
  return bytes(ret)


def cycle(targz):
  i = 0
  training_data = []
  while True:
    i += 1
    unopt_file = targz.extractfile(f'{i}.unopt.o')
    opt_file = targz.extractfile(f'{i}.opt.o')
    unopt_bytes = unopt_file.read()
    opt_bytes = opt_file.read()

    unopt_tokens = tokenize(unopt_bytes)
    opt_tokens = tokenize(opt_bytes)
    #print(f"len unopt tokens {len(unopt_tokens)} len opt tokens {len(opt_tokens)} len unopt {len(unopt)} len opt {len(opt)}")
    if len(unopt_tokens) >= ENC_SEQ_LEN or len(opt_tokens) >= DEC_SEQ_LEN:

      continue
    opt_tokens.insert(0, DECSTART)
    mask = [True] * len(unopt_tokens)
    mask.extend([False] * (ENC_SEQ_LEN - len(unopt_tokens)))
    unopt_tokens.extend([PAD] * (ENC_SEQ_LEN - len(unopt_tokens)))
    opt_tokens.extend([PAD] * (DEC_SEQ_LEN - len(opt_tokens)))
    training_data.append([unopt_tokens, opt_tokens, mask])
    if len(training_data) == BATCH_SIZE:
      batch = training_data[:BATCH_SIZE]
      training_data = training_data[BATCH_SIZE:]
      mysrc = torch.tensor(list(x[0] for x in batch)).long().to(DEVICE)
      mytgt = torch.tensor(list(x[1] for x in batch)).long().to(DEVICE)
      mysrc_mask = torch.tensor(list(x[2] for x in batch)).bool().to(DEVICE)
      yield mysrc, mysrc_mask, mytgt
  print("we should never get here")



def save_checkpoint(model,  optim, loss, scaler, scheduler):
  if DEVICE == 'cuda':
    torch.save({
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optim.state_dict(),
      'loss': loss.item(),
      'scaler': scaler.state_dict(),
      'scheduler': scheduler.state_dict()},
      CHECKPOINT)

def load_checkpoint(model, optim, loss):
  if os.path.exists(CHECKPOINT):
    print(f"loading {CHECKPOINT}")
    checkpoint = torch.load(CHECKPOINT)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
  return model, optim,  loss

@timeit
def train(rank):

  if WORLD_SIZE > 1:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group(backend='nccl', rank=rank,world_size=WORLD_SIZE)
    torch.cuda.set_device(rank)

  model = XTransformer(
    dim=512,
    pad_value=PAD,
    tie_token_emb=True,
    enc_attn_flash=True,
    dec_attn_flash=True,
    return_tgt_loss=True,
    enc_num_tokens=VOCAB_SIZE,
    enc_depth=8,
    enc_heads=8,
    enc_max_seq_len=ENC_SEQ_LEN,
    dec_num_tokens=VOCAB_SIZE,
    dec_depth=8,
    dec_heads=8,
    dec_max_seq_len=DEC_SEQ_LEN).cuda()


  report_model_size(model)
  optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
  if DEVICE == 'cuda':
    scaler = torch.cuda.amp.GradScaler()
  scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim,T_0=100)
  model, optim, loss = load_checkpoint(model, optim, 0)

  targz = tarfile.open(f'/{ROOTDIR}/compiler_data.tar.gz','r:gz')

  for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()
    optim.zero_grad()

    src, src_mask, tgt = next(cycle(targz))
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      loss = model(src, tgt, mask=src_mask)
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()

    scheduler.step(i/NUM_BATCHES)
    print(f'{i}: {loss.item()}')

    if i == 0 and DEVICE == 'cuda':
      report_cuda_size()

    if i % GENERATE_EVERY == GENERATE_EVERY-1:
      with FSDP.summon_full_params(model, writeback=False, recurse=False):
        #if i > 0:
        #  save_checkpoint(model,  optim, loss, scaler, scheduler)
        model.eval()
        src, src_mask, tgt = next(cycle(targz))
        #src, src_mask, tgt  = src[:1], src_mask[:1], tgt[:1]
        start_tokens = torch.tensor([DECSTART]).to(DEVICE)
        #getting an off by one error when trying to pass the mask here because ???
        #so skip for now
        sample = model.generate(src, start_tokens, DEC_SEQ_LEN)

        print_stmt = f'\nRANK: {rank} start\n'
        print_stmt += f"\ninput tokenized:  \n{detokenize(src.tolist()[0])} \n"
        print_stmt += f"\npredicted detokenized:  \n{detokenize(sample.tolist())}\n"
        print_stmt += f"\nactual detokenized:     \n{detokenize(tgt.tolist()[0])}\n"
        print_stmt += f'\nRANK: {rank} end\n'
        print(print_stmt)


  if WORLD_SIZE > 1:
    torch.distributed.destroy_process_group()

def main():
  print(f'spawning {WORLD_SIZE} processes(s)')
  if WORLD_SIZE == 1:
    train(0)
  else:
    torch.multiprocessing.spawn(train, args=(), nprocs=WORLD_SIZE,join=True)

if __name__ == '__main__':
  main()