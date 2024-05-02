#Superoptimizing Super Compiler

import os
import ast
import gzip
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
import tarfile
from x_transformers import XTransformer

if not torch.cuda.is_available():
    raise RuntimeError('CUDA is not available')
torch.set_default_device('cuda')

DEVICE = 'cuda'
ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
NUM_TOKENS = 261
ENC_SEQ_LEN = 2048
DEC_SEQ_LEN = 2048
NUM_EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
CHECKPOINT_EVERY = 1000
GENERATE_EVERY = 100

ENCSTART, ENCEND, DECSTART, DECEND, PAD = [256, 257, 258, 259, 260]

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



def getdata(path, batch_size=1):
    opt_objs = []
    unopt_objs = []
    training_data = []
    for targz in sorted(os.listdir(path)):
        if not targz.endswith('.tar.gz'):
            continue
        print(f"loading {path}/{targz}")
        with tarfile.open(targz, 'r:gz') as tar:
            # Get a list of file names in the archive
            file_names = tar.getnames()
            # Read a specific file from the archive
            for member in tar.getmembers():
                if member.name.endswith('.opt.cpp.o'):
                    file_obj = tar.extractfile(member)
                    content = file_obj.read()
                    opt_objs.append(content)
                elif member.name.endswith('.unopt.cpp.o'):
                    file_obj = tar.extractfile(member)
                    content = file_obj.read()
                    unopt_objs.append(content)
            if len(opt_objs) != len(unopt_objs):
                raise RuntimeError(f'Mismatch in length of opt_objs and unopt_objs')
            for unopt,opt in zip(unopt_objs, opt_objs):
                unopt_tokens = tokenize(unopt)
                opt_tokens = tokenize(opt)
                if len(unopt_tokens) >= ENC_SEQ_LEN or len(opt_tokens) >= DEC_SEQ_LEN:
                    continue
                unopt_tokens.insert(0,ENCSTART)
                unopt_tokens.append(ENCEND)
                opt_tokens.insert(0, DECSTART)
                opt_tokens.append(DECEND)
                mask = [True] * len(unopt_tokens)
                mask.extend([False] * (ENC_SEQ_LEN - len(unopt_tokens)))
                unopt_tokens.extend([PAD] * (ENC_SEQ_LEN - len(unopt_tokens)))
                opt_tokens.extend([PAD] * (DEC_SEQ_LEN - len(opt_tokens)))
                training_data.append([unopt_tokens, opt_tokens, mask])
        while len(training_data) > batch_size:
            batch = training_data[:batch_size]
            training_data = training_data[batch_size:]
            mysrc = torch.tensor(list(x[0] for x in batch)).long().to(DEVICE)
            mytgt = torch.tensor(list(x[1] for x in batch)).long().to(DEVICE)
            mysrc_mask = torch.tensor(list(x[2] for x in batch)).bool().to(DEVICE)
            yield mysrc, mysrc_mask, mytgt


def train(rank):
  model = XTransformer(
    dim=512,
    pad_value=PAD,
    tie_token_emb=True,
    enc_attn_flash=True,
    dec_attn_flash=True,
    return_tgt_loss=True,
    enc_num_tokens=NUM_TOKENS,
    enc_depth=4,
    enc_heads=4,
    enc_max_seq_len=ENC_SEQ_LEN,
    dec_num_tokens=NUM_TOKENS,
    dec_depth=4,
    dec_heads=4,
    dec_max_seq_len=DEC_SEQ_LEN)

  optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
  if DEVICE == 'cuda':
    scaler = torch.cuda.amp.GradScaler()
  scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim,T_0=100)

  for i in tqdm.tqdm(range(5), mininterval=10., desc='training'):
    model.train()
    optim.zero_grad()

    src, src_mask, tgt = next(getdata('.'))
    if DEVICE == 'cuda':
      with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        loss = model(src, tgt, mask=src_mask)
      scaler.scale(loss).backward()
      scaler.step(optim)
      scaler.update()
    else:
      loss = model(src, tgt, mask=src_mask)
      loss.backward()
      optim.step()
    scheduler.step(i/1)
    print(f'{i}: {loss.item()}')

    if i == 0 and DEVICE == 'cuda':
      pass
      #report_cuda_size()
    if i % GENERATE_EVERY == 0:
        model.eval()
        src, src_mask, tgt = next(getdata('.', 1))
        src, src_mask, tgt  = src[:1], src_mask[:1], tgt[:1]
        start_tokens = torch.tensor([DECSTART]).to(DEVICE)
        sample = model.generate(src, start_tokens, DEC_SEQ_LEN, mask = src_mask)

        print_stmt = f'\nRANK: {rank} start\n'
        print_stmt += f"\ninput tokenized:  \n{detokenize(src.tolist()[0])} \n"
        print_stmt += f"\npredicted detokenized:  \n{detokenize(sample.tolist())}\n"
        print_stmt += f"\nactual detokenized:     \n{detokenize(tgt.tolist()[0])}\n"
        print_stmt += f'\nRANK: {rank} end\n'
        print(print_stmt)


# Main function
def main():
    train(0)

if __name__ == '__main__':
    main()