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
import sentencepiece as spm
import base64
import zstandard as zstd
import iced_x86
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import IterableDataset, DataLoader

from util import report_cuda_size, timeit, report_model_size

# Constants and configurations
ENC_SEQ_LEN = 4096
DEC_SEQ_LEN = 4096
BOS, EOS, MASK, DECSTART, ENCSTART, PAD = 65000, 65001, 65002, 65003, 65004, 65005
VOCAB_SIZE = 65006

BATCH_SIZE = 1
LEARNING_RATE = 1e-4
NUM_BATCHES = 10000
GENERATE_EVERY = 1000

ROOTDIR = os.path.abspath(os.path.dirname(__file__))
CHECKPOINT_DIR = f"{ROOTDIR}/checkpoints"

# Load SentencePiece and ZSTD models
enc_sp = spm.SentencePieceProcessor(model_file='encoder.model')
dec_sp = spm.SentencePieceProcessor(model_file='decoder.model')
with open(f'{ROOTDIR}/zstd_enc.dictionary', 'rb') as f:
    enc_zstd = f.read()
with open(f'{ROOTDIR}/zstd_dec.dictionary', 'rb') as f:
    dec_zstd = f.read()


def disassemble(obj_bytes: bytes):
  if obj_bytes[:4] == b'\x7fELF':
    #this is a .o file. we should parse it correctly but for our simple yarpgen code we will just thwack off the header
    #and start disassembling from what is probably the beginning of .text
    obj_bytes = obj_bytes[64:]
    print()
  decoder = iced_x86.Decoder(64, obj_bytes)
  formatter = iced_x86.Formatter(iced_x86.FormatterSyntax.NASM)
  for instr in decoder:
    disasm = formatter.format(instr)
    print(disasm)
  print()

def zstd_tokenize(data: bytes, is_encoder=True) -> [int]:
  dictionary = zstd.ZstdCompressionDict(enc_zstd if is_encoder else dec_zstd)
  compressor = zstd.ZstdCompressor(dict_data=dictionary)
  compressed_data = compressor.compress(data)
  compressed_data = list(compressed_data)
  return compressed_data

def zstd_detokenize(tokens:[int], is_encoder=True) -> bytes:
  compressed_data = bytes(tokens)
  dictionary = zstd.ZstdCompressionDict(enc_zstd if is_encoder else dec_zstd)
  decompressor = zstd.ZstdDecompressor(dict_data=dictionary)
  decompressed_data = decompressor.decompress(compressed_data)
  return decompressed_data


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

def tokenize(inp: bytes):
  return list(inp)

def detokenize(tokens: [int]):
  ret = []
  for t in tokens:
    if t < 256:
      ret.append(t)
  return bytes(ret)

class CompilerDataset(IterableDataset):
    def __init__(self, targz_path):
        self.targz_path = targz_path

    def __iter__(self):
        targz = tarfile.open(self.targz_path, 'r:gz')
        idx = 0
        while True:
            idx += 1
            try:
                unopt_file = targz.extractfile(f'{idx}.unopt.o')
                opt_file = targz.extractfile(f'{idx}.opt.o')
                if unopt_file is None or opt_file is None:
                    break  # End of archive
                unopt_bytes = unopt_file.read()
                opt_bytes = opt_file.read()

                unopt_tokens = spm_tokenize(unopt_bytes)
                opt_tokens = spm_tokenize(opt_bytes)

                if len(unopt_tokens) >= ENC_SEQ_LEN or len(opt_tokens) >= DEC_SEQ_LEN:
                    continue  # Skip this sample as it's too large

                opt_tokens.insert(0, DECSTART)
                mask = [True] * len(unopt_tokens) + [False] * (ENC_SEQ_LEN - len(unopt_tokens))
                unopt_tokens.extend([PAD] * (ENC_SEQ_LEN - len(unopt_tokens)))
                opt_tokens.extend([PAD] * (DEC_SEQ_LEN - len(opt_tokens)))

                yield torch.tensor(unopt_tokens), torch.tensor(opt_tokens), torch.tensor(mask)

            except Exception as e:
                print(f"Error processing file {idx}: {str(e)}")
                continue

        targz.close()


class CompilerModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = XTransformer(
            dim=128,
            pad_value=PAD,
            tie_token_emb=True,
            enc_attn_flash=True,
            dec_attn_flash=True,
            return_tgt_loss=True,
            enc_num_tokens=VOCAB_SIZE,
            enc_depth=4,
            enc_heads=4,
            enc_max_seq_len=ENC_SEQ_LEN,
            dec_num_tokens=VOCAB_SIZE,
            dec_depth=4,
            dec_heads=4,
            dec_max_seq_len=DEC_SEQ_LEN
        )
        self.sample_input = None

    def training_step(self, batch, batch_idx):
        src, tgt, src_mask = batch
        loss = self.model(src, tgt, mask=src_mask)
        self.log('train_loss', loss)
        if self.sample_input is None:
            self.sample_input = (src[0].unsqueeze(0), tgt[0].unsqueeze(0), src_mask[0].unsqueeze(0))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100)
        return [optimizer], [scheduler]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if (self.global_step + 1) % GENERATE_EVERY == 0:
            self.generate_sample()

    def generate_sample(self):
        self.eval()
        if self.sample_input is not None:
            src, tgt, src_mask = self.sample_input
            src, tgt, src_mask = src.to(self.device), tgt.to(self.device), src_mask.to(self.device)
            start_tokens = torch.tensor([DECSTART]).to(self.device)
            sample = self.model.generate(src, start_tokens, DEC_SEQ_LEN)

            print_stmt = f'\nStep {self.global_step} sample:\n'
            print_stmt += f"\nInput tokenized:\n{spm_detokenize(src.tolist()[0])}\n"
            print_stmt += f"\nPredicted detokenized:\n{spm_detokenize(sample.tolist())}\n"
            print_stmt += f"\nActual detokenized:\n{spm_detokenize(tgt.tolist()[0])}\n"
            print(print_stmt)
        self.train()


def main():
    dataset = CompilerDataset(f'{ROOTDIR}/compiler_data.tar.gz')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)

    model = CompilerModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename='compiler-{step:09d}-{train_loss:.2f}',
        save_top_k=3,
        monitor='train_loss'
    )

    trainer = pl.Trainer(
        max_steps=NUM_BATCHES,
        callbacks=[checkpoint_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1
    )

    trainer.fit(model, dataloader)

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    main()