import os
import torch
import tqdm
import tarfile
import random
from x_transformers import XTransformer
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import torch.cuda.amp as amp
import multiprocessing
from util import report_cuda_size, report_model_size
import pytorch_lightning as pl
from lightning.pytorch.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam
if not torch.cuda.is_available():
    raise Exception("cuda not available")

DEEPSPEED = True

DEVICE = 'cuda'
ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
NUM_TOKENS = 261
ENC_SEQ_LEN = 4096
DEC_SEQ_LEN = 4096
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
CHECKPOINT_EVERY = 1000
GENERATE_EVERY = 10000
NUM_EPOCHS = 10000

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

class XTransformerDataset(Dataset):
    def __init__(self, rootdir):
        self.rootdir = rootdir
        self.tars = list(sorted(os.listdir(f'{rootdir}/data/')))[:10]

        self.training_data = []
        self.load_data()

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        src, tgt, mask = self.training_data[idx]
        src = torch.tensor(src, dtype=torch.long)
        tgt = torch.tensor(tgt, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.bool)
        return src, tgt, mask

    def load_data(self):
        while self.tars:
            tar = random.choice(self.tars)
            self.tars.remove(tar)

            if not tar.endswith('.tar.gz'):
                continue

            opt_objs = []
            unopt_objs = []
            print(f"loading {self.rootdir}/data/{tar}")
            with tarfile.open(f"{self.rootdir}/data/{tar}", 'r:gz') as tar:
                for member in tar.getmembers():
                    if not member.name.endswith('.o'):
                        continue

                    if member.name.endswith('.opt.cpp.o'):
                        file_obj = tar.extractfile(member)
                        content = file_obj.read()
                        opt_objs.append(content)
                    elif member.name.endswith('.unopt.cpp.o'):
                        file_obj = tar.extractfile(member)
                        content = file_obj.read()
                        unopt_objs.append(content)

                if len(opt_objs) != len(unopt_objs):
                    print("WARNING: Mismatch in length of opt_objs and unopt_objs.")
                    continue

                for unopt, opt in zip(unopt_objs, opt_objs):
                    unopt_tokens = tokenize(unopt)
                    opt_tokens = tokenize(opt)

                    if len(unopt_tokens) >= ENC_SEQ_LEN or len(opt_tokens) >= DEC_SEQ_LEN:
                        #print(f"skip len unopt {len(unopt_tokens)} opt {len(opt_tokens)}")
                        continue

                    unopt_tokens.insert(0, ENCSTART)
                    unopt_tokens.append(ENCEND)
                    opt_tokens.insert(0, DECSTART)
                    opt_tokens.append(DECEND)
                    mask = [True] * len(unopt_tokens)
                    mask.extend([False] * (ENC_SEQ_LEN - len(unopt_tokens)))
                    unopt_tokens.extend([PAD] * (ENC_SEQ_LEN - len(unopt_tokens)))
                    opt_tokens.extend([PAD] * (DEC_SEQ_LEN - len(opt_tokens)))
                    self.training_data.append((unopt_tokens, opt_tokens, mask))

class XTransformerModule(pl.LightningModule):
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
            enc_depth=8,
            enc_heads=8,
            enc_max_seq_len=ENC_SEQ_LEN,
            dec_num_tokens=NUM_TOKENS,
            dec_depth=8,
            dec_heads=8,
            dec_max_seq_len=DEC_SEQ_LEN
        )
        #self.model = torch.compile(self.model)

    def forward(self, src, tgt, mask):
        return self.model(src, tgt, mask=mask)

    def training_step(self, batch, batch_idx):
        src, tgt, mask = batch
        src, tgt, mask = src.to(self.device), tgt.to(self.device), mask.to(self.device)
        loss = self(src, tgt, mask)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt, mask = batch
        src, tgt, mask = src.to(self.device), tgt.to(self.device), mask.to(self.device)
        loss = self(src, tgt, mask)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        if DEEPSPEED:
            return DeepSpeedCPUAdam(self.parameters())
        else:
            optim = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=100)
            return [optim], [scheduler]

    def generate(self, src, start_tokens, seq_len, mask):
        return self.model.generate(src, start_tokens, seq_len, mask=mask)

    def val_dataloader(self):
        val_dataset = XTransformerDataset(ROOTDIR)
        return DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=multiprocessing.cpu_count())

# Main function
def main():
    dataset = XTransformerDataset(ROOTDIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=multiprocessing.cpu_count())


    model = XTransformerModule()
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator='gpu' if DEVICE == 'cuda' else 'cpu',
        devices=2,
        strategy="deepspeed_stage_3_offload",
        precision="bf16-mixed",
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='val_loss', dirpath='checkpoints', save_top_k=1, mode='min'),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
            pl.callbacks.DeviceStatsMonitor(False),
        ],
        #fast_dev_run=True
    )

    trainer.fit(model, dataloader, val_dataloaders=model.val_dataloader())

    # Generate samples after training
    # Generate samples after training
    #for batch in dataloader:
    #    src, tgt, mask = batch
    #    src, tgt, mask = src.to(DEVICE), tgt.to(DEVICE), mask.to(DEVICE)
    #    start_tokens = torch.tensor([DECSTART]).to(DEVICE)
    #    sample = model.generate(src, start_tokens, DEC_SEQ_LEN, mask=mask)
    #    print_stmt = f"\ninput tokenized:  \n{detokenize(src.tolist()[0])} \n"
    #    print_stmt += f"\npredicted detokenized:  \n{detokenize(sample.tolist())}\n"
    #    print_stmt += f"\nactual detokenized:     \n{detokenize(tgt.tolist()[0])}\n"
    #    print(print_stmt)

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    main()