import multiprocessing
from  subprocess import PIPE, run
import os
import platform
import multiprocessing.dummy
import tarfile
import shutil
import heapq
from collections import defaultdict
import os
import glob
from collections import defaultdict
import heapq
import base64
import tokenizers
from pathlib import Path


from util import randstring, chunkify
from util import ROOTDIR, TMP, HOMEDIR

CCFLAGS = '-Wall -fcf-protection=none -fno-asynchronous-unwind-tables -fno-unwind-tables -march=znver4 '

def gen_yarpgen(_):
  path = f'{TMP}/{randstring(32)}'
  os.makedirs(path, exist_ok=False)
  ret = run(f'{ROOTDIR}/bin/{platform.system()}/yarpgen --std=c++ -o {path}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
  return f"{path}/func.cpp"

def gen_csmith(_):
  ret = run(f'{ROOTDIR}/bin/{platform.system()}/csmith --concise --max-funcs 1 --no-safe-math --nomain'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
  prog = ret.stdout.decode('utf-8')
  path = f'{TMP}/{randstring(32)}.o'
  with open(path,'wt') as f:
    f.write(prog)
  return path

def gen_ldrgen(_):
  ret = run(f'/{HOMEDIR}/.opam/4.14.1/bin/frama-c -ldrgen -ldrgen-int-only'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
  prog = ret.stdout.decode('utf-8')
  path = f'{TMP}/{randstring(32)}.o'
  with open(path,'wt') as f:
    f.write(prog)
  return path

def gen_ccg(_):
  ret = run(f'/{ROOTDIR}/bin/{platform.system()}/ccg --max-function 1 --max-localvars 4 --max-function-parameters 8 --min-statements-per-block 1 --max-statements-per-block 4 --max-expression-nesting 4 --max-block-nesting 4'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
  prog = ret.stdout.decode('utf-8')
  path = f'{TMP}/{randstring(32)}.o'
  with open(path,'wt') as f:
    f.write(prog)
  return path

def compile(path, objpath, opt):
  clang = f'clang++-18 -xc++ -stdlib=libc++ ' if path.endswith('.cpp') else f'clang -xc '
  clangret = run(f'{clang} -c {path} -o {objpath} {opt} {CCFLAGS} -include stdint.h'.split(), stdout=PIPE, stderr=PIPE)
  stripret = run(f'llvm-strip-18 {objpath}'.split(), stdout=PIPE, stderr=PIPE)
  objcopyret = run(f'llvm-objcopy-18 --remove-section .eh_frame --remove-section .note.GNU-stack --remove-section .comment --remove-section .llvm_addrsig {objpath}'.split(), stdout=PIPE, stderr=PIPE)
  return f"{path}.o"

def compile_unopt(path):
  ext = ".unopt" + ('.cpp' if path.endswith('.cpp') else '.c')
  return compile(path, f"{path}{ext}.o", "-O0")

def compile_opt(path):
  ext = ".opt" + ('.cpp' if path.endswith('.cpp') else '.c')
  return compile(path, f"{path}{ext}.o", "-O3")

def generate_code():
  shutil.rmtree(TMP, ignore_errors=True)
  ncpu = multiprocessing.cpu_count()
  numtars = 10000
  numruns = 20
  numexisting = len(os.listdir(f"{ROOTDIR}/data"))
  for i in range(numtars):
    print(f"tar {i+numexisting}")
    with multiprocessing.dummy.Pool(ncpu) as p:
      ret = p.map(gen_yarpgen, list(range(ncpu*numruns)))
    with multiprocessing.dummy.Pool(ncpu) as p:
      unopt = p.map(compile_unopt, ret)
    with multiprocessing.dummy.Pool(ncpu) as p:
      opt = p.map(compile_opt, ret)
    dirnum = 0
    for d in os.listdir(TMP):
      os.rename(os.path.join(TMP,d),os.path.join(TMP,str(dirnum)))
      dirnum+=1
    with tarfile.open(f"/tmp/sopt{randstring(32)}.tar.gz", "w:gz") as tar:
      tar.add(TMP, arcname=os.path.basename(TMP))
    shutil.rmtree(TMP)
    print()
    #ccgpret = p.map(gen_ccg, list(range(ncpu)))
    #csmithret = p.map(gen_csmith, list(range(ncpu)))
    #ldrgenret = p.map(gen_ldrgen, list(range(ncpu)))

def bbpe_train(vocab_size):
  i = 0
  os.makedirs(f'{TMP}/opt',exist_ok=True)
  os.makedirs(f'{TMP}/unopt',exist_ok=True)
  for targz in list(os.listdir(f"{ROOTDIR}/data"))[:2]:
    print(i)
    i += 1
    if not targz.endswith('.tar.gz'):
      continue

    with tarfile.open(f"{ROOTDIR}/data/{targz}", 'r:gz') as tar:
      for member in tar.getmembers():
        if not member.name.endswith('opt.cpp.o'):
          continue

        file_obj = tar.extractfile(member)
        content = file_obj.read()
        subfolder = 'unopt' if 'unopt' in member.name else 'opt'
        with open(f'{TMP}/{subfolder}/{randstring(32)}.o','wb+') as f:
          f.write(content)

  paths = [str(x) for x in Path(f"{TMP}/unopt").glob("**/*.o")]
  tokenizer = tokenizers.ByteLevelBPETokenizer()
  tokenizer.train(files=paths, vocab_size=vocab_size, min_frequency=2, special_tokens=[])


def sentencepiece_train(vocab_size):
  os.makedirs(f'{TMP}',exist_ok=True)
  i = 0
  with open(f'{TMP}/sopt_opt.txt','wt+') as optf, open(f'{TMP}/sopt_unopt.txt','wt+') as unoptf:
    for targz in list(os.listdir(f"{ROOTDIR}/data")):
      print(i)
      i+=1
      if not targz.endswith('.tar.gz'):
        continue

      with tarfile.open(f"{ROOTDIR}/data/{targz}", 'r:gz') as tar:
        for member in tar.getmembers():
          if not member.name.endswith('opt.cpp.o'):
            continue

          file_obj = tar.extractfile(member)
          lines = list(chunkify(base64.b64encode(file_obj.read()).decode('ascii'),65000))
          outf = optf if member.name.endswith('.opt.cpp.o') else unoptf
          for line in lines:
            outf.writelines(line + '\n')

  run(f'spm_train --input={TMP}/sopt_unopt.txt --num_threads=32 --train_extremely_large_corpus=1 --max_sentence_length=10000000 --max_sentencepiece_length=16 --unk_id=0 --bos_id=-1 --eos_id=-1 --pad_id=1 --model_type=unigram --model_prefix=encoder --vocab_size=65000 --character_coverage=1.0'.split())
  run(f'spm_train --input={TMP}/sopt_opt.txt --num_threads=32 --train_extremely_large_corpus=1 --max_sentence_length=10000000 --max_sentencepiece_length=16 --unk_id=0 --bos_id=-1 --eos_id=-1 --pad_id=1 --model_type=unigram --model_prefix=decoder --vocab_size=65000 --character_coverage=1.0'.split())


if __name__ == '__main__':
  #generate_code()

  vocab_size = 8192
  sentencepiece_train(vocab_size)
  #bbpe_train(vocab_size)
