import multiprocessing
from subprocess import PIPE, run
import platform
import os
from concurrent.futures import ProcessPoolExecutor
from util import randstring, ROOTDIR, TMP
import tarfile
import shutil
import base64
import zstandard as zstd

from util import chunkify

warning_disables = '-Wno-old-style-cast -Wno-c++98-compat-pedantic -Wno-unsafe-buffer-usage -Wno-missing-prototypes -Wno-unused-parameter ' \
    '-Wno-implicit-int-conversion -Wno-unreachable-code -Wno-tautological-compare -Wno-tautological-value-range-compare -Wno-tautological-type-limit-compare ' \
    '-Wno-tautological-unsigned-zero-compare'
#shift-sign-overflow
CCFLAGS = f'-Wall -fcf-protection=none -fno-asynchronous-unwind-tables -fno-unwind-tables -march=znver4 -Weverything -fopenmp {warning_disables}'

SUFFIX = '-18'
#SUFFIX = ''



def gen_compile(index):
    path = f'{TMP}/{randstring(32)}'
    os.makedirs(path, exist_ok=False)
    ret = run(f'{ROOTDIR}/bin/{platform.system()}/yarpgen --std=c++ -o {path}'.split(), stdin=PIPE, stdout=PIPE,
              stderr=PIPE)
    clang = f'clang++{SUFFIX} -xc++  '
    unopt_objpath = f'{path}/{index}.unopt.o'
    opt_objpath = f'{path}/{index}.opt.o'

    # Compile unoptimized version
    clangret = run(f'{clang} -c {path}/func.cpp -o {unopt_objpath} -O0 {CCFLAGS} -include stdint.h'.split(),
                   stdout=PIPE, stderr=PIPE)
    if len(clangret.stderr) > 0:
        print(clangret.stderr.decode('utf-8'))
    stripret = run(f'llvm-strip{SUFFIX} {unopt_objpath}'.split(), stdout=PIPE, stderr=PIPE)

    if len(stripret.stderr) > 0:
        print(stripret.stderr.decode('utf-8'))
    objcopyret = run(
        f'llvm-objcopy{SUFFIX} --remove-section .eh_frame --remove-section .note.GNU-stack --remove-section .comment --remove-section .llvm_addrsig {unopt_objpath}'.split(),
        stdout=PIPE, stderr=PIPE)

    if len(objcopyret.stderr) > 0:
        print(objcopyret.stderr.decode('utf-8'))

    # Compile optimized version
    clangret = run(f'{clang} -c {path}/func.cpp -o {opt_objpath} -O3 {CCFLAGS} -include stdint.h'.split(), stdout=PIPE,
                   stderr=PIPE)

    if len(clangret.stderr) > 0:
        print(clangret.stderr.decode('utf-8'))
    stripret = run(f'llvm-strip{SUFFIX} {opt_objpath}'.split(), stdout=PIPE, stderr=PIPE)
    if len(stripret.stderr) > 0:
        print(stripret.stderr.decode('utf-8'))

    objcopyret = run(
        f'llvm-objcopy{SUFFIX} --remove-section .eh_frame --remove-section .note.GNU-stack --remove-section .comment --remove-section .llvm_addrsig {opt_objpath}'.split(),
        stdout=PIPE, stderr=PIPE)
    if len(objcopyret.stderr) > 0:
        print(objcopyret.stderr.decode('utf-8'))

    print(f"Generated {path}")
    return (unopt_objpath, opt_objpath)


def generate_and_save_code(num_progs=10000, output_file='compiler_data.tar.gz'):
    ncpu = multiprocessing.cpu_count()
    i = 0
    with tarfile.open(output_file, "w:gz") as tar:
        with ProcessPoolExecutor(max_workers=ncpu//2) as executor:
            futures = [executor.submit(gen_compile, i) for i in range(num_progs)]
            for i, future in enumerate(futures):
                print(f"{i}")
                unopt_path, opt_path = future.result()

                # Add files to tar
                tar.add(unopt_path, arcname=f'{i}.unopt.o')
                tar.add(opt_path, arcname=f'{i}.opt.o')

                # Clean up the generated files
                os.remove(unopt_path)
                os.remove(opt_path)
                shutil.rmtree(os.path.dirname(unopt_path))

    print(f"Generated and saved {num_progs} pairs to {output_file}")

def zstd_train(dictsize=1024*1024):
  os.makedirs(f'{TMP}/unopt', exist_ok=True)
  os.makedirs(f'{TMP}/opt',exist_ok=True)
  opt_samples = []
  unopt_samples = []
  with tarfile.open(f"{ROOTDIR}/compiler_data.tar.gz", 'r:gz') as tar:
    i = 0
    for member in tar.getmembers():
      i += 1
      print(i)
      if member.name.endswith('.opt.o'):
        l = opt_samples
      elif member.name.endswith('.unopt.o'):
        l = unopt_samples
      l.append(tar.extractfile(member).read())

  unopt_dict = zstd.train_dictionary(dictsize, unopt_samples)
  with open(f'{ROOTDIR}/zstd_enc.dictionary','wb') as f:
      f.write(unopt_dict.as_bytes())
  opt_dict = zstd.train_dictionary(dictsize, unopt_samples)
  with open(f'{ROOTDIR}/zstd_dec.dictionary','wb') as f:
      f.write(opt_dict.as_bytes())

def sentencepiece_train():
  os.makedirs(f'{TMP}',exist_ok=True)
  with open(f'{TMP}/sopt_opt.txt','wt+') as optf, open(f'{TMP}/sopt_unopt.txt','wt+') as unoptf:
      with tarfile.open(f"{ROOTDIR}/compiler_data.tar.gz", 'r:gz') as tar:
        i = 0
        for member in tar.getmembers():
          i +=1
          print(i)
          if not member.name.endswith('opt.o'):
            continue

          file_obj = tar.extractfile(member)
          lines = list(chunkify(base64.b64encode(file_obj.read()).decode('ascii'),65000))
          outf = optf if member.name.endswith('.opt.o') else unoptf
          for line in lines:
            outf.writelines(line + '\n')
  #print("training unopt tokenizer")
  #run(f'spm_train --input={TMP}/sopt_unopt.txt --num_threads=8 --model_type=unigram --model_prefix=encoder --vocab_size=65000 --character_coverage=1.0'.split())
  print("training opt tokenizer")
  run(f'spm_train --input={TMP}/sopt_opt.txt --num_threads=8  --model_type=unigram --model_prefix=decoder --vocab_size=65000 --character_coverage=1.0'.split())

if __name__ == '__main__':
    pass
    zstd_train()
    #sentencepiece_train()
    #generate_and_save_code(num_progs=20000, output_file='compiler_data.tar.gz')