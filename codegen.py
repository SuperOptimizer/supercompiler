import multiprocessing
from  subprocess import PIPE, run
import os
import platform
import multiprocessing.dummy
import tarfile
import shutil


from util import randstring
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
  numtars = 100
  numruns = 100
  for i in range(numtars):
    for j in range(numruns):
      with multiprocessing.dummy.Pool(ncpu) as p:
        ret = p.map(gen_yarpgen, list(range(ncpu)))
      with multiprocessing.dummy.Pool(ncpu) as p:
        unopt = p.map(compile_unopt, ret)
      with multiprocessing.dummy.Pool(ncpu) as p:
        opt = p.map(compile_opt, ret)
    dirnum = 0
    for d in os.listdir(TMP):
      os.rename(os.path.join(TMP,d),os.path.join(TMP,str(dirnum)))
      dirnum+=1
    with tarfile.open(f"/tmp/sopt{i}.tar.gz", "w:gz") as tar:
      tar.add(TMP, arcname=os.path.basename(TMP))
    shutil.rmtree(TMP)
    print()
    #ccgpret = p.map(gen_ccg, list(range(ncpu)))
    #csmithret = p.map(gen_csmith, list(range(ncpu)))
    #ldrgenret = p.map(gen_ldrgen, list(range(ncpu)))


if __name__ == '__main__':
  generate_code()