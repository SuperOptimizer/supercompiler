import multiprocessing
from subprocess import PIPE, run
import platform
import os
from concurrent.futures import ProcessPoolExecutor
from util import randstring, ROOTDIR, TMP
import tarfile
import shutil

warning_disables = '-Wno-old-style-cast -Wno-c++98-compat-pedantic -Wno-unsafe-buffer-usage -Wno-missing-prototypes -Wno-unused-parameter ' \
    '-Wno-implicit-int-conversion -Wno-unreachable-code -Wno-tautological-compare -Wno-tautological-value-range-compare -Wno-tautological-type-limit-compare' \
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


if __name__ == '__main__':
    generate_and_save_code(num_progs=10000, output_file='compiler_data.tar.gz')