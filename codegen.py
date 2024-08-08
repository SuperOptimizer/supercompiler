import multiprocessing
from subprocess import PIPE, run
import platform
import os
from concurrent.futures import ProcessPoolExecutor
from util import randstring, ROOTDIR, TMP
import tarfile
import shutil


CCFLAGS = '-Wall -fcf-protection=none -fno-asynchronous-unwind-tables -fno-unwind-tables -march=znver3 '


def gen_compile(index):
    path = f'{TMP}/{randstring(32)}'
    os.makedirs(path, exist_ok=False)
    ret = run(f'{ROOTDIR}/bin/{platform.system()}/yarpgen --std=c++ -o {path}'.split(), stdin=PIPE, stdout=PIPE,
              stderr=PIPE)
    clang = f'clang++ -xc++  '
    unopt_objpath = f'{path}/{index}.unopt.o'
    opt_objpath = f'{path}/{index}.opt.o'

    # Compile unoptimized version
    clangret = run(f'{clang} -c {path}/func.cpp -o {unopt_objpath} -O0 {CCFLAGS} -include stdint.h'.split(),
                   stdout=PIPE, stderr=PIPE)
    stripret = run(f'llvm-strip {unopt_objpath}'.split(), stdout=PIPE, stderr=PIPE)
    objcopyret = run(
        f'llvm-objcopy --remove-section .eh_frame --remove-section .note.GNU-stack --remove-section .comment --remove-section .llvm_addrsig {unopt_objpath}'.split(),
        stdout=PIPE, stderr=PIPE)

    # Compile optimized version
    clangret = run(f'{clang} -c {path}/func.cpp -o {opt_objpath} -O3 {CCFLAGS} -include stdint.h'.split(), stdout=PIPE,
                   stderr=PIPE)
    stripret = run(f'llvm-strip {opt_objpath}'.split(), stdout=PIPE, stderr=PIPE)
    objcopyret = run(
        f'llvm-objcopy --remove-section .eh_frame --remove-section .note.GNU-stack --remove-section .comment --remove-section .llvm_addrsig {opt_objpath}'.split(),
        stdout=PIPE, stderr=PIPE)

    print(f"Generated {path}")
    return (unopt_objpath, opt_objpath)


def generate_and_save_code(num_progs=10000, output_file='compiler_data.tar.gz'):
    ncpu = multiprocessing.cpu_count()
    i = 0
    with tarfile.open(output_file, "w:gz") as tar:
        with ProcessPoolExecutor(max_workers=ncpu) as executor:
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