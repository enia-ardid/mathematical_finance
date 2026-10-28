"""Prepare an isolated frozen runtime and a macOS launchd job; do not install it."""

import argparse
from pathlib import Path
import plistlib
import shutil
import sys

from run_prospective import CODE, ROOT, read_protocol, sha, write_once


def prepare(study, runtime):
    study=Path(study).resolve();runtime=Path(runtime).resolve()
    protocol=read_protocol(study)
    runtime.mkdir(parents=True,exist_ok=False)
    for name in CODE:
        shutil.copy2(ROOT/name,runtime/name)
        if sha(runtime/name)!=protocol['code_sha256'][name]:raise ValueError('Runtime copy differs from frozen source.')
    logs=study/'logs';logs.mkdir(exist_ok=True)
    label='org.enia.mathematical-finance.prospective-v1'
    job=dict(Label=label,ProgramArguments=[sys.executable,str(runtime/'run_prospective.py'),'tick','--study',str(study)],
             WorkingDirectory=str(runtime),StartCalendarInterval={'Minute':15},RunAtLoad=True,
             StandardOutPath=str(logs/'collector.out.log'),StandardErrorPath=str(logs/'collector.err.log'),
             EnvironmentVariables={'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
             ProcessType='Background',Nice=10)
    path=study/(label+'.plist')
    with path.open('xb') as stream:plistlib.dump(job,stream)
    with path.open('rb') as stream:
        if plistlib.load(stream)!=job:raise ValueError('Invalid launchd configuration.')
    write_once(study/'runtime.json',dict(runtime=str(runtime),job=str(path),protocol_sha256=sha(study/'protocol.json'),
                                        state='prepared; installation requires a separate launchctl action'))
    print(f'Prepared {path}; no background service installed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'artifacts/prospective-v1')
    parser.add_argument('--runtime',type=Path,default=ROOT/'artifacts/prospective-runtime-v1')
    args=parser.parse_args();prepare(args.study,args.runtime)
