import subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
for args in [('merge.py',),('validate.py','candidate'),('validate.py','merged')]:
 print('START',*args,flush=True);subprocess.run([sys.executable,'-u',str(HERE/args[0]),*args[1:]],check=True)
print('RELEASE VALIDATION COMPLETE',flush=True)
