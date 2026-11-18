# Colab — precalculo v2 de Animales

Para usar solo si quieres un respaldo fuera de tu 5090.
**Colab gratis (T4) no alcanza**: Animales son ~13-15 h y el límite es 12 h.
Con **Colab Pro (A100 o L4)** son ~3-5 h.

---

## Celda 1 — montar Drive y dependencias

```python
from google.colab import drive
drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive/Inpactor3_v2'
import os
os.makedirs(ROOT, exist_ok=True)

!pip install -q "transformers==4.46.3" kaggle
import torch
print(torch.__version__, torch.cuda.get_device_name(0))
```

## Celda 2 — token de Kaggle

```python
import os, getpass
os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
tok = getpass.getpass('Kaggle API token: ')
open(os.path.expanduser('~/.kaggle/access_token'), 'w').write(tok.strip())
os.chmod(os.path.expanduser('~/.kaggle/access_token'), 0o600)
```

## Celda 3 — datos y código

```python
!mkdir -p /content/data
!kaggle datasets download -d cristianosoriog/inpactor2-animales -p /content/dl --quiet
import zipfile, glob
for z in glob.glob('/content/dl/*.zip'):
    with zipfile.ZipFile(z) as f:
        f.extractall('/content/data')
print(sorted(os.listdir('/content/data')))
```

Si el zip falla con 404 (ya pasó con datasets grandes), baja archivo por archivo:

```python
import json, subprocess, os
D = 'cristianosoriog/inpactor2-animales'
out = subprocess.run(['kaggle','datasets','files','-d',D,'--page-size','200','--format','json'],
                     capture_output=True, text=True, check=True).stdout
for it in json.loads(out):
    if it['name'].endswith('.npz'):
        subprocess.run(['kaggle','datasets','download','-d',D,'-f',it['name'],
                        '-p','/content/dl','--quiet'], check=True)
print('ok')
```

## Celda 4 — código desde GitHub

```python
import os
os.makedirs('/content/code', exist_ok=True)
SHA = '30f4947'
GH = f'https://raw.githubusercontent.com/Inpactor3/Inpactor3-SegNT/{SHA}'
for nombre, ruta in [
    ('precompute_embeddings.py', f'{GH}/training/precompute_embeddings.py'),
    ('subsample_train.py',       f'{GH}/training/subsample_train.py'),
    ('train_mlp.py',             f'{GH}/training/train_mlp.py'),
    ('block_layout.py',          f'{GH}/training/block_layout.py'),
    ('model.py',                 f'{GH}/colab_bridge/src/model.py'),
    ('dataset.py',               f'{GH}/colab_bridge/src/dataset.py'),
]:
    !curl -sL {ruta} -o /content/code/{nombre}
%cd /content/code
import sys; sys.path.insert(0, '/content/code')
print(sorted(os.listdir('.')))
```

## Celda 5 — submuestreo y precalculo

```python
import glob, subprocess, sys, time
SEQ = '/content/data/bundle4_Animales/Animales'
SUB = '/content/trabajo/sub/Animales'
EMB = f'{ROOT}/emb_v2/Animales'
NT  = 'InstaDeepAI/nucleotide-transformer-v2-100m-multi-species'

# lotes: en T4 usa 8/32; en A100 o L4 puedes subir a 32/128
BS, TB = 8, 32

def sh(cmd):
    print('$', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)

sh([sys.executable, 'subsample_train.py', '--seq_dir', SEQ,
    '--out_dir', SUB, '--ratio', '1'])

for splits in ('val,test', 'train'):
    t0 = time.time()
    sh([sys.executable, 'precompute_embeddings.py', '--seq_dir', SUB,
        '--out_dir', EMB, '--splits', splits, '--trust_remote_code',
        '--nt_model', NT, '--max_tokens', '1024',
        '--batch_size', str(BS), '--token_batch', str(TB)])
    print(splits, f'listo en {(time.time()-t0)/3600:.2f} h', flush=True)

for s in ('train','val','test'):
    print(s, len(glob.glob(f'{EMB}/{s}/*_emb.npy')))
```

## Celda 6 — si se corta

El precalculo **se salta las especies ya hechas**, y los embeddings quedan en Drive.
Solo vuelve a ejecutar la celda 5: continúa donde quedó.

Comprueba el avance:

```python
for s in ('train','val','test'):
    print(s, len(glob.glob(f'{EMB}/{s}/*_emb.npy')))
```

## Celda 7 — entrenar el cabezal (cuando esté todo)

```python
!python train_mlp.py --emb_dir {EMB} --out_dir f'{ROOT}/ckpt/Animales_v2backbone' \
  --epochs 100 --patience 10 --batch_size 4096 --balance global \
  --class_weight --calibrate --select balanced_accuracy
```

---

## Si sale `CUDA out of memory`

Baja el lote en la celda 5: `BS, TB = 4, 16`.

## Nota

Con Colab gratis (T4) esto tarda 13-15 h y la sesión se corta a las 12.
Con la 5090 son ~2 h. Úsalo solo como respaldo.
