# Celdas para pegar en Colab (una sola vez, en navegador)

## Antes de empezar
1. Abrir https://colab.research.google.com en navegador
2. Nuevo notebook → Entorno de ejecución → Cambiar tipo → **GPU T4**
3. Pegar las celdas de abajo **una a la vez** y ejecutar.

Después de la celda 2 ya podés cerrar el navegador y trabajar 100% desde VSCode.

---

## Celda 1 · Instalar SSH + cloudflared

```python
import os, subprocess, random, string, time

# Password SSH aleatorio para esta sesion
SSH_PASS = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
print(f"\n>>> SSH password para esta sesion: {SSH_PASS}\n")

# Instalar openssh y configurar root
subprocess.run("apt-get -qq update && apt-get -qq install -y openssh-server", shell=True)
subprocess.run(f"echo 'root:{SSH_PASS}' | chpasswd", shell=True)
subprocess.run("sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config", shell=True)
subprocess.run("sed -i 's/#PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config", shell=True)
subprocess.run("service ssh start", shell=True)
print("ssh server activo")

# Instalar cloudflared (tunel sin cuenta)
subprocess.run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared", shell=True)
subprocess.run("chmod +x /usr/local/bin/cloudflared", shell=True)
print("cloudflared instalado")
```

---

## Celda 2 · Levantar tunel SSH y mostrar comando

```python
import subprocess, time, re, os

# Lanzar cloudflared en background con TCP sobre puerto 22
log_file = "/tmp/cloudflared.log"
os.system(f"nohup cloudflared tunnel --url tcp://localhost:22 > {log_file} 2>&1 &")

# Esperar a que aparezca la URL
print("Esperando URL del tunel (10-20 s)...")
tunel_url = None
for _ in range(30):
    time.sleep(2)
    try:
        with open(log_file) as f:
            data = f.read()
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", data)
        if m:
            tunel_url = m.group(0)
            break
    except FileNotFoundError:
        pass

if not tunel_url:
    print("ERROR: no se encontro URL. Mira /tmp/cloudflared.log")
else:
    host = tunel_url.replace("https://", "")
    print("\n" + "="*60)
    print(" TUNEL LISTO · copia este comando en VSCode Remote-SSH:")
    print("="*60)
    print(f"\n ssh -o ProxyCommand='cloudflared access tcp --hostname {host}' root@{host}")
    print(f"\n Password: {SSH_PASS}")
    print("="*60)
```

El output te da el comando SSH exacto. Copialo. Pasá a VSCode.

---

## Celda 3 (opcional) · Pre-instalar librerias pesadas para que esten listas

Pegá esta celda y ejecutá — tarda ~2 min. Después ya podés cerrar el navegador.

```python
!pip install -q transformers torch biopython numpy tqdm
!nvidia-smi | head -15
print("\nTodo instalado. Ya podes cerrar el navegador y conectar desde VSCode.")
```

---

# En tu máquina local (VSCode)

## 1. Instalar cloudflared local (una sola vez)

```bash
sudo wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
sudo chmod +x /usr/local/bin/cloudflared
```

## 2. Conectar desde VSCode

1. Instalar extension **Remote - SSH** (si no la tenés).
2. `Ctrl+Shift+P` → `Remote-SSH: Connect to Host` → `Add New SSH Host...`
3. Pegar el comando SSH que te dio la celda 2.
4. Al conectar, te pide el password → el de la celda 1.
5. Ya estás dentro del runtime Colab desde VSCode. GPU accesible con `!nvidia-smi` o `torch.cuda`.

## 3. Subir los JSONL de hongos cuando esten listos

Desde una terminal local (NO el VSCode remoto):

```bash
# Reemplaza XXX con el host del tunel
scp -o ProxyCommand='cloudflared access tcp --hostname XXX.trycloudflare.com' \
    /media/cami/sdb2-usb-WDC_WD10_JPCX-24/InpactorRM/windows/Hongos/train/*.jsonl \
    root@XXX.trycloudflare.com:/content/hongos_train/
```

Pero más fácil: usá `embed_pipeline.py` (en esta misma carpeta) que corre dentro del Colab y procesa un directorio local tuyo via SSHFS reverso o descarga directa.

---

# Notas importantes

- Colab apaga runtime tras 90 min de inactividad. Si trabajás activo, dura 12 h máx (GPU gratis) o 24 h (créditos Pro).
- El password SSH es aleatorio cada sesión → no lo guardes en memoria permanente.
- Si cierras el navegador y el runtime sigue vivo, VSCode queda conectado hasta que Google lo termine.
- Para persistir resultados sin Drive: `scp` de vuelta a tu PC local al final de la sesión.
