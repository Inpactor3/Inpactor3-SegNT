"""Ajuste fino parcial del backbone: descongela las ultimas capas del NT-500M.

Diferencia con el entrenamiento normal: aqui NO se usan embeddings en cache. Cada
epoca vuelve a pasar las secuencias por el backbone, por eso cuesta horas.

Que se entrena:
  * las ultimas `--unfreeze_layers` capas del transformer (por defecto 4)
  * el cabezal MLP
  El resto del backbone queda congelado.

Uso (prueba rapida primero):
  python finetune_backbone.py --seq_dir datos/bundle4_Animales/Animales \
      --out_dir trabajo/finetune --smoke

Uso real:
  python finetune_backbone.py --seq_dir datos/bundle4_Animales/Animales \
      --out_dir trabajo/finetune --max_windows 150000 --epochs 3

Se puede reanudar: si existe checkpoint en out_dir, continua desde esa epoca.
"""
import argparse
import glob
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

NT = "InstaDeepAI/nucleotide-transformer-500m-human-ref"


class Head(nn.Module):
    def __init__(self, hidden=1280, dropout=0.2):
        super().__init__()
        self.pool_proj = nn.Linear(hidden, 512)
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, x):
        f = self.dropout(torch.relu(self.pool_proj(x)))
        return self.cls_head(f).squeeze(-1)


class Modelo(nn.Module):
    def __init__(self, unfreeze_layers):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(NT)
        for p in self.backbone.parameters():
            p.requires_grad = False
        capas = self.backbone.encoder.layer
        n = len(capas)
        self.n_capas = n
        for capa in capas[max(0, n - unfreeze_layers):]:
            for p in capa.parameters():
                p.requires_grad = True
        # la normalizacion final tambien se entrena
        for p in self.backbone.encoder.emb_layer_norm_after.parameters():
            p.requires_grad = True
        self.head = Head(self.backbone.config.hidden_size)

    def forward(self, ids, mask, n_bloques):
        out = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1)
        B = ids.shape[0] // n_bloques
        por_ventana = pooled.view(B, n_bloques, -1)
        return self.head(por_ventana.max(dim=1).values)

    def params_entrenables(self, lr_backbone, lr_head):
        bb, hd = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (hd if name.startswith("head.") else bb).append(p)
        return [{"params": bb, "lr": lr_backbone},
                {"params": hd, "lr": lr_head}]


def tokens_de_npz(files, tokenizer, max_tokens, n_bloques, limite, seed):
    """Devuelve (ids int16 (N*nb, max_tokens), labels (N,), pesos)."""
    rng = random.Random(seed)
    idx = list(range(len(files)))
    rng.shuffle(idx)
    ids_out, labs = [], []
    for i in idx:
        z = np.load(files[i], allow_pickle=True)
        seq, lab = z["seq"], z["label"]
        if len(seq) == 0:
            continue
        take = list(range(len(seq)))
        rng.shuffle(take)
        for j in take:
            if limite and len(labs) >= limite:
                break
            bloque = [seq[j, b].tobytes().decode("ascii", "ignore")
                      for b in range(min(n_bloques, seq.shape[1]))]
            enc = tokenizer(bloque, truncation=True, padding="max_length",
                            max_length=max_tokens, return_tensors="np")
            ids_out.append(enc["input_ids"].astype(np.int16))
            labs.append(int(lab[j]))
        if limite and len(labs) >= limite:
            break
    if not labs:
        raise SystemExit("no se cargaron ventanas")
    return np.stack(ids_out), np.array(labs, dtype=np.float32)


def evaluar(modelo, ids, labs, n_bloques, device, batch=8, max_tokens=1000):
    modelo.eval()
    probs = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16,
                                         enabled=(device == "cuda")):
        for i in range(0, len(labs), batch):
            x = torch.from_numpy(ids[i:i + batch].astype(np.int64)).to(device)
            m = torch.ones_like(x)
            lg = modelo(x, m, n_bloques).float()
            probs.append(torch.sigmoid(lg).cpu().numpy())
    return np.concatenate(probs)


def metricas(p, y, thr=0.5):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    rec = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    order = np.argsort(-p); yy = y[order]
    tpc = np.cumsum(yy); fpc = np.cumsum(1 - yy)
    pc = tpc / np.maximum(tpc + fpc, 1); rc = tpc / max(int(y.sum()), 1)
    ap = float(np.sum(np.diff(np.concatenate([[0], rc])) * pc))
    return {"balanced_accuracy": (rec + spec) / 2, "f1": f1, "precision": prec,
            "recall": rec, "specificity": spec, "pr_auc": ap, "n": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_windows", type=int, default=150000)
    ap.add_argument("--val_windows", type=int, default=10000)
    ap.add_argument("--test_windows", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--unfreeze_layers", type=int, default=4)
    ap.add_argument("--lr_backbone", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--batch_windows", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="prueba rapida de 2 minutos")
    args = ap.parse_args()

    if args.smoke:
        args.max_windows, args.val_windows, args.test_windows = 80, 40, 40
        args.epochs, args.batch_windows = 1, 2

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"device {device} · smoke={args.smoke}", flush=True)

    tok = AutoTokenizer.from_pretrained(NT)
    tr = sorted(glob.glob(os.path.join(args.seq_dir, "train", "*.npz")))
    va = sorted(glob.glob(os.path.join(args.seq_dir, "val", "*.npz")))
    te = sorted(glob.glob(os.path.join(args.seq_dir, "test", "*.npz")))
    nb = int(np.load(tr[0], allow_pickle=True)["seq"].shape[1]) if tr else 4
    print(f"bloques por ventana: {nb}", flush=True)

    print("tokenizando train...", flush=True)
    t0 = time.time()
    ids_tr, y_tr = tokens_de_npz(tr, tok, args.max_tokens, nb, args.max_windows, args.seed)
    print(f"  {ids_tr.shape} en {time.time()-t0:.0f}s · pos {int(y_tr.sum())}", flush=True)
    ids_va, y_va = tokens_de_npz(va, tok, args.max_tokens, nb, args.val_windows, args.seed + 1)
    ids_te, y_te = tokens_de_npz(te, tok, args.max_tokens, nb, args.test_windows, args.seed + 2)
    print(f"  val {ids_va.shape} · test {ids_te.shape}", flush=True)

    modelo = Modelo(args.unfreeze_layers).to(device)
    n_tr = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"parametros entrenables: {n_tr/1e6:.1f} M", flush=True)

    opt = torch.optim.AdamW(modelo.params_entrenables(args.lr_backbone, args.lr_head),
                            weight_decay=1e-4)
    pos = float(y_tr.sum()); neg = float(len(y_tr) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ep_ini, mejor = 0, -1.0
    ck_path = os.path.join(args.out_dir, "ckpt.pt")
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=device)
        modelo.load_state_dict(ck["modelo"], strict=False)
        opt.load_state_dict(ck["opt"])
        ep_ini, mejor = ck["epoch"], ck["mejor"]
        print(f"reanudando desde epoca {ep_ini} (mejor val {mejor:.4f})", flush=True)

    historial = []
    for ep in range(ep_ini, args.epochs):
        modelo.train()
        perm = np.random.permutation(len(y_tr))
        t0 = time.time()
        tot, n = 0.0, 0
        for i in range(0, len(perm), args.batch_windows):
            b = perm[i:i + args.batch_windows]
            x = torch.from_numpy(ids_tr[b].reshape(-1, args.max_tokens).astype(np.int64)).to(device)
            y = torch.from_numpy(y_tr[b]).to(device)
            m = torch.ones_like(x)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(device == "cuda")):
                lg = modelo(x, m, nb)
                loss = loss_fn(lg.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in modelo.parameters() if p.requires_grad], 1.0)
            opt.step()
            tot += loss.item() * len(b); n += len(b)
            if (i // args.batch_windows) % 50 == 0:
                print(f"  ep{ep+1} {i}/{len(perm)} loss {tot/max(n,1):.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        pv = evaluar(modelo, ids_va, y_va, nb, device)
        mv = metricas(pv, y_va)
        print(f"ep {ep+1}: loss {tot/n:.4f} · val BA {mv['balanced_accuracy']:.4f} "
              f"F1 {mv['f1']:.4f} · {time.time()-t0:.0f}s", flush=True)
        historial.append({"epoch": ep + 1, "loss": tot / n, **mv})

        n_capas = modelo.n_capas
        prefijos = [f"backbone.encoder.layer.{n}"
                    for n in range(max(0, n_capas - args.unfreeze_layers), n_capas)]
        trainables = {k: v for k, v in modelo.state_dict().items()
                      if any(k.startswith(pr) for pr in prefijos)
                      or k.startswith("head.")
                      or "emb_layer_norm_after" in k}
        torch.save({"modelo": trainables, "opt": opt.state_dict(), "epoch": ep + 1,
                    "mejor": max(mejor, mv["balanced_accuracy"]),
                    "unfreeze_layers": args.unfreeze_layers}, ck_path)
        with open(os.path.join(args.out_dir, "historial.json"), "w") as f:
            json.dump(historial, f, indent=2)
        mejor = max(mejor, mv["balanced_accuracy"])

    pt = evaluar(modelo, ids_te, y_te, nb, device)
    mt = metricas(pt, y_te)
    print(f"\nTEST: BA {mt['balanced_accuracy']:.4f} · F1 {mt['f1']:.4f} · "
          f"P {mt['precision']:.4f} · R {mt['recall']:.4f} · PR-AUC {mt['pr_auc']:.4f}")
    with open(os.path.join(args.out_dir, "test.json"), "w") as f:
        json.dump(mt, f, indent=2)


if __name__ == "__main__":
    main()
