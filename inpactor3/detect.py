"""Detector de regiones con LTR, pensado como modulo integrable en Inpactor2.

Recibe un genoma en FASTA y devuelve un BED con las regiones marcadas y su probabilidad.
No escribe pesos ni depende de datos confidenciales: solo usa un checkpoint y un genoma.

Ejemplos:
  python -m inpactor3.detect --genome g.fa --head best.pt --out g.bed --dry-run
  python -m inpactor3.detect --genome g.fa --head best.pt --out g.bed --threshold 0.5
"""
import argparse
import gzip
import os
import re
import sys
import tempfile
from pathlib import Path

W_DEF, CHUNK = 50000, 6000


def leer_genoma(path):
    op = gzip.open if str(path).endswith(".gz") else open
    seqs, nombre, buf = {}, None, []
    with op(path, "rt", errors="ignore") as f:
        for linea in f:
            if linea.startswith(">"):
                if nombre:
                    seqs[nombre] = "".join(buf).upper()
                nombre = linea[1:].split()[0]
                buf = []
            else:
                buf.append(linea.strip())
        if nombre:
            seqs[nombre] = "".join(buf).upper()
    return seqs


def limpiar(seq):
    """El lector solo entiende A, C, G, T y N."""
    return re.sub(r"[^ACGTN]", "N", seq.upper())


def offsets(chunks, window, chunk=CHUNK):
    """Mismo calculo que training/block_layout.block_offsets (redondeo)."""
    span = window - chunk
    n = max(chunks - 1, 1)
    return [round(i * span / n) for i in range(chunks)]


def contar_ventanas(genoma, window, chroms=None, limit=0):
    total = 0
    for chrom, seq in genoma.items():
        if chroms and chrom not in chroms:
            continue
        total += max(0, (len(seq) - window) // window + 1)
        if limit and total >= limit:
            return limit
    return total


def ventanas(genoma, window, offs, chroms=None, limit=0):
    """Generador: no materializa todos los bloques de golpe."""
    generadas = 0
    for chrom, seq in genoma.items():
        if chroms and chrom not in chroms:
            continue
        for st in range(0, len(seq) - window + 1, window):
            bloques = [seq[st + o: st + o + CHUNK] for o in offs]
            yield chrom, st, st + window, [limpiar(b) for b in bloques]
            generadas += 1
            if limit and generadas >= limit:
                return


class HeadLayers:
    """Solo las capas; el tipo de agregacion se decide fuera."""

    def __new__(cls, hidden):
        import torch.nn as nn

        class _Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool_proj = nn.Linear(hidden, 512)
                self.dropout = nn.Dropout(0.2)
                self.cls = nn.Sequential(nn.Linear(512, 256), nn.GELU(),
                                         nn.Dropout(0.2), nn.Linear(256, 1))
        return _Head()


def agregar(head, per, modo):
    """per: (B, C, H). Devuelve logits (B,)."""
    import torch
    hidden = per.shape[-1]
    if modo.startswith("logits"):
        f = torch.relu(head.pool_proj(per.reshape(-1, hidden)))
        return head.cls(f).view(per.shape[0], per.shape[1]).max(dim=1).values
    pooled = per.max(dim=1).values
    f = torch.relu(head.pool_proj(pooled))
    return head.cls(f).squeeze(-1)


def escribir_atomico(path, lineas):
    d = os.path.dirname(os.path.abspath(path)) or "."
    with tempfile.NamedTemporaryFile("w", dir=d, delete=False, prefix=".tmp_",
                                     suffix=os.path.basename(path)) as tmp:
        tmp.writelines(lineas)
        nombre = tmp.name
    os.replace(nombre, path)


def escribir_bed(filas, path):
    escribir_atomico(path, (f"{c}\t{s}\t{e}\tinpactor3_p{p:.3f}\t{int(p * 1000)}\t.\n"
                            for c, s, e, p in filas))


def escribir_gff(filas, path):
    lineas = ["##gff-version 3\n"]
    lineas += (f"{c}\tinpactor3\tLTR_region\t{s + 1}\t{e}\t.\t.\t.\t"
               f"Name=LTR_region;probability={p:.4f}\n" for c, s, e, p in filas)
    escribir_atomico(path, lineas)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Detector de regiones con LTR")
    ap.add_argument("--genome", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--out", required=True, help="BED de salida")
    ap.add_argument("--gff", default="", help="GFF3 opcional")
    ap.add_argument("--threshold", type=float, default=None,
                    help="si se omite se usa el del checkpoint")
    ap.add_argument("--window", type=int, default=W_DEF)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2, help="ventanas por paso")
    ap.add_argument("--chroms", default="")
    ap.add_argument("--limit", type=int, default=0, help="maximo de ventanas (pruebas)")
    ap.add_argument("--backbone", choices=("auto", "v2", "ntv3"), default="auto")
    ap.add_argument("--nt-model", default="")
    ap.add_argument("--allow-empty", action="store_true",
                    help="permitir salida vacia cuando no hay ventanas")
    ap.add_argument("--dry-run", action="store_true", help="solo contar ventanas")
    args = ap.parse_args(argv)

    genoma = leer_genoma(args.genome)
    chroms = set(args.chroms.split(",")) if args.chroms else None
    offs = offsets(args.chunks, args.window)
    n_ventanas = contar_ventanas(genoma, args.window, chroms, args.limit)
    print(f"genoma: {len(genoma)} secuencias · ventanas: {n_ventanas} · "
          f"bloques por ventana: {len(offs)} · ventana: {args.window} pb", file=sys.stderr)

    if n_ventanas == 0:
        if not args.allow_empty:
            print("sin ventanas: no se escribe la salida (usa --allow-empty si es "
                  "intencional)", file=sys.stderr)
            return 2
        escribir_bed([], args.out)
        return 0
    if args.dry_run:
        print(f"[dry-run] se evaluarian {n_ventanas} ventanas y se escribiria {args.out}",
              file=sys.stderr)
        return 0

    import numpy as np
    import torch
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.head, map_location=dev, weights_only=False)
    hidden = int(ck.get("hidden", 512))
    modo = str(ck.get("mode", "logits_max"))
    thr = args.threshold if args.threshold is not None else float(ck.get("threshold", 0.5))

    backbone = "ntv3" if args.backbone == "ntv3" else (
        "v2" if args.backbone == "v2" else ("ntv3" if "NTv3" in args.nt_model else "v2"))
    repo = args.nt_model or ("InstaDeepAI/NTv3_100M_pre" if backbone == "ntv3"
                             else "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species")

    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    if backbone == "ntv3":
        model = AutoModelForMaskedLM.from_pretrained(repo, trust_remote_code=True)
    else:
        try:
            model = AutoModel.from_pretrained(repo, trust_remote_code=True)
        except Exception:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            cls = get_class_from_dynamic_module("modeling_esm.EsmModel", repo)
            model = cls.from_pretrained(repo)
    model = model.eval().to(dev)

    head = HeadLayers(hidden).to(dev)
    head.load_state_dict(ck["head"])
    head.eval()
    mu = (torch.as_tensor(ck["mu"], device=dev, dtype=torch.float32)
          if ck.get("mu") is not None else None)
    sd = (torch.as_tensor(ck["sd"], device=dev, dtype=torch.float32)
          if ck.get("sd") is not None else None)

    filas = []
    lote = []
    hechas = 0

    def procesar(lote):
        planos = [b for _, _, _, bl in lote for b in bl]
        if backbone == "ntv3":
            enc = tok(planos, add_special_tokens=False, padding=True,
                      pad_to_multiple_of=128, return_tensors="pt").to(dev)
            with torch.no_grad():
                h = model(**enc).last_hidden_state[:, :CHUNK].mean(dim=1)
        else:
            enc = tok(planos, return_tensors="pt", padding="max_length",
                      truncation=True, max_length=1024).to(dev)
            with torch.no_grad():
                out = model(input_ids=enc["input_ids"],
                            attention_mask=enc["attention_mask"]).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                h = (out * m).sum(1) / m.sum(1).clamp(min=1)
        if mu is not None:
            h = (h.float() - mu) / sd
        per = h.view(len(lote), len(offs), -1)
        with torch.no_grad():
            lg = agregar(head, per, modo)
        return torch.sigmoid(lg).float().cpu().numpy()

    for item in ventanas(genoma, args.window, offs, chroms, args.limit):
        lote.append(item)
        if len(lote) == args.batch:
            probs = procesar(lote)
            filas += [(c, s, e, float(p)) for (c, s, e, _), p in zip(lote, probs) if p >= thr]
            hechas += len(lote)
            print(f"  {hechas}/{n_ventanas}", file=sys.stderr)
            lote = []
    if lote:
        probs = procesar(lote)
        filas += [(c, s, e, float(p)) for (c, s, e, _), p in zip(lote, probs) if p >= thr]
        hechas += len(lote)
        print(f"  {hechas}/{n_ventanas}", file=sys.stderr)

    escribir_bed(filas, args.out)
    if args.gff:
        escribir_gff(filas, args.gff)
    print(f"listo: {len(filas)} regiones sobre {hechas} ventanas "
          f"(umbral {thr:.3f}) -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
