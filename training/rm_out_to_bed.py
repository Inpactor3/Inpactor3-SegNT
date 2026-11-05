"""Convierte una salida .out de RepeatMasker a BED (name|Class/Family)."""
import argparse
import gzip


def abrir(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bed", required=True)
    args = ap.parse_args()
    n = 0
    with abrir(args.out) as f, open(args.bed, "w") as w:
        for line in f:
            partes = line.split()
            if len(partes) < 15 or not partes[0].isdigit():
                continue
            chrom, ini, fin = partes[4], partes[5], partes[6]
            hebra = "-" if partes[8] == "C" else "+"
            nombre, clase = partes[9], partes[10]
            w.write(f"{chrom}\t{int(ini)-1}\t{fin}\t{nombre}|{clase}\t0\t{hebra}\n")
            n += 1
    print(f"{n} elementos -> {args.bed}")


if __name__ == "__main__":
    main()
