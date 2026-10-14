"""Muestreo de las 5 categorias de ventana NEGATIVA (sin TE).

MI DIFERENCIADOR frente a Jhonier y Alejandro.

Las ventanas negativas se clasifican en 5 tipos usando:
- BED limpio de RepeatMasker (annotations/<species>.bed)
- GFF de referencia con genes (data/reference_annotations/<species>.gff3)

Categorias:
    1. Intergenic puro      30%  (sin TE, sin gen, no en buffer)
    2. Genic                20%  (dentro de un gen)
    3. Other_TE (no-LTR)    20%  (solapa con TE pero NO con LTR)
    4. Heterochromatin      15%  (regiones centromericas/pericentromericas)
    5. Simple_repeats       15%  (del .excluded.bed - simple_repeat o low_complexity)

El TIPO se guarda como metadata — todas colapsan a label=0.

Uso:
    python scripts/sample_5_negative_types.py <species> [--n_total 5000]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANN_DIR = ROOT / "annotations"
GFF_DIR = ROOT / "data" / "reference_annotations"
GENOME_PATHS = {
    "arabidopsis":  ROOT / "data" / "tair10_full"    / "TAIR10.fasta",
    "scerevisiae":  ROOT / "data" / "pilot_genomes"  / "scerevisiae.fasta",
    "celegans":     ROOT / "data" / "pilot_genomes"  / "celegans.fasta",
    "brachypodium": ROOT / "data" / "pilot_genomes"  / "brachypodium.fasta",
    "neurospora":   ROOT / "data" / "pilot_genomes"  / "neurospora.fasta",
    "drosophila":   ROOT / "data" / "pilot_genomes"  / "drosophila.fasta",
}

GFF_PATHS = {
    "arabidopsis":  GFF_DIR / "tair10.gff3",
    "scerevisiae":  GFF_DIR / "scerevisiae.gff3",
    "celegans":     GFF_DIR / "celegans.gff3",
}

CATEGORIES = {
    "intergenic":    0.30,
    "genic":         0.20,
    "other_te":      0.20,
    "heterochrom":   0.15,
    "simple_rep":    0.15,
}

WINDOW_SIZE   = 50_000  # 50 kb
BUFFER_BP     = 1_000   # buffer >=1 kb desde TE
MIN_FRACTION  = 0.1     # fraccion minima de la categoria dentro de la ventana


def read_genome_sizes(fa: Path) -> dict[str, int]:
    sizes = {}
    cur = None
    acc = 0
    with fa.open() as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    sizes[cur] = acc
                cur = line[1:].strip().split()[0]
                acc = 0
            else:
                acc += len(line.strip())
    if cur:
        sizes[cur] = acc
    return sizes


def read_bed(bed_file: Path):
    """Lee BED simple -> dict chrom -> lista de (start, end, name)."""
    hits = {}
    if not bed_file.exists():
        return hits
    with bed_file.open() as f:
        for line in f:
            s = line.strip().split("\t")
            if len(s) < 3:
                continue
            chrom = s[0]
            start = int(s[1])
            end = int(s[2])
            name = s[3] if len(s) > 3 else ""
            hits.setdefault(chrom, []).append((start, end, name))
    for k in hits:
        hits[k].sort()
    return hits


def normalize_chrom(name: str) -> str:
    """Normaliza nombres de cromosoma a formato comun."""
    n = name.strip()
    # Quitar prefijo Chr, chr, CHR
    for pref in ("Chr", "chr", "CHR"):
        if n.startswith(pref):
            n = n[len(pref):]
    return n


# Mapeo de accessions NCBI a nombres de cromosoma para especies piloto
NCBI_TO_CHR = {
    # C. elegans (WBcel235)
    "NC_003279.8":  "I",
    "NC_003280.10": "II",
    "NC_003281.10": "III",
    "NC_003282.8":  "IV",
    "NC_003283.11": "V",
    "NC_003284.9":  "X",
    "NC_001328.1":  "MtDNA",
}


def chrom_match_set(chrom: str) -> list[str]:
    """Devuelve posibles nombres equivalentes para intentar matchear."""
    out = [chrom]
    norm = normalize_chrom(chrom)
    if norm != chrom:
        out.append(norm)
    if chrom in NCBI_TO_CHR:
        out.append(NCBI_TO_CHR[chrom])
    return out


def read_gff_genes(gff_file: Path):
    """Extrae intervalos de genes del GFF."""
    genes = {}
    if not gff_file.exists():
        return genes
    with gff_file.open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            s = line.strip().split("\t")
            if len(s) < 9:
                continue
            if s[2] != "gene":
                continue
            chrom = s[0]
            start = int(s[3]) - 1
            end = int(s[4])
            genes.setdefault(chrom, []).append((start, end))
    for k in genes:
        genes[k].sort()
    return genes


def lookup_intervals(intervals: dict, chrom: str) -> list:
    """Busca intervalos para una cromosoma probando varias normalizaciones."""
    for key in chrom_match_set(chrom):
        if key in intervals:
            return intervals[key]
    return []


def interval_overlap(a_start, a_end, intervals, extra_buffer=0):
    """¿La ventana [a_start, a_end] solapa con alguno de los intervalos?"""
    for b_start, b_end in intervals:
        bs = max(0, b_start - extra_buffer)
        be = b_end + extra_buffer
        if a_start < be and a_end > bs:
            return True
    return False


def interval_fraction(a_start, a_end, intervals):
    """Fraccion de la ventana cubierta por los intervalos (0.0 a 1.0)."""
    length = a_end - a_start
    if length <= 0:
        return 0.0
    # Normalizar a tuplas (start, end) -- descarta extras si BED tiene (s,e,name)
    norm = [(iv[0], iv[1]) for iv in intervals]
    merged = []
    for b_start, b_end in sorted(norm):
        s = max(a_start, b_start)
        e = min(a_end, b_end)
        if s < e:
            merged.append((s, e))
    if not merged:
        return 0.0
    # Fusionar solapes
    merged.sort()
    cur_s, cur_e = merged[0]
    covered = 0
    for s, e in merged[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            covered += cur_e - cur_s
            cur_s, cur_e = s, e
    covered += cur_e - cur_s
    return covered / length


def centromere_regions(chrom_sizes: dict) -> dict:
    """Aproximacion: pericentromero = 10% central del cromosoma.

    (Idealmente usar coordenadas de centromero por especie; esto es una
    aproximacion funcional para el piloto.)
    """
    out = {}
    for ch, size in chrom_sizes.items():
        mid = size // 2
        peri = int(size * 0.05)
        out[ch] = [(mid - peri, mid + peri)]
    return out


def classify_window(chrom, start, end, te_hits, genes, simple_hits, pericentros):
    """Clasifica una ventana en 1 de las 5 categorias de negativa."""
    # Si solapa cualquier LTR -> no es negativa valida
    te_in = [h for h in lookup_intervals(te_hits, chrom)
             if h[0] < end and h[1] > start]
    for s, e, name in te_in:
        if "LTR" in name:
            return None  # positiva, descartar

    # Buffer: cualquier TE dentro de +/- BUFFER_BP tambien descarta como "intergenic"
    has_te_in_buffer = interval_overlap(
        start, end,
        [(s, e) for s, e, _ in lookup_intervals(te_hits, chrom)],
        extra_buffer=BUFFER_BP,
    )

    # Cat 3: Other TE (solapa TE pero no LTR)
    if te_in:
        non_ltr_frac = interval_fraction(
            start, end,
            [(s, e) for s, e, name in te_in if "LTR" not in name],
        )
        if non_ltr_frac >= MIN_FRACTION:
            return "other_te"

    # Cat 4: Heterocromatina (dentro de region pericentromerica aproximada)
    if interval_overlap(start, end, lookup_intervals(pericentros, chrom)):
        return "heterochrom"

    # Cat 5: Simple_repeats (solapa con lo marcado como simple_repeat/low_complexity)
    simple_frac = interval_fraction(start, end, lookup_intervals(simple_hits, chrom))
    if simple_frac >= 0.02:  # 2% ya cuenta (simple reps son pequeños)
        return "simple_rep"

    # Fraccion de ventana cubierta por genes
    gene_frac = interval_fraction(start, end, lookup_intervals(genes, chrom))

    # Cat 2: Genic (ventana dominada por un gen -- >=40% cubierto)
    if gene_frac >= 0.4:
        return "genic"

    # Cat 1: Intergenic (poco gen, sin TE y sin buffer)
    if gene_frac < 0.4 and not has_te_in_buffer:
        return "intergenic"

    return None  # queda en el buffer, se descarta


def sample_windows(species: str, n_total: int = 5000, seed: int = 42):
    random.seed(seed)
    print(f"\n=== Muestreando negativas de {species} ===")

    genome = GENOME_PATHS[species]
    bed = ANN_DIR / f"{species}.bed"
    exc_bed = ANN_DIR / f"{species}.excluded.bed"
    gff = GFF_PATHS.get(species)

    if not bed.exists():
        print(f"  No existe {bed}  (correr clean_rm_output.py primero)")
        return

    chrom_sizes = read_genome_sizes(genome)
    te_hits = read_bed(bed)
    simple_hits = read_bed(exc_bed)
    genes = read_gff_genes(gff) if gff else {}
    pericentros = centromere_regions(chrom_sizes)

    print(f"  cromosomas: {len(chrom_sizes)}  "
          f"TE: {sum(len(v) for v in te_hits.values()):,}  "
          f"genes: {sum(len(v) for v in genes.values()):,}")

    # Barrido de ventanas candidatas (sin solape para train)
    candidates_by_cat = {k: [] for k in CATEGORIES}
    for chrom, size in chrom_sizes.items():
        for start in range(0, size - WINDOW_SIZE + 1, WINDOW_SIZE):
            end = start + WINDOW_SIZE
            cat = classify_window(chrom, start, end,
                                   te_hits, genes, simple_hits, pericentros)
            if cat:
                candidates_by_cat[cat].append((chrom, start, end))

    # Mostrar disponibilidad
    print("  disponibilidad por categoria:")
    for cat, candidates in candidates_by_cat.items():
        print(f"    {cat:12s}: {len(candidates):6,} ventanas candidatas")

    # Muestreo con cuotas
    chosen = []
    for cat, frac in CATEGORIES.items():
        want = int(round(n_total * frac))
        avail = candidates_by_cat[cat]
        if len(avail) >= want:
            picked = random.sample(avail, want)
        else:
            picked = avail
            print(f"  WARN: {cat} solo tiene {len(avail)} candidatas, "
                  f"queria {want}")
        for ch, s, e in picked:
            chosen.append({
                "window_id": f"{species}_{ch}_{s}",
                "species": species,
                "chrom": ch,
                "start": s,
                "end": e,
                "label": 0,
                "negative_type": cat,
            })

    # Guardar
    out_jsonl = ANN_DIR / f"{species}.negatives.jsonl"
    with out_jsonl.open("w") as f:
        for w in chosen:
            f.write(json.dumps(w) + "\n")

    # Resumen
    cnt = {k: 0 for k in CATEGORIES}
    for w in chosen:
        cnt[w["negative_type"]] += 1

    out_summary = ANN_DIR / f"{species}.negatives.summary.json"
    out_summary.write_text(json.dumps({
        "species": species,
        "n_total": len(chosen),
        "target_total": n_total,
        "by_category": cnt,
        "target_fractions": CATEGORIES,
    }, indent=2))

    print(f"\n  TOTAL muestreado: {len(chosen):,} ventanas negativas")
    for cat, n in cnt.items():
        pct = 100 * n / max(len(chosen), 1)
        tgt = 100 * CATEGORIES[cat]
        print(f"    {cat:12s}: {n:5,}  ({pct:4.1f}%  vs objetivo {tgt:.0f}%)")
    print(f"  salida: {out_jsonl.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("species", nargs="?", default="all",
                    help="especie (arabidopsis/scerevisiae/celegans/...) o 'all'")
    ap.add_argument("--n_total", type=int, default=5000,
                    help="numero total de ventanas negativas por especie")
    args = ap.parse_args()

    if args.species == "all":
        species = sorted(set(GENOME_PATHS) & set(
            p.stem for p in ANN_DIR.glob("*.bed")
            if not p.stem.endswith(".excluded")
        ))
    else:
        species = [args.species]

    for sp in species:
        sample_windows(sp, n_total=args.n_total)


if __name__ == "__main__":
    main()
