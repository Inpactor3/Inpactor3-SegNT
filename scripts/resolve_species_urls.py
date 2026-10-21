"""Resuelve las URLs de descarga de NCBI RefSeq para las 449 especies
muestreadas, usando el assembly_summary_refseq.txt.

Prioridad: RefSeq complete > representativo > latest. Nivel: chromosome > scaffold.

Salida: docs/species_download_manifest.tsv
    columnas: split, reino, especie, status, assembly_accession, size_mb, url_fna
"""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "data" / "ncbi_index" / "assembly_summary_refseq.txt"
SPECIES_CSV = ROOT / "docs" / "species_split_proposal.csv"
OUT = ROOT / "docs" / "species_download_manifest.tsv"


def load_species_list():
    specs = []
    with SPECIES_CSV.open() as f:
        for row in csv.DictReader(f):
            specs.append({
                "split": row["split"],
                "reino": row["reino"],
                "especie": row["especie"],
            })
    return specs


# Mapear niveles de assembly a prioridad numerica
LEVEL_RANK = {
    "Complete Genome": 0,
    "Chromosome":      1,
    "Scaffold":        2,
    "Contig":          3,
}

# Mapear refseq_category a prioridad
REFSEQ_RANK = {
    "reference genome": 0,
    "representative genome": 1,
    "na": 2,
}


def build_species_index():
    """Lee el assembly_summary y arma dict especie -> mejor ensamblado."""
    print(f"Leyendo {INDEX} ...")
    best_by_species = {}  # species_name -> dict con la mejor fila

    with INDEX.open() as f:
        header = None
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                header = line.lstrip("#").strip().split("\t")
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 20:
                continue
            row = dict(zip(header, parts))
            sp = row.get("organism_name", "").strip()
            if not sp:
                continue
            # Tomar solo "Genus species" (dos palabras)
            sp_norm = " ".join(sp.split()[:2])
            level = LEVEL_RANK.get(row.get("assembly_level", ""), 99)
            cat = REFSEQ_RANK.get(row.get("refseq_category", "").lower(), 2)
            # Preferir Complete Genome y refseq reference
            rank = (level, cat)
            if sp_norm not in best_by_species or rank < best_by_species[sp_norm]["_rank"]:
                best_by_species[sp_norm] = {
                    "_rank": rank,
                    "accession": row.get("assembly_accession", ""),
                    "level": row.get("assembly_level", ""),
                    "category": row.get("refseq_category", ""),
                    "ftp_path": row.get("ftp_path", ""),
                    "organism": sp,
                    "taxid": row.get("taxid", ""),
                    "genome_size": row.get("genome_size", ""),
                }
    print(f"Index listo: {len(best_by_species):,} especies con ensamblado")
    return best_by_species


def resolve(species_list, index):
    out_rows = []
    found = 0
    missing = 0

    for sp in species_list:
        name = sp["especie"].strip()
        norm = " ".join(name.split()[:2])
        hit = index.get(norm) or index.get(name)
        if hit:
            ftp = hit["ftp_path"].rstrip("/")  # evitar // duplicado
            # URL del genoma FASTA
            base = ftp.split("/")[-1]
            url_fna = f"{ftp}/{base}_genomic.fna.gz"
            try:
                size_mb = int(hit["genome_size"]) / 1e6
            except (ValueError, TypeError):
                size_mb = 0
            out_rows.append({
                "split": sp["split"],
                "reino": sp["reino"],
                "especie": name,
                "status": "found",
                "assembly_accession": hit["accession"],
                "assembly_level": hit["level"],
                "refseq_category": hit["category"],
                "genome_size_mb": f"{size_mb:.1f}",
                "url_fna": url_fna,
            })
            found += 1
        else:
            out_rows.append({
                "split": sp["split"],
                "reino": sp["reino"],
                "especie": name,
                "status": "not_found",
                "assembly_accession": "",
                "assembly_level": "",
                "refseq_category": "",
                "genome_size_mb": "",
                "url_fna": "",
            })
            missing += 1

    print(f"\nEncontradas: {found} / {len(species_list)}")
    print(f"No encontradas: {missing}")
    return out_rows


def main():
    species = load_species_list()
    print(f"Especies a resolver: {len(species)}")

    index = build_species_index()
    rows = resolve(species, index)

    # Guardar TSV
    with OUT.open("w", newline="") as f:
        fields = ["split", "reino", "especie", "status", "assembly_accession",
                  "assembly_level", "refseq_category", "genome_size_mb", "url_fna"]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"\n[tsv] {OUT}")

    # Resumen por reino + split
    print("\n=== Resumen por reino y status ===")
    counts = defaultdict(lambda: {"found": 0, "not_found": 0, "size_total": 0.0})
    for r in rows:
        key = (r["reino"], r["split"])
        counts[key][r["status"]] += 1
        try:
            counts[key]["size_total"] += float(r["genome_size_mb"] or 0)
        except ValueError:
            pass

    for (reino, split), v in sorted(counts.items()):
        total = v["found"] + v["not_found"]
        gb = v["size_total"] / 1024
        print(f"  {reino:10s} / {split:6s}: {v['found']:3d}/{total:3d} encontradas  "
              f"(total ~{gb:5.1f} GB)")


if __name__ == "__main__":
    main()
