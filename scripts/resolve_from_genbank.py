"""Para las especies marcadas 'not_found' en RefSeq, intenta encontrarlas
en el assembly_summary_genbank.txt.

Actualiza docs/species_download_manifest.tsv con las nuevas URLs.
"""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GB_INDEX = ROOT / "data" / "ncbi_index" / "assembly_summary_genbank.txt"
MANIFEST = ROOT / "docs" / "species_download_manifest.tsv"

LEVEL_RANK = {
    "Complete Genome": 0,
    "Chromosome":      1,
    "Scaffold":        2,
    "Contig":          3,
}

REFSEQ_RANK = {
    "reference genome": 0,
    "representative genome": 1,
    "na": 2,
}


def build_genbank_index():
    """Lee assembly_summary_genbank.txt y arma dict especie -> mejor ensamblado."""
    print(f"Leyendo GenBank index: {GB_INDEX}")
    best = {}
    with GB_INDEX.open() as f:
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
            sp_norm = " ".join(sp.split()[:2])
            level = LEVEL_RANK.get(row.get("assembly_level", ""), 99)
            cat = REFSEQ_RANK.get(row.get("refseq_category", "").lower(), 2)
            rank = (level, cat)
            if sp_norm not in best or rank < best[sp_norm]["_rank"]:
                best[sp_norm] = {
                    "_rank": rank,
                    "accession": row.get("assembly_accession", ""),
                    "level": row.get("assembly_level", ""),
                    "category": row.get("refseq_category", ""),
                    "ftp_path": row.get("ftp_path", ""),
                    "genome_size": row.get("genome_size", ""),
                }
    print(f"GenBank index listo: {len(best):,} especies")
    return best


def main():
    # Leer manifest actual (strip \r por si acaso)
    rows = []
    with MANIFEST.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            clean = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            rows.append(clean)

    print(f"Manifest: {len(rows)} especies totales")
    not_found = [r for r in rows if r["status"] == "not_found"]
    print(f"'not_found' a buscar en GenBank: {len(not_found)}")

    gb_index = build_genbank_index()

    newly_found = 0
    for row in rows:
        if row["status"] != "not_found":
            continue
        name = row["especie"].strip()
        norm = " ".join(name.split()[:2])
        hit = gb_index.get(norm)
        if hit:
            ftp = hit["ftp_path"].rstrip("/")
            base = ftp.split("/")[-1]
            url_fna = f"{ftp}/{base}_genomic.fna.gz"
            try:
                size_mb = int(hit["genome_size"]) / 1e6
            except (ValueError, TypeError):
                size_mb = 0
            row["status"] = "found"
            row["assembly_accession"] = hit["accession"]
            row["assembly_level"] = hit["level"]
            row["refseq_category"] = hit["category"] or "genbank"
            row["genome_size_mb"] = f"{size_mb:.1f}"
            row["url_fna"] = url_fna
            newly_found += 1

    print(f"\nEncontradas nuevas en GenBank: {newly_found}")

    # Reescribir manifest
    with MANIFEST.open("w", newline="") as f:
        fields = ["split", "reino", "especie", "status", "assembly_accession",
                  "assembly_level", "refseq_category", "genome_size_mb", "url_fna"]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    # Resumen final
    print("\n=== Resumen final por reino y status ===")
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
        print(f"  {reino:10s} / {split:6s}: {v['found']:3d}/{total:3d}  "
              f"(~{gb:5.1f} GB total)")


if __name__ == "__main__":
    main()
