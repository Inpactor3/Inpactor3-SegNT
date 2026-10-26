"""Clasifica las especies de un CSV por grupo animal usando la API de GBIF.

Uso:
  python taxonomia_especies.py --csv docs/animales_especies.csv \
      --out docs/animales_taxonomia.csv
"""
import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request

ORDENES_PECES = {
    "Perciformes", "Cypriniformes", "Salmoniformes", "Acipenseriformes",
    "Cyprinodontiformes", "Siluriformes", "Gadiformes", "Characiformes",
    "Clupeiformes", "Pleuronectiformes", "Scorpaeniformes", "Tetraodontiformes",
    "Anguilliformes", "Esociformes", "Osmeriformes", "Atheriniformes",
    "Beloniformes", "Mugiliformes", "Labriformes", "Spariformes", "Gobiiformes",
    "Cichliformes", "Osteoglossiformes", "Polypteriformes", "Lepisosteiformes",
    "Myxiniformes", "Petromyzontiformes", "Carcharhiniformes", "Lamniformes",
    "Rajiformes", "Myliobatiformes", "Squaliformes",
}
ORDENES_REPTILES = {"Testudines", "Squamata", "Crocodilia", "Rhynchocephalia"}

GRUPOS = {
    ("Cnidaria", "Anthozoa"): "Corales y anémonas",
    ("Cnidaria", None): "Medusas y cnidarios",
    ("Arthropoda", "Insecta"): "Insectos",
    ("Arthropoda", "Arachnida"): "Arácnidos",
    ("Arthropoda", "Malacostraca"): "Crustáceos",
    ("Arthropoda", "Branchiopoda"): "Crustáceos",
    ("Arthropoda", "Collembola"): "Otros artrópodos",
    ("Arthropoda", "Maxillopoda"): "Crustáceos",
    ("Arthropoda", None): "Otros artrópodos",
    ("Chordata", "Mammalia"): "Mamíferos",
    ("Chordata", "Aves"): "Aves",
    ("Chordata", "Reptilia"): "Reptiles",
    ("Chordata", "Testudines"): "Reptiles",
    ("Chordata", "Myxini"): "Peces",
    ("Chordata", "Actinopteri"): "Peces",
    ("Chordata", "Cephalaspidomorphi"): "Peces",
    ("Chordata", "Amphibia"): "Anfibios",
    ("Chordata", "Actinopterygii"): "Peces",
    ("Chordata", "Chondrichthyes"): "Peces",
    ("Chordata", "Ascidiacea"): "Cordados no vertebrados",
    ("Chordata", "Leptocardii"): "Cordados no vertebrados",
    ("Chordata", None): "Otros cordados",
    ("Mollusca", None): "Moluscos",
    ("Echinodermata", None): "Equinodermos",
    ("Annelida", None): "Anélidos",
    ("Nematoda", None): "Nematodos",
    ("Platyhelminthes", None): "Gusanos planos",
    ("Porifera", None): "Esponjas",
    ("Bryozoa", None): "Briozoos",
    ("Brachiopoda", None): "Braquiópodos",
    ("Hemichordata", None): "Hemicordados",
    ("Ctenophora", None): "Ctenóforos",
}


def grupo(phylum, clase, orden):
    if (phylum, clase) in GRUPOS:
        return GRUPOS[(phylum, clase)]
    if phylum == "Chordata" and orden in ORDENES_REPTILES:
        return "Reptiles"
    if phylum == "Chordata" and orden in ORDENES_PECES:
        return "Peces"
    if (phylum, None) in GRUPOS:
        return GRUPOS[(phylum, None)]
    return f"Otros ({phylum or '?'})"


def nombre_cientifico(tag):
    limpio = re.sub(r"_+GC[AF]_.*$", "", tag)
    return limpio.replace("_", " ").strip()


def gbif(nombre):
    url = "https://api.gbif.org/v1/species/match?name=" + urllib.parse.quote(nombre)
    with urllib.request.urlopen(url, timeout=25) as r:
        d = json.load(r)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    filas = list(csv.DictReader(open(args.csv)))
    cache = {}
    for f in filas:
        tag = f["species"]
        nombre = nombre_cientifico(tag)
        if nombre not in cache:
            try:
                d = gbif(nombre)
            except Exception as e:
                d = {"matchType": "ERROR", "confidence": 0, "_err": str(e)[:60]}
            cache[nombre] = d
            time.sleep(0.15)
        d = cache[nombre]
        f["nombre_cientifico"] = nombre
        f["phylum"] = d.get("phylum", "")
        f["clase"] = d.get("class", "")
        f["orden"] = d.get("order", "")
        f["familia"] = d.get("family", "")
        f["confianza_gbif"] = d.get("confidence", 0)
        f["grupo"] = (grupo(f["phylum"], f["clase"], f["orden"])
                      if d.get("phylum") else "Sin clasificar")

    campos = list(filas[0].keys())
    with open(args.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)

    # resumen por grupo
    grupos = {}
    for f in filas:
        g = f["grupo"]
        e = grupos.setdefault(g, {"sp": 0, "win": 0, "pos": 0, "neg": 0, "ejemplos": []})
        e["sp"] += 1
        e["win"] += int(f["n_windows"])
        e["pos"] += int(f["n_pos"])
        e["neg"] += int(f["n_neg"])
        if len(e["ejemplos"]) < 3:
            e["ejemplos"].append(f["nombre_cientifico"])

    print(f"{'grupo':28s} {'especies':>8s} {'ventanas':>9s} {'pos':>8s} {'neg':>8s}")
    for g, e in sorted(grupos.items(), key=lambda kv: -kv[1]["win"]):
        print(f"{g:28s} {e['sp']:8d} {e['win']:9d} {e['pos']:8d} {e['neg']:8d}")
    print(f"\nCSV con taxonomía: {args.out}")


if __name__ == "__main__":
    main()
