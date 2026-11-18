"""Arregla Animales v2: reorganiza al split nuevo y recalcula solo las 14 corruptas."""
import modal

app = modal.App("inpactor-fix-animales")
SHA = "30f4947"
GH = f"https://raw.githubusercontent.com/Inpactor3/Inpactor3-SegNT/{SHA}"
NT = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"

MAP = {"Cariama_cristata__GCF_000690535.1": "test", "Acropora_acuminata___GCA_014633975.1": "test", "Arctopsyche_grandis___GCF_051622035.1": "test", "Aricia_agestis___GCF_905147365.1": "test", "Chrysoperla_carnea__GCF_905475395.1": "test", "Ellobius_talpinus__GCA_001685095.1": "test", "Gigantopelta_aegis__GCF_016097555.1": "test", "Acanthopagrus_latus__GCF_904848185.1": "test", "Gopherus_agassizii__GCA_002896415.1": "test", "Dermatophagoides_farinae__GCF_024713945.1": "train", "Catharus_ustulatus__GCF_009819885.2": "train", "Eubucco_bourcierii__GCA_013396675.1": "train", "Eulacestoma_nigropectus__GCA_013397775.1": "train", "Halcyon_senegalensis__GCA_013397595.1": "train", "Haliaeetus_albicilla__GCF_947461875.1": "train", "Cyphastrea_salae__GCA_964194085.1": "train", "Porites_divaricata__GCA_964035745.1": "train", "Branchiostoma_lanceolatum___GCF_035083965.1": "train", "Halocynthia_aurantium__GCA_013436065.1": "train", "Branchinecta_lindahli__GCA_023053555.1": "train", "Bolinopsis_microptera__GCF_026151205.1": "train", "Acanthaster_planci___GCF_054643075.1": "train", "Astropecten_irregularis__GCA_963971285.1": "train", "Acrocera_orbiculus__GCA_947359355.1": "train", "Adranes_taylori__GCA_009867245.2": "train", "Aethina_tumida__GCF_024364675.1": "train", "Anopheles_moucheti__GCF_943734755.1": "train", "Athalia_rosae___GCF_917208135.1": "train", "Drosophila_buzzatii__GCA_035045845.1": "train", "Drosophila_kikkawai__GCF_030179895.1": "train", "Drosophila_willistoni___GCF_018902025.1": "train", "Nasonia_giraulti___GCA_016647725.1": "train", "Xenos_peckii___GCA_040167675.1": "train", "Balaenoptera_musculus__GCF_009873245.2": "train", "Bison_bison__GCF_000754665.1": "train", "Ceratotherium_simum__GCF_057802395.1": "train", "Eonycteris_spelaea__GCA_062009626.1": "train", "Calvadosia_cruxmelitensis__GCA_900245855.1": "train", "Pelagia_noctiluca___GCA_963924455.1": "train", "Dreissena_rostriformis__GCA_055670145.1": "train", "Haliotis_cracherodii__GCF_022045235.1": "train", "Caenorhabditis_remanei___GCF_010183535.1": "train", "Desoria_tigrina__GCA_906901685.1": "train", "Acipenser_ruthenus__GCF_902713425.1": "train", "Cirrhinus_mrigala__GCA_036247105.2": "train", "Coregonus_clupeaformis__GCF_020615455.1": "train", "Cyprinodon_variegatus__GCF_000732505.1": "train", "Eptatretus_burgeri__GCF_056783735.1": "train", "Actinemys_marmorata__GCA_022086475.1": "train", "Chelonoidis_abingdonii__GCF_003597395.2": "train", "Charadrius_alexandrinus__GCA_053477275.1": "val", "Acropora_hyacinthus__GCA_964291705.1": "val", "Blera_fallax___GCA_946965025.1": "val", "Gnatocerus_cornutus___GCF_029298725.1": "val", "Onthophagus_taurus___GCF_036711975.1": "val", "Capromys_pilorides__GCA_004027915.1": "val", "Amphilophus_citrinellus__GCA_013435755.1": "val", "Emydura_subglobosa__GCA_054346155.1": "val"}
CORRUPT = {"Acropora_acuminata___GCA_014633975.1": "test", "Arctopsyche_grandis___GCF_051622035.1": "test", "Aricia_agestis___GCF_905147365.1": "test", "Branchiostoma_lanceolatum___GCF_035083965.1": "train", "Acanthaster_planci___GCF_054643075.1": "train", "Athalia_rosae___GCF_917208135.1": "train", "Drosophila_willistoni___GCF_018902025.1": "train", "Nasonia_giraulti___GCA_016647725.1": "train", "Xenos_peckii___GCA_040167675.1": "train", "Pelagia_noctiluca___GCA_963924455.1": "train", "Caenorhabditis_remanei___GCF_010183535.1": "train", "Blera_fallax___GCA_946965025.1": "val", "Gnatocerus_cornutus___GCF_029298725.1": "val", "Onthophagus_taurus___GCF_036711975.1": "val"}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .pip_install("torch", "transformers==4.46.3", "numpy", "kaggle", "six")
    .run_commands(
        "mkdir -p /code",
        f"curl -sL {GH}/training/precompute_embeddings.py -o /code/precompute_embeddings.py",
        f"curl -sL {GH}/training/subsample_train.py -o /code/subsample_train.py",
        f"curl -sL {GH}/training/train_mlp.py -o /code/train_mlp.py",
        f"curl -sL {GH}/training/block_layout.py -o /code/block_layout.py",
        f"curl -sL {GH}/colab_bridge/src/model.py -o /code/model.py",
        f"curl -sL {GH}/colab_bridge/src/dataset.py -o /code/dataset.py",
    )
)
vol = modal.Volume.from_name("inpactor-emb", create_if_missing=True)
SUFIJOS = ("_emb.npy", "_labels.npy", "_pool.npy", "_meta.jsonl", "_offsets.npy")


@app.function(image=image, volumes={"/emb": vol}, timeout=3600)
def reorganizar():
    import os
    import shutil
    base, dst = "/emb/Animales", "/emb/AnimalesNuevo"
    movidos = borrados = 0
    for split in ("train", "val", "test"):
        d = f"{base}/{split}"
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            sp = next((f[:-len(s)] for s in SUFIJOS if f.endswith(s)), None)
            if sp is None:
                continue
            if sp in CORRUPT:
                os.remove(f"{d}/{f}")
                borrados += 1
                continue
            nuevo = MAP.get(sp)
            if nuevo is None:
                continue
            od = f"{dst}/{nuevo}"
            os.makedirs(od, exist_ok=True)
            shutil.move(f"{d}/{f}", f"{od}/{f}")
            movidos += 1
    vol.commit()
    return f"movidos {movidos} archivos, borrados {borrados}"


@app.function(image=image, gpu="T4", volumes={"/emb": vol}, timeout=3600,
              secrets=[modal.Secret.from_name("kaggle-token")], retries=2)
def recomputar(sp: str, split: str):
    import glob
    import os
    import shutil
    import subprocess
    import sys

    tok = os.environ.get("KAGGLE_API_TOKEN", "")
    if tok:
        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        p = os.path.expanduser("~/.kaggle/access_token")
        open(p, "w").write(tok.strip())
        os.chmod(p, 0o600)

    tmp = "/tmp/t"
    shutil.rmtree(tmp, ignore_errors=True)
    seq = f"{tmp}/seq/{split}"
    os.makedirs(seq, exist_ok=True)
    r = subprocess.run(["kaggle", "datasets", "download",
                        "-d", "cristianosoriog/inpactor2-animales-fix",
                        "-f", f"{sp}.npz", "-p", f"{tmp}/dl", "--quiet"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return f"FALLO descarga {sp}: {r.stderr.strip()[-300:]}"
    hall = glob.glob(f"{tmp}/dl/**/*.npz", recursive=True)
    if not hall:
        return f"FALLO sin npz {sp}"
    shutil.copy2(hall[0], os.path.join(seq, f"{sp}.npz"))

    entrada = f"{tmp}/seq"
    if split == "train":
        sub = f"{tmp}/sub"
        r = subprocess.run([sys.executable, "/code/subsample_train.py", "--seq_dir",
                            entrada, "--out_dir", sub, "--ratio", "1"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return f"FALLO submuestreo {sp}: {r.stderr.strip()[-300:]}"
        entrada = sub

    r = subprocess.run([sys.executable, "/code/precompute_embeddings.py",
                        "--seq_dir", entrada, "--out_dir", "/emb/AnimalesNuevo",
                        "--splits", split, "--trust_remote_code", "--nt_model", NT,
                        "--max_tokens", "1024", "--batch_size", "8",
                        "--token_batch", "32"], capture_output=True, text=True)
    if r.returncode != 0:
        return f"FALLO precalculo {sp}: {r.stderr.strip()[-300:]}"
    vol.commit()
    import numpy as np
    n = int(np.load(f"/emb/AnimalesNuevo/{split}/{sp}_labels.npy").shape[0])
    return f"OK {split}/{sp}: {n} ventanas"


@app.function(image=image, gpu="A10G", volumes={"/emb": vol}, timeout=3600)
def entrenar():
    import json
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "/code/train_mlp.py",
                        "--emb_dir", "/emb/AnimalesNuevo",
                        "--out_dir", "/emb/Modelos/Animales_v2fix",
                        "--epochs", "100", "--patience", "10", "--batch_size", "4096",
                        "--balance", "global", "--class_weight", "--calibrate",
                        "--select", "balanced_accuracy"], capture_output=True, text=True)
    print(r.stdout[-5000:], flush=True)
    if r.returncode != 0:
        print("STDERR", r.stderr[-2000:], flush=True)
        return "FALLO entrenamiento"
    vol.commit()
    return json.dumps(json.load(open("/emb/Modelos/Animales_v2fix/test_metrics.json")), indent=2)


@app.local_entrypoint()
def main():
    print(reorganizar.remote())
    for s in ("train", "val", "test"):
        print(s, len([x for x in CORRUPT.values() if x == s]))
    print(recomputar.map(list(CORRUPT.keys()), list(CORRUPT.values())))
    print(entrenar.remote())
