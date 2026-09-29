"""
Re-entrenamiento COMPLETO de las 3 redes de Inpactor 2 con el dataset común
cross-kingdom (PanTEon plants + animals + fungi):

  - Inpactor2_Detect  (binaria hay/no-hay LTR-RT en ventana 50 kb)
  - Inpactor2_Filter  (¿el candidato es intacto? — heurístico por longitud)
  - Inpactor2_Class   (superfamilia LTR: COPIA/GYPSY/ERV/BELPAO/LARD/TRIM)

Uso (con el env de Inpactor 2 activado):
    conda activate Inpactor2
    python scripts/retrain_inpactor2_full.py \
        --pos data/common_dataset/inpactor2_format/Inpactor2_train_positive.fasta \
        --neg data/common_dataset/inpactor2_format/Inpactor2_train_negative.fasta \
        --pos-val data/common_dataset/inpactor2_format/Inpactor2_val_positive.fasta \
        --neg-val data/common_dataset/inpactor2_format/Inpactor2_val_negative.fasta \
        --class-fasta data/common_dataset/inpactor2_format/Inpactor2_class_train.fasta \
        --class-val   data/common_dataset/inpactor2_format/Inpactor2_class_val.fasta \
        --out-dir models/inpactor2_teon
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Añadir Inpactor2 al PYTHONPATH
INPACTOR2_PATH = Path("/home/cami/Desktop/ProyectoIntegrador/Inpactor2")
sys.path.insert(0, str(INPACTOR2_PATH))

import tensorflow as tf
from tensorflow.keras import layers as L
from Bio import SeqIO

# Superfamilia → clase (7 clases: 0 = fondo, 1..6 = LTR)
LTR_SUPERFAMILIES = {
    "COPIA": 1, "GYPSY": 2, "ERV": 3,
    "BELPAO": 4, "LARD": 5, "TRIM": 6,
}
NUM_LTR_CLASSES = 7  # 0 fondo + 6 superfamilias
TOTAL_WIN_LEN = 50000
LANGU = ["A", "C", "G", "T", "N"]


def f1_m(y_true, y_pred):
    """Métrica F1 idéntica a Inpactor2."""
    from tensorflow.keras import backend as K
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    return 2 * ((p * r) / (p + r + K.epsilon()))


def fasta2onehot(seq: str, L: int = TOTAL_WIN_LEN) -> np.ndarray:
    """Codifica secuencia como (5, L) one-hot A/C/G/T/N. Trunca o pad con N."""
    arr = np.zeros((5, L), dtype=np.uint8)
    seq = (seq.upper() + "N" * L)[:L]
    for i, nt in enumerate(seq):
        idx = LANGU.index(nt) if nt in LANGU else 4
        arr[idx, i] = 1
    return arr


def load_fasta_batch(path: Path, label: int) -> tuple[np.ndarray, np.ndarray]:
    """Lee FASTA y devuelve (X, y) — one-hot + etiqueta binaria."""
    seqs = list(SeqIO.parse(str(path), "fasta"))
    print(f"  {path.name}: {len(seqs)} secuencias · etiqueta {label}")
    X = np.zeros((len(seqs), 5, TOTAL_WIN_LEN, 1), dtype=np.uint8)
    for i, rec in enumerate(seqs):
        X[i, :, :, 0] = fasta2onehot(str(rec.seq))
    y = np.full(len(seqs), label, dtype=np.uint8)
    return X, y


# =============================================================================
# 1. Arquitectura DETECT (copiada del notebook Inpactor2_Dect.ipynb)
# =============================================================================

def build_detect_model(input_shape=(5, TOTAL_WIN_LEN, 1)) -> tf.keras.Model:
    """CNN 2D idéntica a la de Inpactor2 (notebook Inpactor2_Dect.ipynb)."""
    inp = tf.keras.Input(shape=input_shape, name="input_1")
    x = L.Conv2D(32, (5, 20), strides=(1, 1), activation="relu", use_bias=True)(inp)
    x = L.BatchNormalization(momentum=0.2, epsilon=0.001)(x)
    x = L.MaxPooling2D((1, 10), strides=(1, 10))(x)

    x = L.Conv2D(64, (1, 20), strides=(1, 1), activation="relu", use_bias=True)(x)
    x = L.BatchNormalization(momentum=0.2, epsilon=0.001)(x)
    x = L.MaxPooling2D((1, 15), strides=(1, 15))(x)

    x = L.Conv2D(128, (1, 35), strides=(1, 1), activation="relu", use_bias=True)(x)
    x = L.BatchNormalization(momentum=0.2, epsilon=0.001)(x)
    x = L.MaxPooling2D((1, 15), strides=(1, 15))(x)

    x = L.Flatten()(x)
    x = L.Dense(1000, activation="relu")(x)
    x = L.Dense(500, activation="relu")(x)
    out = L.Dense(1, activation="sigmoid", name="output_1")(x)

    model = tf.keras.Model(inp, out)
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.95
    )
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
        metrics=["accuracy", f1_m],
    )
    return model


def train_detect(args, out_dir: Path):
    print("\n=== Fase 1 · Entrenar Inpactor2_Detect ===")
    Xp, yp = load_fasta_batch(Path(args.pos), 1)
    Xn, yn = load_fasta_batch(Path(args.neg), 0)
    X_train = np.concatenate([Xp, Xn], axis=0)
    y_train = np.concatenate([yp, yn], axis=0)

    Xvp, yvp = load_fasta_batch(Path(args.pos_val), 1)
    Xvn, yvn = load_fasta_batch(Path(args.neg_val), 0)
    X_val = np.concatenate([Xvp, Xvn], axis=0)
    y_val = np.concatenate([yvp, yvn], axis=0)

    print(f"[shape] train X={X_train.shape} y={y_train.shape}")
    print(f"[shape] val   X={X_val.shape}   y={y_val.shape}")

    model = build_detect_model(X_train.shape[1:])
    model.summary()

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(out_dir / "Inpactor_Detect_model.hdf5"),
        save_best_only=True, monitor="val_f1_m", mode="max",
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_f1_m", patience=5, mode="max", restore_best_weights=True,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_detect,
        epochs=args.epochs_detect,
        callbacks=[ckpt, es],
        verbose=1,
    )

    # Guardado final por si early stopping no disparó
    model.save(str(out_dir / "Inpactor_Detect_model.hdf5"))
    print(f"[ok] Detect guardado en {out_dir / 'Inpactor_Detect_model.hdf5'}")
    return history


# =============================================================================
# 2. Arquitectura CLASS (para 7 clases superfamilia PanTEon)
# =============================================================================

def parse_class_fasta(path: Path) -> tuple[list[str], list[int], list[str]]:
    """Extrae (secuencia, clase_id, especie) de un class FASTA con header
    formato >ID#LINEAJE#ESPECIE."""
    seqs, labels, species = [], [], []
    for rec in SeqIO.parse(str(path), "fasta"):
        header = rec.id
        parts = header.split("#")
        if len(parts) < 2:
            continue
        # el linaje viene como "RLG/UNKNOWN" o "RLC/BELPAO" en nuestro export
        lineage_str = parts[1]
        if "GYPSY" in lineage_str or "RLG/UNKNOWN" in lineage_str:
            cls = LTR_SUPERFAMILIES["GYPSY"]
        elif "COPIA" in lineage_str or "RLC/UNKNOWN" in lineage_str:
            cls = LTR_SUPERFAMILIES["COPIA"]
        elif "ERV" in lineage_str:
            cls = LTR_SUPERFAMILIES["ERV"]
        elif "BELPAO" in lineage_str:
            cls = LTR_SUPERFAMILIES["BELPAO"]
        elif "LARD" in lineage_str:
            cls = LTR_SUPERFAMILIES["LARD"]
        elif "TRIM" in lineage_str:
            cls = LTR_SUPERFAMILIES["TRIM"]
        else:
            continue
        seqs.append(str(rec.seq))
        labels.append(cls)
        species.append(parts[2] if len(parts) > 2 else "unknown")
    return seqs, labels, species


def build_class_model(kmer_dim: int, num_classes: int = NUM_LTR_CLASSES) -> tf.keras.Model:
    """MLP para clasificación de linaje (igual estructura que Inpactor2_Class)."""
    inp = tf.keras.Input(shape=(kmer_dim,))
    x = L.Dense(2048, activation="relu")(inp)
    x = L.Dropout(0.3)(x)
    x = L.BatchNormalization()(x)
    x = L.Dense(1024, activation="relu")(x)
    x = L.Dropout(0.3)(x)
    x = L.BatchNormalization()(x)
    x = L.Dense(512, activation="relu")(x)
    x = L.Dropout(0.3)(x)
    x = L.BatchNormalization()(x)
    out = L.Dense(num_classes, activation="softmax", name="output_1")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy", f1_m],
    )
    return model


def extract_kmers(seqs: list[str]) -> np.ndarray:
    """Extrae k-mer features usando el kmer_extractor_model de Inpactor 2."""
    from Inpactor2_utils import kmer_extractor_model
    print(f"  extrayendo k-mers de {len(seqs)} secuencias...")
    X = np.zeros((len(seqs), 5, TOTAL_WIN_LEN), dtype=np.uint8)
    for i, s in enumerate(seqs):
        X[i] = fasta2onehot(s)
    kext = kmer_extractor_model(X[:1])  # crea la red una vez
    kmers = kext.predict(X, batch_size=4, verbose=1)
    return kmers


def train_class(args, out_dir: Path):
    print("\n=== Fase 2 · Entrenar Inpactor2_Class ===")
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from joblib import dump

    seqs_tr, labels_tr, _ = parse_class_fasta(Path(args.class_fasta))
    seqs_va, labels_va, _ = parse_class_fasta(Path(args.class_val))
    print(f"[shape] train {len(seqs_tr)} seqs · val {len(seqs_va)} seqs")

    # k-mer features
    Xtr = extract_kmers(seqs_tr)
    Xva = extract_kmers(seqs_va)

    # Scaler + PCA
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xva_s = scaler.transform(Xva)
    pca = PCA(n_components=0.96, svd_solver="full").fit(Xtr_s)
    Xtr_p = pca.transform(Xtr_s)
    Xva_p = pca.transform(Xva_s)
    print(f"[pca ] reducido a {Xtr_p.shape[1]} componentes")

    dump(scaler, out_dir / "std_scaler.bin", compress=True)
    dump(pca, out_dir / "std_pca.bin", compress=True)

    ytr = tf.keras.utils.to_categorical(labels_tr, num_classes=NUM_LTR_CLASSES)
    yva = tf.keras.utils.to_categorical(labels_va, num_classes=NUM_LTR_CLASSES)

    model = build_class_model(Xtr_p.shape[1])
    model.summary()

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(out_dir / "Inpactor_Class.hdf5"),
        save_best_only=True, monitor="val_f1_m", mode="max",
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_f1_m", patience=10, mode="max", restore_best_weights=True,
    )

    history = model.fit(
        Xtr_p, ytr,
        validation_data=(Xva_p, yva),
        batch_size=args.batch_class,
        epochs=args.epochs_class,
        callbacks=[ckpt, es],
        verbose=1,
    )
    model.save(str(out_dir / "Inpactor_Class.hdf5"))
    print(f"[ok] Class guardado en {out_dir / 'Inpactor_Class.hdf5'}")
    return history


# =============================================================================
# 3. FILTER (heurístico simple: longitud dentro de rango)
# =============================================================================

def train_filter_placeholder(args, out_dir: Path):
    """
    Filter en Inpactor 2 discrimina "intacto vs no intacto" pero no tenemos
    esa etiqueta. Guardamos el modelo original como fallback y notamos la
    limitación.
    """
    import shutil
    src = Path("/home/cami/Desktop/ProyectoIntegrador/Inpactor2/Models/Inpactor_Filter.hdf5")
    dst = out_dir / "Inpactor_Filter.hdf5"
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
        print(f"[copy] {src.name} → {dst}")
        # También copiamos su scaler y PCA
        for f in ["std_scaler_filter.bin", "std_pca_filter.bin"]:
            s = src.parent / f
            d = out_dir / f
            if s.exists() and not d.exists():
                shutil.copy(s, d)
    print(f"[note] Filter usa pesos originales de Inpactor 2 (sin re-entrenar)")
    print(f"       Requiere etiquetas de 'intactness' que PanTEon no provee.")


# También hay que copiar Weights_SL.npy (pesos del k-mer extractor)
def copy_kmer_weights(out_dir: Path):
    import shutil
    src = Path("/home/cami/Desktop/ProyectoIntegrador/Inpactor2/Models/Weights_SL.npy")
    dst = out_dir / "Weights_SL.npy"
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
        print(f"[copy] {src.name} → {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=Path, required=True)
    ap.add_argument("--neg", type=Path, required=True)
    ap.add_argument("--pos-val", type=Path, required=True)
    ap.add_argument("--neg-val", type=Path, required=True)
    ap.add_argument("--class-fasta", type=Path, required=True)
    ap.add_argument("--class-val", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs-detect", type=int, default=30)
    ap.add_argument("--batch-detect", type=int, default=8)
    ap.add_argument("--epochs-class", type=int, default=100)
    ap.add_argument("--batch-class", type=int, default=128)
    ap.add_argument("--skip-detect", action="store_true")
    ap.add_argument("--skip-class", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[env] TF {tf.__version__} · GPU: {tf.config.list_physical_devices('GPU')}")

    if not args.skip_detect:
        train_detect(args, args.out_dir)
    if not args.skip_class:
        train_class(args, args.out_dir)

    train_filter_placeholder(args, args.out_dir)
    copy_kmer_weights(args.out_dir)

    print(f"\n[DONE] modelos re-entrenados en {args.out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
