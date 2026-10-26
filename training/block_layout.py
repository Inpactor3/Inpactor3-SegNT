"""Layout de bloques por ventana — compartido por entrenamiento, precalculo y evaluacion.

Una ventana de 50 kb aporta `n_blocks` bloques de `chunk_bp` pb:
  * `windowed` (entrenamiento): bloques repartidos 0 .. 50 kb - 6 kb.
    4 bloques -> offsets 0, 14667, 29333, 44000 (24 kb de entrada).
  * `eval_full` (evaluacion A100): bloques contiguos desde 0, con el ultimo
    ajustado para no salirse de la ventana.
    8 bloques -> offsets 0, 6286, 12571, ..., 44000 (48 kb de entrada).

El modelo hace max pooling sobre los embeddings de los bloques, asi que el tamano
de la entrada entrenada (4 bloques) y el de evaluacion (8 bloques) son compatibles.
"""
import numpy as np


def block_offsets(n_blocks, window_size=50000, chunk_bp=6000, mode="windowed"):
    total = n_blocks * chunk_bp
    if total > window_size:
        raise ValueError(f"{n_blocks} bloques x {chunk_bp} pb = {total} pb > ventana "
                         f"{window_size} pb")
    if mode == "eval_full":
        return np.arange(n_blocks, dtype=np.int64) * chunk_bp
    span = window_size - chunk_bp
    return np.array([round(i * span / max(n_blocks - 1, 1)) for i in range(n_blocks)],
                    dtype=np.int64)


def bp_per_window(n_blocks, chunk_bp=6000):
    return n_blocks * chunk_bp
