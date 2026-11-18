"""Pruebas de humo sin GPU ni modelos descargados."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inpactor3.detect import (CHUNK, contar_ventanas, limpiar, offsets,
                              ventanas)  # noqa: E402
from training.block_layout import block_offsets  # noqa: E402


class TestOffsets(unittest.TestCase):
    def test_coincide_con_entrenamiento(self):
        for chunks in (2, 4, 8, 16):
            for window in (24000, 50000, 100000):
                if chunks * CHUNK > window:
                    continue
                self.assertEqual(offsets(chunks, window),
                                 [int(x) for x in block_offsets(chunks, window, CHUNK)])


class TestSecuencia(unittest.TestCase):
    def test_limpiar_convierte_iupac(self):
        self.assertEqual(limpiar("acgtRYSW"), "ACGTNNNN")

    def test_limpiar_conserva_n(self):
        self.assertEqual(limpiar("ACGTN"), "ACGTN")


class TestVentanas(unittest.TestCase):
    def setUp(self):
        self.genoma = {"chr1": "A" * 120000}
        self.offs = offsets(4, 50000)

    def test_conteo(self):
        self.assertEqual(contar_ventanas(self.genoma, 50000), 2)

    def test_generador_no_materializa(self):
        it = ventanas(self.genoma, 50000, self.offs)
        self.assertFalse(isinstance(it, list))
        primera = next(it)
        self.assertEqual(primera[0], "chr1")
        self.assertEqual(len(primera[3]), 4)
        self.assertEqual(len(primera[3][0]), CHUNK)

    def test_chroms_inexistente(self):
        self.assertEqual(contar_ventanas(self.genoma, 50000, {"nope"}), 0)


if __name__ == "__main__":
    unittest.main()
