# Manifiesto de procedencia de datos

Registro exigido por la cláusula séptima del Acuerdo Inpactor3: **origen, licencia y
formato de cada fuente**, con la fecha de incorporación.

| Fuente | Origen / URL | Licencia | Formato | Uso | Fecha |
|---|---|---|---|---|---|
| Genomas del corpus (400) | `/media/.../InpactorRM/genomes_449` (descarga NCBI) | dominio público (NCBI) | FASTA `.fna.gz` | entrenamiento | 2026-10 |
| Anotaciones curadas | `annotations/` y `annotations_dedup/` del proyecto | según InpactorDB/PanTEon | BED | etiquetado | 2026-10 |
| Librerías de consensos | `data/panteon_libraries/` | **no pública / confidencial** | FASTA | anotación (no se publica) | 2026-10 |
| TAIR10 + BED oro | Kaggle `inpactor2-gold` / TAIR | uso abierto | FASTA + BED | test oro | 2026-10 |
| dm6 + `rmsk` | UCSC | uso abierto | FASTA + TXT | test oro | 2026-11 |
| S. pombe + Tf2 | Ensembl EBI + PomBase | uso abierto | FASTA + GFF3 → BED | test de plata | 2026-11 |
| Dfam FamDB | dfam.org | CC0 / uso libre | HDF5 `.h5` | RepeatMasker | 2026-11 |
| NT v2 100M | HuggingFace `InstaDeepAI/...` | CC-BY-NC-SA 4.0 | safetensors | lector | 2026-10 |
| NTv3 100M | HuggingFace `InstaDeepAI/NTv3_100M_pre` | no comercial, acceso restringido | safetensors | lector alternativo | 2026-11 |

## Reglas aplicadas

1. **Nada de librerías no públicas en el repositorio.** `data/panteon_libraries/` y
   `annotations/` están en `.gitignore`; se conservan solo en local.
2. **Nada de pesos en el repositorio.** `*.pt`, `*.npz` y `*.safetensors` están excluidos.
3. **Trazabilidad:** cada dataset publicado en Kaggle corresponde a una fila de esta tabla.
4. **Datos personales:** el proyecto no usa datos genómicos de individuos; solo genomas de
   referencia públicos.
5. **Atribución:** las fuentes se citan en `NOTICE.md` y en cada publicación derivada.

## Artefactos derivados publicados

| Dataset de Kaggle | Contenido | ¿Autorizado? |
|---|---|---|
| `inpactor2-mil-code` | scripts | sí (código propio) |
| `inpactor2-hongos-ventanas`, `-plantas-ventanas` | bundles de secuencias | sí |
| `inpactor2-animales-24k`, `-100k` | bundles de secuencias | sí |
| `inpactor2-gold-ventanas`, `-plata-ventanas` | genomas oro y cabezales | sí, con autorización de los directores |
| `inpactor2-ventanas-resultados` | métricas de experimentos | sí, con autorización |
| `inpactor2-paquete-descargas` | código, guía y registro de errores | sí |
