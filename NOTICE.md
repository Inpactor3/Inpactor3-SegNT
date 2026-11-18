# Créditos, licencias y confidencialidad

Proyecto: **Inpactor3-SegNT** — detección de retrotransposones LTR con lector congelado.
Universidad de Caldas · CI²DT² · Proyecto Integrador II, 2026.

## Autores del software

- Jhonier Serna Cardona
- Alejandro Benítez Molina
- Cristian Camilo Osorio Granada

Docente orientador: **Cesar Augusto Franco** (Universidad de Caldas).
Director científico externo: **Miguel Andrés Meñaca**.

## Software de origen

Este trabajo continúa la familia **Inpactor** (2018) e **Inpactor2** (2023), obras de sus
autores publicadas bajo **GNU GPLv3**. Todo código derivado conserva esa licencia y la
atribución original, conforme a los términos del Acuerdo Inpactor3 y de la propia GPLv3.

## Componentes de terceros

| Componente | Titular | Licencia | Uso |
|---|---|---|---|
| Nucleotide Transformer v2 (100M multi-especie) | InstaDeep, NVIDIA, TUM | CC-BY-NC-SA 4.0 | lector congelado |
| Nucleotide Transformer v3 (100M) | InstaDeep, NVIDIA, TUM | licencia no comercial, acceso restringido | lector alternativo evaluado |
| Dfam | Dfam Consortium | CC0 / uso libre | librerías de repeticiones |
| InpactorDB / PanTEon | autores citados | según su publicación | anotaciones de referencia |
| Genomas de referencia | NCBI, UCSC, Ensembl | dominio público / uso abierto | datos de entrada |

Los pesos de los lectores **no se redistribuyen**: se descargan desde HuggingFace con la
cuenta y la aceptación de licencia de cada usuario.

## Datos que no se publican

Conforme a la cláusula primera del Acuerdo Inpactor3, se consideran confidenciales:

- las librerías curadas aportadas por laboratorios colaboradores;
- los pesos, hiperparámetros, semillas, particiones y métricas no publicadas;
- el código en desarrollo previo a la autorización de publicación;
- cualquier detalle de la infraestructura de supercomputación.

El repositorio **no contiene** librerías de terceros, pesos ni archivos de gran tamaño: se
excluyen con `.gitignore` y con el hook de pre-commit incluido en `.githooks/`.

## Cómo citar

> Serna, J.; Benítez, A.; Osorio, C. (2026). *Inpactor3-SegNT: detección de
> retrotransposones LTR con un lector de ADN congelado*. Proyecto Integrador II,
> Universidad de Caldas.

Y los trabajos originales de Inpactor e Inpactor2, más la referencia del lector
(Dalla-Torre et al., *Nucleotide Transformer*) y, para NTv3, el preprint de bioRxiv de
diciembre de 2025.
