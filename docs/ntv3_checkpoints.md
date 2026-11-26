# Checkpoints NTv3

Copia aqui los cabezales NTv3 compatibles con `inpactor3.detect` (deben tener
`head`, `hidden`, `mu`, `sd` y `mode`). Un archivo por reino, por ejemplo:

```
models/ntv3/Animales_best.pt
models/ntv3/Plantas_best.pt
models/ntv3/Hongos_best.pt
```

El visor los detecta automaticamente y los marca como `NTv3` en `/analizar`.
No se versionan: `models/**/*.pt` esta ignorado por git.
