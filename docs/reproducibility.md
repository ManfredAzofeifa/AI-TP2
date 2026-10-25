# Reproducir la auditoría en otra computadora

La configuración del equipo se conserva: MPNet de 768 características,
regresión con la cuadrícula original, PCA óptimo de 384 componentes y
detector con D=2, 30 intervalos, cuantil 5 % y reserva de ham del 20 %.
Las alternativas siguen documentadas y no se adoptan en esta revisión.

## Qué incluye el commit

- Notebook canónico con las definiciones y la auditoría opcional desactivada.
  Sus salidas se limpian para el desarrollo; las cifras quedan en el análisis.
- Pruebas, CSV de las 30 corridas por alternativa, figuras revisadas,
  fuentes LaTeX y los dos PDF compilados. El informe de `submission/` es
  preliminar: falta el punto 3 y no constituye la entrega final.
- `scripts/run_puntos_1_2.py`: ejecutor de las celdas existentes, sin copias
  de modelos ni métricas en archivos Python. Sustituye el ejecutor temporal
  que estaba únicamente en `build/audit/` de la máquina de la auditoría.
- `requirements.txt`: dependencias comunes. `requirements-audit.txt` agrega
  las versiones directas observadas en la auditoría con Python 3.14.9 en
  macOS arm64; no es un bloqueo completo de dependencias transitivas ni una
  validación de esas versiones en todas las plataformas.

Para leer los resultados y PDF no hace falta descargar MPNet ni ejecutar
experimentos. El dataset original de UCI ya se encuentra versionado.

## Crear el entorno

Se recomienda Python 3.14 para coincidir con las versiones validadas del
proyecto. Cada persona crea su propio entorno virtual; no se copia `.venv/`.

macOS y Linux, desde la raíz del repositorio:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-audit.txt
.venv/bin/python -m pytest
.venv/bin/python scripts/run_puntos_1_2.py --check
```

Windows PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-audit.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts/run_puntos_1_2.py --check
```

Para usar las dependencias comunes sin fijar las herramientas de la auditoría,
sustituir `requirements-audit.txt` por `requirements.txt`. No se agregan
dependencias del punto 3 ni se eligen sus modelos.

## Caché local y ejecución

`--check` comprueba identidad de textos, codificador, revisión, forma, tipo y
finitud de la caché. No ejecuta experimentos, no descarga modelos ni borra
resultados. Devuelve código 1 si falta una caché válida; eso es normal en un
clon nuevo. La caché validada de esta auditoría ocupa aproximadamente 17 MB,
pero es regenerable y no se incluye en Git, al igual que los pesos de MPNet.

Con caché válida, para ejecutar solo puntos 1 y 2 y sus pruebas:

```bash
.venv/bin/python scripts/run_puntos_1_2.py
```

Si falta la caché, este comando se detiene antes de limpiar `results/`.
Para autorizar la descarga del codificador fijado en el notebook y regenerar
las características, añadir explícitamente `--allow-download`:

```bash
.venv/bin/python scripts/run_puntos_1_2.py --allow-download
```

La primera extracción puede tardar más de 10 minutos y descargar unos
420 MB. El acceso a Hugging Face queda desactivado por el ejecutor cuando
no se proporciona ese permiso. No se cambian el modelo ni su revisión.

Para repetir también las 30 corridas por alternativa de la auditoría:

```bash
.venv/bin/python scripts/run_puntos_1_2.py --audit
```

Se pueden combinar `--audit --allow-download` si se autorizan ambas cosas.
En Windows, usar `.\.venv\Scripts\python.exe` en los mismos comandos.
Invocar un comando de ejecución implica aceptar las corridas largas que
describe; `--check` permite revisar la caché antes de decidir.

El script resuelve la raíz desde su propia ubicación, crea un kernel nuevo
del mismo intérprete que lo invoca y excluye el punto 3 y la comparación final
pendiente. Guarda el notebook ejecutado en
`build/notebook-run/puntos-1-2-ejecutado.ipynb`, también si una celda falla.
El notebook canónico permanece sin modificaciones. No requiere PyCharm ni
un kernel registrado previamente con rutas de otra computadora.

La configuración del notebook limpia `results/figures/` y `results/tables/`.
El ejecutor hace antes un respaldo local en `build/results-before-*/`.
Una ejecución sin `--audit` no regenera los tres CSV `auditoria-*.csv`; para
reproducir también esas tablas, incluir esa opción. Revisar el diff de los
resultados antes de volver a versionarlos. Pueden aparecer pequeñas
diferencias numéricas entre plataformas.

## Compilar los PDF

Se necesita una distribución LaTeX con `latexmk` y BibTeX. En Windows,
seguir la configuración de MiKTeX y Git Bash descrita en `README.md`.
Desde una terminal compatible con `sh`, ejecutar:

```bash
./build_pdf.sh review
./build_pdf.sh
```

Los comandos usan las figuras ya versionadas; no requieren ejecutar MPNet.
Los PDF quedan en `docs/report/review.pdf` y
`submission/tp2_detector_spam.pdf`. Se conserva el carácter preliminar del
segundo mientras falte el punto 3.

## Archivos que cada persona regenera

`.venv/`, `.idea/`, `data/processed/`, la caché de Hugging Face,
`build/` y los intermedios de LaTeX permanecen fuera del commit. No se
necesitan credenciales para ejecutar las pruebas ni para leer los resultados.
No copiar módulos antiguos de `src/`: la implementación vigente está en el
notebook y sus pruebas cargan esas mismas definiciones.
