# Guía de colaboración

Las tareas, las reglas del proyecto (organización del notebook, contratos y
protocolo experimental) y la checklist de entrega están en [TODO.md](TODO.md);
los agentes de IA siguen [AGENTS.md](AGENTS.md). Esta guía cubre el flujo de
trabajo con Git y el notebook.

## Flujo de Git

- `main` debe representar el estado integrado y revisado.
- Crear ramas cortas con prefijos como `feat/`, `docs/`, `test/` o `chore/`,
  indicando el punto (por ejemplo, `feat/punto1-pca`).
- Evitar ramas permanentes por estudiante.
- Mantener cada pull request enfocado en una sola tarea.
- Solicitar al menos una revisión antes de integrar cambios.

## Distribución del trabajo

Toda la implementación se mantiene en el notebook canónico. Quien trabaja
un punto edita las celdas y los documentos de su apartado:

| Punto | Archivos |
|---|---|
| 1 | Celdas 1.1 a 1.5 del notebook, `tests/test_punto1.py`, `01-punto1-regresion-logistica.tex`, `04-unit-tests.tex` |
| 2 | Celdas 2.1 y 2.2 del notebook, `tests/test_punto2.py`, `02-punto2-deteccion-anomalias.tex` |
| 3 | Celdas 3.1 a 3.3 del notebook, `tests/test_punto3.py`, `03-punto3-modelos-lenguaje.tex` |
| Integración | Funciones compartidas, configuración, comparación y pruebas del notebook, `tests/test_utils.py`, `tests/notebook_loader.py`, `00`, `05` y `06` del informe |

Para cambiar un archivo de otra persona, coordinarlo primero con ella.

## Edición del notebook

El notebook contiene la implementación y los experimentos y se trata como un archivo difícil de
fusionar:

- Cada persona edita solo las celdas de su sección; la persona integradora
  edita la configuración, la comparación final y las pruebas.
- Avisar al equipo antes de reorganizar, añadir o eliminar muchas celdas.
- Durante el desarrollo, limpiar las salidas antes de hacer commit
  (**Edit → Clear Outputs of All Cells**) para reducir conflictos.
- Reiniciar el kernel y ejecutar todas las celdas antes de integrar resultados.
- Conservar las tablas y figuras finales únicamente en un commit deliberado de
  preparación de entrega.

## Reproducibilidad

- Mantener imports, semilla, protocolo y decisiones compartidas en la celda de
  configuración del notebook.
- Registrar cambios de dependencias en `requirements.txt`, con la versión
  validada.
- Mantener las funciones de 1.1 y 1.2 como implementación propia del modelo
  afín con sesgo y sigmoide. El 1.3 se ejecuta con datos sintéticos aislados;
  los apartados posteriores usan el dataset de UCI.
- Registrar el codificador y la dimensión real de sus vectores. Cambiarlo
  exige regenerar características, calibración y resultados; no editar el
  dataset original ni reutilizar métricas de otro modelo.
- El protocolo vigente usa MPNet-base-v2 con 768 dimensiones y PCA a
  384, 192 y 2 componentes. Cada corrida selecciona hiperparámetros en una
  validación interna de su entrenamiento; el PCA de selección se ajusta solo
  con el subconjunto de ajuste. La prueba no interviene en esas decisiones.
- Nunca incluir credenciales, tokens, archivos `.env` ni claves de servicios.

## Revisión antes de integrar

- `pytest` pasa usando las definiciones del notebook, sin descargar modelos
  ni ejecutar los experimentos completos.
- El notebook abre sin errores y conserva un orden de ejecución comprensible.
- Las funciones nuevas contienen documentación compatible con Doxygen.
- Los cambios relacionados se reflejan en el informe LaTeX cuando corresponde.
- `TODO.md`, `AGENTS.md` y `docs/report/review.tex` están actualizados;
  `docs/report/review.pdf` se recompiló con `./build_pdf.sh review`.
- No se agregaron cachés, entornos virtuales ni resultados sin revisar.
