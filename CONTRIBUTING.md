# Guía de colaboración

Las tareas, las reglas del proyecto (independencia de archivos, contratos y
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

## Propiedad de los archivos

Cada punto tiene sus propios archivos; quien trabaja un punto edita solo
estos:

| Punto | Archivos |
|---|---|
| 1 | `src/tp2/punto1.py`, `tests/test_punto1.py`, sección 1 del notebook, `01-punto1-regresion-logistica.tex`, `04-unit-tests.tex` |
| 2 | `src/tp2/punto2.py`, `tests/test_punto2.py`, sección 2 del notebook, `02-punto2-deteccion-anomalias.tex` |
| 3 | `src/tp2/punto3.py`, `tests/test_punto3.py`, sección 3 del notebook, `03-punto3-modelos-lenguaje.tex` |
| Integración | `src/tp2/utils.py`, `tests/test_utils.py`, configuración, comparación y pruebas del notebook, `00`, `05` y `06` del informe |

Para cambiar un archivo de otra persona, coordinarlo primero con ella.

## Edición del notebook

El notebook es el punto de integración y se trata como un archivo difícil de
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
- Nunca incluir credenciales, tokens, archivos `.env` ni claves de servicios.

## Revisión antes de integrar

- `pytest` pasa, incluida la prueba de independencia de archivos.
- El notebook abre sin errores y conserva un orden de ejecución comprensible.
- Las funciones nuevas contienen documentación compatible con Doxygen.
- Los cambios relacionados se reflejan en el informe LaTeX cuando corresponde.
- `TODO.md` y `AGENTS.md` están actualizados (casillas y estado actual).
- No se agregaron cachés, entornos virtuales ni resultados sin revisar.
