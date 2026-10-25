# Datos

- `raw/SMSSpamCollection`: el dataset SMS Spam Collection de UCI, versionado
  para que todo el equipo use el mismo archivo. Si falta, el notebook lo
  descarga del repositorio UCI (`download_sms_spam`). Tiene 5 574
  mensajes (4 827 ham y 747 spam), una línea por mensaje: etiqueta, tabulador
  y texto. No editarlo a mano.
- `processed/`: cachés regenerables, como las características del codificador
  y las predicciones de los modelos de lenguaje. Está excluida de Git; para
  regenerar una caché, borrar el archivo y volver a ejecutar el notebook.

## Fuente y licencia

Almeida, T. A. y Gómez Hidalgo, J. M. (2011). *SMS Spam Collection*. UCI
Machine Learning Repository. https://doi.org/10.24432/C5CC84

Se distribuye con licencia Creative Commons Attribution 4.0 International
(CC BY 4.0).
