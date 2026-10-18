# Datos

- `raw/`: el archivo `SMSSpamCollection`, que el notebook descarga del
  repositorio UCI la primera vez (`data.download_sms_spam`).
- `processed/`: cachés regenerables, como las características del codificador
  y las predicciones de los modelos de lenguaje.

Ambas carpetas están excluidas de Git. Para regenerar una caché, borrar el
archivo correspondiente y volver a ejecutar el notebook.
