"""!
@file tp2_jupyter_config.py
@brief Configuración privada: loopback y token nuevo en cada arranque, sin secretos versionados.
"""

import secrets

c = get_config()  # Proporcionado por Jupyter al cargar este archivo.
c.ServerApp.ip = "127.0.0.1"
c.ServerApp.open_browser = False
c.ServerApp.allow_remote_access = False
c.ServerApp.port_retries = 0
c.ServerApp.log_level = "WARN"
c.IdentityProvider.token = secrets.token_urlsafe(32)
