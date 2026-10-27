# Jupyter de WSL desde PyCharm en la misma LAN

## Diagnóstico y configuración verificados

El 26/10/2026, Jupyter escuchaba en `127.0.0.1:8888` dentro de WSL.
Windows reenviaba localhost a WSL, pero no había un puerto publicado en su
dirección LAN. Desde el Mac, la conexión expiraba. El túnel no era una
restricción impuesta por Jupyter: era el único camino configurado.

Se conservó Jupyter en localhost y se añadió un reenvío de la dirección LAN
de Windows a `127.0.0.1:8888` de Windows. Este último ya llega a WSL mediante
su reenvío de localhost. No se reinició WSL ni Jupyter para configurar la red.
La regla de firewall permite únicamente TCP 8888 desde la dirección del Mac
y en la interfaz LAN elegida; conserva el perfil de red de Windows.

En PowerShell **como administrador**, sustituir las tres variables por los
valores actuales. Comprobar primero que `http://127.0.0.1:8888/login` responde
desde Windows y que no existe otro reenvío para esa dirección y puerto:

```powershell
$serverAddress = '192.168.1.22'
$clientAddress = '192.168.1.205'
$lanInterface = 'House LAN'
Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8888/login' -TimeoutSec 5
netsh interface portproxy show v4tov4
netsh interface portproxy add v4tov4 listenaddress=$serverAddress listenport=8888 connectaddress=127.0.0.1 connectport=8888
New-NetFirewallRule -Name 'TP2-Jupyter-LAN-Mac' -DisplayName 'TP2 Jupyter desde Mac LAN' -Direction Inbound -Action Allow -Protocol TCP -LocalAddress $serverAddress -LocalPort 8888 -RemoteAddress $clientAddress -InterfaceAlias $lanInterface -Profile Any
```

La regla ya está creada en el equipo usado para el experimento; no repetir
`New-NetFirewallRule` si existe. `iphlpsvc` debe estar en ejecución y el
reenvío de localhost de WSL debe seguir disponible. No se modificó `.wslconfig`.
Si DHCP cambia una dirección, actualizar el reenvío o la dirección remota de
la regla. Reservar ambas direcciones en el router evita ese cambio.

## PyCharm y comprobación

En **Configure Jupyter Server**, editar el servidor existente de la 3080 Ti:
usar `http://192.168.1.22:8888` como URL y conservar su token de Jupyter en el
campo privado del IDE. **Test connection** debe mostrar **Connected**.
El notebook debe seleccionar ese servidor externo, no el servidor local del Mac.
No guardar el token en fuentes, comandos, capturas ni documentación.

Desde el Mac se verificó HTTP 200 en `/login` y HTTP 403 en `/api/kernels`
sin autenticación. Después, PyCharm ejecutó por MCP una celda y confirmó
Linux, el Python de `/home/zekken/AI-TP2/.venv` y la RTX 3080 Ti.
El túnel de Jupyter se retiró después de verificar esa ejecución: la URL
anterior dejó de responder y la LAN siguió disponible. Otros procesos de
túnel se conservaron. La LAN evita
el recorrido por Cloudflare; no acelera por sí misma la inferencia CUDA.
Mantener el acceso en la red local de confianza y conservar la autenticación.

Para retirar solamente esta publicación, en PowerShell administrador:

```powershell
netsh interface portproxy delete v4tov4 listenaddress=192.168.1.22 listenport=8888
Remove-NetFirewallRule -Name 'TP2-Jupyter-LAN-Mac'
```

## Fuentes

- [Red de WSL y acceso desde LAN](https://learn.microsoft.com/windows/wsl/networking).
- [Opciones de Jupyter Server](https://jupyter-server.readthedocs.io/en/stable/other/full-config.html).

## Recuperación del experimento

El kernel que descargaba Mistral ya no existía al retomar el trabajo.
Su salida conservada registra `zmq.error.ZMQError: Too many open files`
durante actualizaciones de widgets del progreso Xet; esto no demuestra por
sí solo qué recurso agotó primero. Los tres archivos completos de pesos se
habían comprobado contra los SHA-256 oficiales y protegido con enlaces duros.
Se recuperaron sus blobs sin descargar de nuevo los 14 496 080 928 bytes.
Las cargas siguientes usan HTTP y desactivan barras de progreso de Hugging
Face para evitar ese camino de widgets. Se conservan el modelo, su revisión,
el prompt y las reglas de evaluación. No atribuir el fallo al túnel sin evidencia.
