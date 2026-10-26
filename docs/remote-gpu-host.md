# Host GPU remoto para TP2: Windows, WSL2, Tailscale y PyCharm

Guía preparada el 26/10/2026. El host todavía no se ha configurado ni
validado remotamente: la persona reportó WSL 2.5.9.0, kernel 6.6.87.2-1,
RTX 3080 Ti con 12288 MiB y controlador Windows 591.86. Falta confirmar
el nombre de la distribución y CUDA dentro de Linux. No se cambian los
modelos, hiperparámetros ni decisiones pendientes del punto 3.

## Diseño elegido

WSL2 permite usar CUDA con el controlador NVIDIA de Windows; no hace falta
dual boot para esta arquitectura. No instalar un controlador NVIDIA Linux
dentro de WSL. El paquete CUDA de PyTorch y el controlador son componentes
distintos: `nvidia-smi` en Windows no prueba que el entorno Python use CUDA.

La persona eligió la opción más sencilla y confirmó que los tres integrantes
pueden instalar un cliente. Se propone Tailscale en Windows y en las tres
computadoras cliente, sin dominio ni entrada pública al servidor.

```text
Mac / computadoras del equipo
        |
        | red privada Tailscale, TCP 2222
        v
Windows: Tailscale Serve TCP
        |
        | 127.0.0.1:2222, reenvío localhost de WSL
        v
WSL2: OpenSSH dedicado, 127.0.0.1:2222
        |
        | SSH: shell, SFTP y reenvío local de puertos
        v
Jupyter por usuario: 127.0.0.1:8888 / 8889 / 8890
```

Tailscale puede atravesar NAT y usar relés cuando no hay conexión directa;
no se necesita una IPv4 pública ni port forwarding del router. Se instala
solo en Windows en el host GPU: la guía oficial advierte problemas al
ejecutarlo simultáneamente en Windows y WSL. Tampoco se necesita Tailscale
Funnel, que publicaría un servicio en Internet.

Cada integrante tiene su usuario Linux, su clon y su Jupyter. La GPU y el
sistema operativo se comparten entre personas de confianza; esto no es
aislamiento de procesos hostiles. Una sola instancia de Jupyter con un
usuario compartido no equivale a un servidor multiusuario. JupyterHub puede
ser una evolución posterior, pero no es requisito para estas tres personas.

## 1. Windows: comprobar WSL y GPU

En PowerShell:

```powershell
wsl --version
wsl -l -v
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

Usar el nombre exacto mostrado por `wsl -l -v` en los comandos `-d` siguientes;
`Ubuntu` es un ejemplo. La columna VERSION de esa distribución debe decir 2.
El número 2.5.9.0 corresponde al programa WSL, no sustituye esta comprobación.
Actualizar con `wsl --update` solo si una comprobación posterior lo requiere.

Abrir la distribución y comprobar desde Linux:

```powershell
wsl -d Ubuntu
```

```bash
uname -r
cat /etc/os-release
ps -p 1 -o comm=
/usr/lib/wsl/lib/nvidia-smi
```

Para los servicios de esta guía, PID 1 debe ser `systemd`. Si no lo es,
editar `/etc/wsl.conf` con `sudoedit` y añadir o ajustar esta sección,
conservando las demás opciones existentes:

```ini
[boot]
systemd=true
```

Guardar el trabajo de WSL y, desde PowerShell, ejecutar `wsl --terminate
Ubuntu`, sustituyendo el nombre. Esto detiene los procesos de esa
distribución; abrirla de nuevo y repetir la comprobación de PID 1.

## 2. Preparar la clave del Mac y los archivos del proyecto

En el Mac, crear una clave dedicada con frase de paso y cargarla en el agente
SSH local. La clave privada permanece en el Mac:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_tp2
ssh-add ~/.ssh/id_ed25519_tp2
cat ~/.ssh/id_ed25519_tp2.pub
```

Copiar únicamente la línea pública `.pub` al portapapeles o a un archivo
que se pueda pegar en la consola WSL. Cada integrante genera su propia clave.

En WSL, instalar las herramientas base:

```bash
sudo apt-get update
sudo apt-get install -y git openssh-server python3-venv python3-pip
git clone --branch codex/wsl-gpu-host https://github.com/ManfredAzofeifa/AI-TP2.git "$HOME/AI-TP2"
cd "$HOME/AI-TP2"
```

Si ya existe el clon, no repetir `git clone`: verificar su rama y estado
antes de actualizarlo. Guardar el repositorio en el sistema de archivos Linux
(`~/AI-TP2`), no usar el entorno virtual del Mac ni uno ubicado en `/mnt/c`.

La rama `codex/wsl-gpu-host` incluye esta guía, las plantillas de
`infra/gpu-host/`, el diagnóstico y sus pruebas. No se transfieren archivos
manualmente ni se necesitan claves, tokens, modelos o cachés del Mac.
Esta preparación se publica en su propia rama; `main` conserva su estado.

Para un clon que ya existe, comprobar primero que `git status --short` no
muestra cambios que deban conservarse. Después:

```bash
git fetch origin
git switch --track origin/codex/wsl-gpu-host
```

Si esa rama local ya existe, usar `git switch codex/wsl-gpu-host` en lugar
de `--track`. Para recibir futuras actualizaciones desde esa rama y con el
árbol de trabajo limpio, ejecutar `git pull --ff-only`. No usar `reset --hard`
para descartar trabajo de notebooks o cambios locales.

## 3. WSL: acceso SSH del propietario

Como el usuario Linux que utilizará el propietario:

```bash
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
touch "$HOME/.ssh/authorized_keys"
chmod 600 "$HOME/.ssh/authorized_keys"
nano "$HOME/.ssh/authorized_keys"
```

Añadir la clave pública del Mac en una sola línea, sin borrar claves
existentes. Antes de habilitar el servicio:

```bash
ssh-keygen -lf "$HOME/.ssh/authorized_keys"
sudo groupadd -f tp2-ssh
sudo usermod -aG tp2-ssh "$USER"
sudo ssh-keygen -A
sudo install -d -m 0755 /run/sshd
sudo install -m 0644 infra/gpu-host/sshd_config /etc/ssh/sshd_config_tp2
sudo install -m 0644 infra/gpu-host/tp2-sshd.service /etc/systemd/system/tp2-sshd.service
sudo /usr/sbin/sshd -t -f /etc/ssh/sshd_config_tp2
sudo systemctl daemon-reload
sudo systemctl enable --now tp2-sshd.service
systemctl is-active tp2-sshd.service
ss -ltn | rg '127.0.0.1:2222'
sudo ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
```

Si `rg` no está instalado, usar `ss -ltn` y buscar la dirección indicada.
El servicio dedicado escucha solo en loopback, usa claves, permite SFTP y
reenvío SSH, y admite únicamente al grupo `tp2-ssh`; no modifica la
configuración de un SSH existente en otro puerto. El paquete de Ubuntu
puede iniciar su servicio SSH normal: revisar `ss -ltn` y el estado de
`ssh.service` / `ssh.socket`. Si ese servicio acaba de instalarse para este
proyecto y no se usa para otra cosa, detenerlo con `sudo systemctl disable
--now ssh.service ssh.socket`; no detener un servicio ajeno sin revisarlo.

## 4. Windows: Tailscale privado y puente hacia WSL

Instalar Tailscale desde su [descarga oficial](https://tailscale.com/download/windows)
e iniciar sesión. Invitar las otras dos identidades al mismo tailnet desde
la consola de administración. Cada persona instala Tailscale e inicia sesión
con su propia identidad. No compartir una contraseña de cuenta.

Revisar los permisos de red para que solo esas tres identidades lleguen al
host GPU por TCP 2222. En un tailnet existente, no reemplazar toda la política
ni asumir que una regla específica anula permisos amplios ya presentes.
La política efectiva debe permitir ese puerto al equipo y denegarlo a otras
identidades; hacer la prueba de acceso con ambas clases de usuario.

Con WSL y `tp2-sshd` funcionando, abrir PowerShell y comprobar el salto local:

```powershell
Test-NetConnection 127.0.0.1 -Port 2222
```

Debe devolver `TcpTestSucceeded: True`. WSL reenvía puertos locales a
Windows; si falla, comprobar el servicio en Linux y `localhostForwarding`
en `.wslconfig` antes de añadir reglas de firewall o proxies adicionales.
No cambiar a networking mirrored como primer intento: conservar el modo
que ya funcione y diagnosticar el salto que falla.

En PowerShell con los permisos que solicite el cliente Tailscale:

```powershell
tailscale serve --bg --tcp=2222 tcp://127.0.0.1:2222
tailscale serve status
tailscale ip -4
```

Guardar la IP de Tailscale de Windows. Serve con `--bg` persiste al reiniciar
Tailscale; el servicio Linux de destino debe seguir vivo. No usar
`tailscale funnel` ni crear una regla pública para Jupyter.

## 5. Mac: primera conexión y PyCharm

Copiar `infra/gpu-host/ssh_config.example` al archivo `~/.ssh/config` del Mac,
reemplazando `WINDOWS_TAILSCALE_IP` y `LINUX_USER`. Mantener permisos 600.
La plantilla crea el alias `tp2-gpu` y reenvía el puerto local 18888 hacia
el Jupyter 8888 de WSL.

```bash
ssh tp2-gpu
```

En la primera conexión, comparar la huella con la obtenida en la consola
WSL; no desactivar la verificación de claves del host. Para mantener solo
el túnel, usar `ssh -N tp2-gpu`. Su puerto 18888 escucha únicamente en el Mac.

Si la edición/licencia de PyCharm habilita intérpretes SSH, añadir uno con:
host = IP Tailscale de Windows, puerto = 2222, usuario Linux y clave del Mac.
El intérprete será `~/AI-TP2/.venv/bin/python` dentro de Linux; comprobar su
ruta absoluta con `printf '%s/.venv/bin/python\n' "$(pwd -P)"` desde el clon.
Conservar el enlace del entorno virtual, sin resolverlo al Python del sistema.
No seleccionar el Python
de Windows ni el entorno virtual del Mac. Alternativamente, Remote
Development por SSH abre un proyecto con backend en el host Linux.

Para notebooks, PyCharm puede usar un **External Server**. Con el túnel
abierto, configurar `http://127.0.0.1:18888` y el token privado de ese
usuario, y probar la conexión. No pegar el token en el repositorio ni en
el chat. El servidor se configura por usuario en PyCharm.

## 6. Entorno Python y prueba de CUDA

En el clon WSL, crear un entorno nuevo con una versión de Python compatible
con los pins existentes del equipo:

```bash
cd "$HOME/AI-TP2"
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/gpu_host_status.py --require-cuda --cuda-smoke-test
```

La prueba realiza únicamente una multiplicación de matrices 16 por 16;
no descarga modelos ni ejecuta el notebook. Debe informar `host.wsl=true`,
`cuda.available=true`, la GPU esperada y `smoke_test="passed"`.

Si falla la instalación o CUDA, conservar el mensaje de resolución de
paquetes y el diagnóstico. No bajar versiones, sustituir el pin de PyTorch
por una compilación nocturna ni instalar un controlador Linux a ciegas.
Se debe elegir una rueda CUDA compatible con el driver, Python y la versión
de PyTorch requerida, usando el selector oficial. El CUDA toolkit completo
solo es necesario para herramientas/extensiones que lo requieran; no se
instala preventivamente para ejecutar una rueda de PyTorch.

## 7. Jupyter privado y persistente por usuario

Desde cada usuario Linux y su propio clon:

```bash
mkdir -p "$HOME/.config/systemd/user" "$HOME/.config/tp2"
chmod 700 "$HOME/.config/tp2"
printf 'TP2_JUPYTER_PORT=8888\n' > "$HOME/.config/tp2/jupyter.env"
chmod 600 "$HOME/.config/tp2/jupyter.env"
install -m 0644 infra/gpu-host/tp2-jupyter.service "$HOME/.config/systemd/user/tp2-jupyter.service"
systemctl --user daemon-reload
systemctl --user enable --now tp2-jupyter.service
systemctl --user is-active tp2-jupyter.service
.venv/bin/python -m jupyter server list
```

El último comando es solo para el usuario humano: muestra el token privado.
El administrador debe habilitar antes la persistencia de cada cuenta con
`sudo loginctl enable-linger NOMBRE`; los integrantes sin sudo no ejecutan
ese paso por sí mismos. Después pueden administrar su servicio de usuario.
La configuración genera uno nuevo en cada arranque, conserva la protección
de Jupyter y enlaza solo `127.0.0.1`. No usar tokens vacíos, `allow_origin='*'`
ni desactivar XSRF. Verificar desde el túnel que una llamada a
`/api/contents` sin token devuelve un rechazo, mientras el cliente con su
token accede. Un puerto TCP abierto por sí solo no prueba autenticación.

Los otros dos usuarios eligen puertos distintos, por ejemplo 8889 y 8890,
en su propio `jupyter.env`. Cada uno ajusta el destino de LocalForward de
su Mac/PC a su puerto; puede conservar 18888 como puerto local en su equipo.

Para crear un usuario adicional, el administrador usa `sudo adduser
NOMBRE`, lo incorpora con `sudo usermod -aG tp2-ssh NOMBRE` y prepara
su clave pública en `/home/NOMBRE/.ssh/authorized_keys`, con directorio
700, archivo 600 y propietario NOMBRE. No añadir integrantes a `sudo` ni
copiarles la clave privada del propietario. Iniciar una sesión SSH propia
de cada usuario para crear su entorno y servicio Jupyter. Las contraseñas
de las cuentas se introducen solo en la consola del host; SSH sigue siendo
exclusivamente con clave.

Los tres pueden abrir notebooks, pero comparten 12 GiB de VRAM nominales.
Ejecutar un trabajo GPU pesado a la vez y cerrar kernels que retengan modelos
antes del siguiente. La conectividad no garantiza que tres modelos o un
modelo grande sin cuantizar entren en memoria. Los modelos y LoRA/DoRA se
deciden por separado; esta preparación no los selecciona.

## 8. Mantener disponible WSL

Systemd permite reiniciar servicios, pero por sí solo no mantiene viva la
VM WSL. Mientras se configura el host, dejar abierta esta tarea en Windows,
con el nombre real de la distribución:

```powershell
wsl -d Ubuntu --exec /bin/sleep infinity
```

Para automatizarla, crear una tarea del Programador de tareas de Windows
bajo **la cuenta que posee esa distribución WSL**, con `wsl.exe` como
programa y `-d Ubuntu --exec /bin/sleep infinity` como argumentos. Un inicio
al iniciar sesión requiere esa sesión; si se necesita acceso antes de
iniciar sesión, configurar localmente la tarea para ese usuario con las
credenciales y permisos que solicite Windows. No usar SYSTEM suponiendo que
verá la distribución de otro usuario. Validar un reinicio real antes de
declarar que el host es desatendido.

En Tailscale Windows, habilitar el funcionamiento desatendido si se requiere
que permanezca conectado tras cerrar sesión. Revisar también la caducidad
de identidad del nodo. La PC debe seguir encendida y sin suspensión durante
el trabajo; un túnel no evita la suspensión, los reinicios ni los cortes de red.

## 9. Diagnósticos y herramientas del IDE

El diagnóstico de infraestructura no lee tokens Jupyter, claves, variables
de entorno, contenido de notebooks ni logs completos. Desde el Mac:

```bash
ssh tp2-gpu 'cd ~/AI-TP2 && .venv/bin/python scripts/gpu_host_status.py --port 8888'
```

Es un comando explícito en WSL y produce JSON con identidad del host,
versiones, GPU/CUDA, servicios, puertos de loopback y estado Git. Puede
usarse como contexto de herramientas de desarrollo. No guardar tokens en
las órdenes ni devolver `jupyter server list` como diagnóstico compartido.

PyCharm puede seguir en el Mac, con su servidor MCP en la dirección local
que muestre Settings | Tools | MCP Server. Usar la configuración real que
copia el IDE; no adivinar puerto ni endpoint, y no publicarlo mediante
Funnel o un túnel web. Un intérprete remoto no convierte automáticamente
la terminal integrada o todos los comandos MCP en comandos remotos.

Antes de ejecutar un experimento, comprobar en el **kernel conectado**:

```python
import platform, sys, torch
print(platform.node(), sys.executable)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "sin CUDA")
```

La salida debe corresponder al host WSL y su GPU, no al Mac. Verificar además
que una edición y una ejecución de celda afectan al notebook y kernel
remotos esperados. El soporte concreto del MCP instalado debe probarse;
tener SSH no garantiza que un plugin pueda gestionar un backend remoto.

## 10. Verificación y deshacer

Antes de dar el host por operativo, verificar desde otra red: Tailscale,
SSH con clave correcta, rechazo sin clave, Jupyter sin token rechazado,
Jupyter con token funcionando, CUDA en ese kernel, archivos por usuario y
persistencia tras cerrar SSH. Después, verificar reinicio Windows/WSL y
desconexión/reconexión. Ninguna de esas pruebas remotas se ha realizado aún.

Para retirar el acceso, revocar primero el equipo/usuario en Tailscale y su
clave en `authorized_keys`. Para apagar solo el puente configurado aquí:

```powershell
tailscale serve --tcp=2222 off
```

En WSL, deshabilitar los servicios propios según corresponda:

```bash
systemctl --user disable --now tp2-jupyter.service
sudo systemctl disable --now tp2-sshd.service
```

Retirar la tarea Windows si se creó. No usar `tailscale serve reset` ni
borrar reglas globales, claves o servicios que pertenezcan a otros proyectos.

## Cloudflare como alternativa

Cloudflare Tunnel también funciona sin IP pública mediante conexiones
salientes. Para una URL web estable se combinaría con Access y una política
de identidad; SSH privado normalmente requiere además su cliente o un
proxy compatible. No es necesario para el esquema elegido y no se creó
ningún recurso Cloudflare. Si más adelante se requiere un portal web sin
cliente, evaluar esa opción o JupyterHub como un cambio separado.

## Fuentes oficiales consultadas el 26/10/2026

- [NVIDIA: CUDA en WSL](https://docs.nvidia.com/cuda/wsl-user-guide/).
- [Microsoft: systemd en WSL](https://learn.microsoft.com/en-us/windows/wsl/systemd).
- [Microsoft: red de WSL](https://learn.microsoft.com/en-us/windows/wsl/networking).
- [Tailscale: Windows y WSL2](https://tailscale.com/docs/install/windows/wsl2).
- [Tailscale Serve TCP](https://tailscale.com/docs/reference/tailscale-cli/serve).
- [Tailscale: planes actuales](https://tailscale.com/pricing).
- [Jupyter: servidores y usuarios](https://jupyter-server.readthedocs.io/en/stable/operators/public-server.html).
- [Jupyter: autenticación](https://jupyter-server.readthedocs.io/en/latest/operators/security.html).
- [PyTorch: instalación CUDA](https://pytorch.org/get-started/locally/).
- [PyCharm: servidor Jupyter externo](https://www.jetbrains.com/help/pycharm/configuring-jupyter-notebook.html).
- [PyCharm: intérprete SSH](https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-ssh.html).
- [PyCharm: MCP](https://www.jetbrains.com/help/pycharm/mcp-server.html).
- [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/).
