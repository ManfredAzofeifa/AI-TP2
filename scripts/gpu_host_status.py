"""!
@file gpu_host_status.py
@brief Diagnóstico JSON del host remoto sin leer tokens, claves ni contenido de notebooks.
"""

import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys


def command(arguments, timeout=5):
    """!
    @brief Ejecuta un diagnóstico acotado sin shell y descarta stderr.
    @param arguments Lista de argumentos de un comando de solo lectura.
    @param timeout Tiempo máximo en segundos.
    @return Tupla de código de salida y stdout; ante un error, código None y texto vacío.
    """
    try:
        result = subprocess.run(
            arguments, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, ""
    return result.returncode, result.stdout.strip() if result.returncode == 0 else ""


def cuda_status(smoke_test=False):
    """!
    @brief Consulta PyTorch y opcionalmente verifica una multiplicación pequeña en CUDA.
    @param smoke_test Autoriza una multiplicación 16 por 16, sin descargar modelos.
    @return Diccionario de versión, CUDA, GPU y resultado de la prueba solicitada.
    """
    result = {"installed": False, "available": False, "smoke_test": "not_requested"}
    try:
        torch = importlib.import_module("torch")
        result.update(installed=True, version=torch.__version__, compiled_cuda=torch.version.cuda)
        result["available"] = bool(torch.cuda.is_available())
        if result["available"]:
            result["devices"] = [
                {"index": index, "name": torch.cuda.get_device_name(index),
                 "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory)}
                for index in range(torch.cuda.device_count())
            ]
        if smoke_test:
            if not result["available"]:
                result["smoke_test"] = "cuda_unavailable"
            else:
                with torch.no_grad():
                    sample = torch.ones((16, 16), device="cuda")
                    product = sample @ sample
                    valid = bool(torch.isfinite(product).all()) and float(product.sum()) == 4096.0
                    torch.cuda.synchronize()
                result["smoke_test"] = "passed" if valid else "failed"
    except Exception as error:
        # No se serializa el mensaje de error: puede contener rutas o valores privados.
        result["error_type"] = type(error).__name__
        if smoke_test:
            result["smoke_test"] = "failed"
    return result


def host_status(repo, jupyter_ports=(8888, 8889, 8890), smoke_test=False):
    """!
    @brief Recoge identidad, GPU, servicios, puertos locales y estado Git sin credenciales.
    @param repo Ruta del repositorio; no se leen diffs ni contenido de archivos privados.
    @param jupyter_ports Puertos locales que se comprueban mediante una conexión TCP breve.
    @param smoke_test Autoriza la prueba pequeña de CUDA.
    @return Diagnóstico JSON serializable; describe el host real que ejecuta el comando.
    """
    repo = Path(repo).resolve()
    result = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": {"name": platform.node(), "system": platform.system(),
                 "release": platform.release(), "machine": platform.machine(),
                 "wsl": "microsoft" in platform.release().lower()},
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "packages": {}, "services": {}, "jupyter_tcp": {},
    }
    for name in ("torch", "jupyterlab", "jupyter-server", "transformers", "sentence-transformers"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    nvidia = shutil.which("nvidia-smi")
    if not nvidia and Path("/usr/lib/wsl/lib/nvidia-smi").is_file():
        nvidia = "/usr/lib/wsl/lib/nvidia-smi"
    code, output = command([
        nvidia or "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    result["nvidia_smi"] = {"ok": code == 0, "devices": []}
    if code == 0:
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 3 and re.fullmatch(r"[0-9.]+", parts[1]):
                result["nvidia_smi"]["devices"].append(
                    {"name": parts[0], "memory_total_mib": float(parts[1]), "driver": parts[2]}
                )
    result["cuda"] = cuda_status(smoke_test)
    for name, arguments in (
        ("ssh", ["systemctl", "is-active", "tp2-sshd.service"]),
        ("jupyter_user", ["systemctl", "--user", "is-active", "tp2-jupyter.service"]),
    ):
        code, value = command(arguments)
        result["services"][name] = value if value in {"active", "activating"} else "not_active_or_unavailable"
    for port in jupyter_ports:
        try:
            with socket.create_connection(("127.0.0.1", int(port)), timeout=0.2):
                listening = True
        except OSError:
            listening = False
        result["jupyter_tcp"][str(port)] = {"reachable_on_loopback": listening,
                                             "authentication_verified": False}
    git = {}
    for key, arguments in (
        ("commit", ["git", "-C", str(repo), "rev-parse", "HEAD"]),
        ("branch", ["git", "-C", str(repo), "branch", "--show-current"]),
        ("status", ["git", "-C", str(repo), "status", "--porcelain"]),
    ):
        code, output = command(arguments)
        git[key] = (bool(output) if key == "status" else output) if code == 0 else None
    git["dirty"] = git.pop("status")
    result["git"] = git
    return result


def main():
    """!
    @brief Imprime JSON y permite exigir CUDA para una comprobación previa a experimentos.
    @return Código 1 si se exige CUDA y no está disponible o falla la prueba solicitada; 0 en otro caso.
    """
    parser = argparse.ArgumentParser(description="Diagnóstico del host GPU sin modelos ni secretos.")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--port", action="append", type=int, help="Puerto Jupyter; se puede repetir.")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--cuda-smoke-test", action="store_true")
    args = parser.parse_args()
    if args.port and any(not 1 <= port <= 65535 for port in args.port):
        parser.error("Los puertos deben estar entre 1 y 65535.")
    report = host_status(args.repo, args.port or (8888, 8889, 8890), args.cuda_smoke_test)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    failed = not report["cuda"]["available"] or report["cuda"]["smoke_test"] == "failed"
    return int((args.require_cuda and failed) or (args.cuda_smoke_test and failed))


if __name__ == "__main__":
    raise SystemExit(main())
