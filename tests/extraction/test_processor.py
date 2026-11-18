"""Current Python client qualification against the default hardened deployment."""

import json
import subprocess
import uuid
from pathlib import Path

import aiofiles
import pytest

from tests.extraction.test_deployment import (
    assert_hardened_processor,
    compose_command,
    render,
    stage_compose_files,
)

pytestmark = pytest.mark.integration
CALLER_IMAGE = (
    "python:3.13-slim-bookworm@"
    "sha256:5024f48ba9441d4b13a95d3945abc6365538e3a31109833367a1923523c6efed"
)
GATEWAY = "/opt/extraction/gateway"


def docker(*args, check=True):
    result = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=40
    )
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_python_client_to_default_hardened_processor(tmp_path):
    directory = stage_compose_files(tmp_path)
    assert_hardened_processor(
        render(directory, "docker-compose.yaml")["services"]["extraction"]
    )
    run_id = uuid.uuid4().hex
    project = f"rag-extraction-{run_id}"
    volume = f"{project}_extraction-credentials"
    network = f"{project}_extraction"
    caller = f"rag-extraction-caller-{run_id}"
    compose = compose_command(directory, project, "docker-compose.yaml")
    images = [f"{project}-extraction", f"{project}-extraction-init"]

    def run_compose(*args, check=True, timeout=40):
        result = subprocess.run(
            [*compose, *args], capture_output=True, text=True, timeout=timeout
        )
        if check:
            assert result.returncode == 0, result.stdout + result.stderr
        return result

    root = Path(__file__).resolve().parents[2]
    client = root / "app/services/extraction_client.py"
    fixtures = Path(__file__).parent / "fixtures"
    library = Path(aiofiles.__file__).parent
    try:
        run_compose("build", "extraction-init", "extraction", timeout=240)
        run_compose("up", "-d", "--wait", "--wait-timeout", "30", "extraction")
        name = run_compose("ps", "-q", "extraction").stdout.strip()
        image = docker("inspect", "--format", "{{.Config.Image}}", name).stdout.strip()
        assert (
            json.loads(docker("network", "inspect", network).stdout)[0]["Internal"]
            is True
        )
        runtime = json.loads(docker("inspect", name).stdout)[0]
        host = runtime["HostConfig"]
        assert (
            host["NetworkMode"] == network
            and host["ReadonlyRootfs"]
            and not host["Privileged"]
        )
        assert (
            host["CapDrop"] == ["ALL"]
            and "no-new-privileges:true" in host["SecurityOpt"]
        )
        assert "seccomp=unconfined" not in host["SecurityOpt"]
        assert runtime["Config"]["User"] == "10001:10001"
        assert runtime["Config"]["Entrypoint"] == [GATEWAY]
        assert runtime["Config"]["Healthcheck"]["Test"] == [
            "CMD",
            GATEWAY,
            "node",
            "dist/health.js",
        ]
        init_name = run_compose("ps", "-a", "-q", "extraction-init").stdout.strip()
        provisioner = json.loads(docker("inspect", init_name).stdout)[0]
        assert provisioner["Config"]["User"] == "10001:10001"
        assert provisioner["Config"]["Entrypoint"] == [GATEWAY]
        assert provisioner["Config"]["Cmd"] == ["node", "dist/provision.js"]
        assert provisioner["Config"]["Healthcheck"]["Test"] == ["NONE"]
        # Compose adds service-specific image labels; the pinned payload must match.
        provisioner_image = json.loads(
            docker("image", "inspect", provisioner["Image"]).stdout
        )[0]
        runtime_image = json.loads(docker("image", "inspect", runtime["Image"]).stdout)[
            0
        ]
        assert provisioner_image["RootFS"] == runtime_image["RootFS"]
        assert provisioner["State"]["ExitCode"] == 0
        assert not host["PortBindings"]
        assert set(runtime["NetworkSettings"]["Networks"]) == {network}
        assert host["Memory"] == 768 * 1024 * 1024 and host["NanoCpus"] == 2 * 10**9
        assert host["PidsLimit"] == 32 and set(host["Tmpfs"]) == {"/jobs"}
        assert (
            len(runtime["Mounts"]) == 1
            and runtime["Mounts"][0]["Destination"] == "/run/extraction"
            and runtime["Mounts"][0]["RW"] is False
        )
        script = """
import asyncio
import errno
import os
import stat
import sys
sys.path.insert(0, "/")
from dataclasses import replace
from pathlib import Path
from client import ExtractionClient, ExtractionError, ExtractionSettings
async def check():
    settings = ExtractionSettings.from_env()
    assert settings.url == "http://extraction:8000"
    assert settings.token_file == "/run/extraction/token"
    assert stat.S_IMODE(os.stat("/run/extraction").st_mode) == 0o755
    assert stat.S_IMODE(os.stat(settings.token_file).st_mode) == 0o444
    try:
        open("/run/extraction/not-writable", "wb")
    except OSError as error:
        assert error.errno == errno.EROFS
    else:
        raise AssertionError("caller credential mount is writable")
    for token_file in ("/jobs/missing-token", "/jobs/wrong-token"):
        if token_file.endswith("wrong-token"):
            token = Path(settings.token_file).read_text()
            Path(token_file).write_text(("a" if token[0] != "a" else "b") + token[1:])
        try:
            await ExtractionClient(replace(settings, token_file=token_file)).extract("/fixtures/pages.pdf", "pdf")
        except ExtractionError as error:
            assert (error.code, error.status_code) == ("UNAVAILABLE", 503)
        else:
            raise AssertionError("missing/wrong service credentials accepted")
    # Stop a real HTTP upload before the body completes, then reuse admission.
    cancellation_input = Path("/jobs/cancel.pdf")
    cancellation_input.write_bytes(b"%PDF-1.7\\n" + b"x" * 131072)
    entered = asyncio.Event()
    class SlowInput:
        def __init__(self, file):
            self.file = file
            self.sent = False
        async def read(self, size):
            if not self.sent:
                self.sent = True
                return await self.file.read(1)
            entered.set()
            await asyncio.Future()
    class CancellingClient(ExtractionClient):
        async def _request(self, method, token, **kwargs):
            if method == "POST":
                kwargs["file"] = SlowInput(kwargs["file"])
            return await super()._request(method, token, **kwargs)
    cancelled_client = CancellingClient(settings)
    task = asyncio.create_task(cancelled_client.extract(str(cancellation_input), "pdf"))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("extraction did not propagate cancellation")
    assert cancelled_client._admitted == 0
    client = ExtractionClient(settings)
    pdf = await client.extract("/fixtures/pages.pdf", "pdf")
    assert pdf == [
        {"kind": "page", "index": 1, "text": "First page text"},
        {"kind": "page", "index": 2, "text": "Second page text"},
    ], pdf
    docx = await client.extract("/fixtures/body.docx", "docx")
    assert len(docx) == 1 and docx[0]["kind"] == "document" and docx[0]["index"] == 1
    text = docx[0]["text"]
    assert text.index("First paragraph") < text.index("Left cell") < text.index("Right cell") < text.index("Last paragraph")
asyncio.run(check())
print("Python client to hardened PDF/DOCX processor passed")
"""
        # The trusted RAG caller uses Python's normal asyncio self-pipe, not the worker gateway.
        caller_command = [
            "run",
            "--rm",
            "--name",
            caller,
            "--init",
            "--network",
            network,
            "--read-only",
            "--user",
            "65532:65532",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--mount",
            f"type=volume,src={volume},dst=/run/extraction,readonly",
            "--tmpfs",
            "/jobs:rw,noexec,nosuid,nodev,size=1m,uid=65532,gid=65532,mode=0700",
            "--mount",
            f"type=bind,src={client},dst=/client.py,readonly",
            "--mount",
            f"type=bind,src={fixtures},dst=/fixtures,readonly",
            "--mount",
            f"type=bind,src={library},dst=/usr/local/lib/python3.13/site-packages/aiofiles,readonly",
            CALLER_IMAGE,
            "python",
            "-B",
            "-c",
            script,
        ]
        result = docker(*caller_command)
        assert "processor passed" in result.stdout
        # Re-provisioning preserves credentials already loaded by the running service.
        run_compose("run", "--rm", "--no-deps", "extraction-init")
        docker("exec", name, GATEWAY, "node", "dist/health.js")
        run_compose(
            "up",
            "-d",
            "--wait",
            "--wait-timeout",
            "30",
            "--scale",
            "extraction=2",
            "extraction",
        )
        replicas = run_compose("ps", "-q", "extraction").stdout.split()
        assert len(replicas) == 2
        for replica in replicas:
            docker("exec", replica, GATEWAY, "node", "dist/health.js")
        assert "processor passed" in docker(*caller_command).stdout
        run_compose("restart", "extraction")
        run_compose(
            "up",
            "-d",
            "--wait",
            "--wait-timeout",
            "30",
            "--scale",
            "extraction=2",
            "extraction",
        )
        assert "processor passed" in docker(*caller_command).stdout
        # Exercise the shipped gateway rather than a test-only seccomp policy.
        probe = """
import errno
import socket
for operation in (lambda: socket.socket(socket.AF_UNIX), socket.socketpair):
    try:
        operation()
    except OSError as error:
        assert error.errno == errno.EPERM
    else:
        raise AssertionError("gateway permitted Unix sockets")
"""
        docker("exec", name, GATEWAY, "python", "-c", probe)
        assert (
            docker(
                "exec",
                name,
                GATEWAY,
                "node",
                "-e",
                'if(require("node:fs").readdirSync("/jobs").length)process.exit(1)',
            ).returncode
            == 0
        )
    finally:
        docker("rm", "-f", caller, check=False)
        run_compose("down", "-v", "--remove-orphans", check=False)
        for image in images:
            docker("image", "rm", image, check=False)
