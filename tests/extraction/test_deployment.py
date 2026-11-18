"""Contracts for the processor required by both standard RAG deployments."""

import json
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROCESSOR_REF = "c7bfe421243c66d4d5d9c6e3d08d276586ac47bf"
PROCESSOR_CONTEXT = (
    f"https://github.com/LibreChat-AI/code-interpreter.git#{PROCESSOR_REF}"
)


def stage_compose_files(directory):
    for name in ("docker-compose.yaml", "api-compose.yaml", "extraction-compose.yaml"):
        shutil.copyfile(ROOT / name, directory / name)
    # Never read deployment credentials while rendering or qualifying the service.
    (directory / ".env").write_text("")
    return directory


def compose_command(directory, project, *files):
    command = ["docker", "compose", "--env-file", "/dev/null", "-p", project]
    for file in files:
        command.extend(["-f", str(directory / file)])
    return command


def render(directory, *files):
    command = compose_command(directory, f"rag-contract-{uuid.uuid4().hex}", *files)
    result = subprocess.run(
        [*command, "config", "--no-env-resolution", "--format", "json"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Compose render failed ({result.returncode}):\n{result.stderr}"
        )
    return json.loads(result.stdout)


def assert_build(service):
    assert service["build"]["context"] == PROCESSOR_CONTEXT
    assert service["build"]["dockerfile"] == "packages/extraction/Dockerfile"
    assert service["build"]["target"] == "runtime"


def assert_credentials_mount(service, readonly):
    mount = next(
        item for item in service["volumes"] if item["target"] == "/run/extraction"
    )
    assert mount["type"] == "volume" and mount["source"] == "extraction-credentials"
    assert mount.get("read_only", False) is readonly


def assert_sandbox(service):
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert service["security_opt"] == ["no-new-privileges:true"]
    for key in (
        "env_file",
        "ports",
        "expose",
        "profiles",
        "privileged",
        "cap_add",
        "entrypoint",
        "container_name",
        "group_add",
    ):
        assert not service.get(key), key


def assert_hardened_processor(service):
    assert_build(service)
    assert_sandbox(service)
    assert service["init"] is True and service["user"] == "10001:10001"
    assert set(service["networks"]) == {"extraction"}
    assert "network_mode" not in service
    assert (
        service["depends_on"]["extraction-init"]["condition"]
        == "service_completed_successfully"
    )
    assert not service.get("healthcheck")
    assert service["environment"] == {"EXTRACTION_TOKEN_FILE": "/run/extraction/token"}
    assert int(service["mem_limit"]) == 768 * 1024 * 1024
    assert float(service["cpus"]) == 2 and int(service["pids_limit"]) == 32
    assert service["tmpfs"] == [
        "/jobs:rw,noexec,nosuid,nodev,size=96m,nr_inodes=1024,uid=10001,gid=10001,mode=0700"
    ]
    assert len(service["volumes"]) == 1
    assert_credentials_mount(service, True)
    assert not service.get("command")


def assert_provisioning(config):
    service = config["services"]["extraction-init"]
    assert_build(service)
    assert_sandbox(service)
    assert service["build"] == config["services"]["extraction"]["build"]
    assert service["command"] == ["node", "dist/provision.js"]
    assert service["user"] == "10001:10001" and service["network_mode"] == "none"
    assert service["healthcheck"] == {"disable": True}
    assert int(service["mem_limit"]) == 128 * 1024 * 1024
    assert float(service["cpus"]) == 0.25 and int(service["pids_limit"]) == 16
    assert service["environment"] == {"EXTRACTION_TOKEN_FILE": "/run/extraction/token"}
    assert len(service["volumes"]) == 1
    assert_credentials_mount(service, False)
    assert config["networks"]["extraction"]["internal"] is True
    assert "extraction-credentials" in config["volumes"]
    assert "extraction-socket" not in config["volumes"]


def assert_caller(app):
    assert not app.get("group_add") and not app.get("profiles")
    assert set(app["networks"]) == {"default", "extraction"}
    assert app["depends_on"]["extraction"]["condition"] == "service_healthy"
    assert_credentials_mount(app, True)
    assert (
        not {
            "RAG_EXTRACTION_ENABLED",
            "RAG_EXTRACTION_FORMATS",
            "RAG_EXTRACTION_SOCKET",
        }
        & app.get("environment", {}).keys()
    )


@pytest.mark.parametrize(
    "recipe,caller",
    [("docker-compose.yaml", "fastapi"), ("api-compose.yaml", "rag_api")],
)
def test_standard_deployment_provisions_hardened_processor(tmp_path, recipe, caller):
    config = render(stage_compose_files(tmp_path), recipe)
    assert_hardened_processor(config["services"]["extraction"])
    assert_provisioning(config)
    assert_caller(config["services"][caller])


@pytest.mark.parametrize(
    "recipes",
    [
        ("docker-compose.yaml", "api-compose.yaml"),
        ("api-compose.yaml", "docker-compose.yaml"),
    ],
)
def test_combined_recipes_share_provisioning_and_processor(tmp_path, recipes):
    config = render(stage_compose_files(tmp_path), *recipes)
    assert set(config["services"]) == {
        "db",
        "fastapi",
        "rag_api",
        "extraction-init",
        "extraction",
    }
    assert_hardened_processor(config["services"]["extraction"])
    assert_provisioning(config)
    for caller in ("fastapi", "rag_api"):
        assert_caller(config["services"][caller])


def test_render_failure_includes_stderr_without_dumping_model(monkeypatch, tmp_path):
    diagnostic = "services.extraction.security_opt items at 0 and 1 are equal"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            1,
            stdout="do not dump the rendered application model",
            stderr=diagnostic,
        ),
    )
    with pytest.raises(AssertionError, match="security_opt") as caught:
        render(tmp_path, "docker-compose.yaml", "api-compose.yaml")
    assert diagnostic in str(caught.value)
    assert "do not dump" not in str(caught.value)
