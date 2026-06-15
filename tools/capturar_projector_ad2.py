from __future__ import annotations

import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "output" / "tensorboard" / "ad2"
SCREENSHOT_DIR = ROOT / "output" / "ad2_screenshots"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


@dataclass(frozen=True)
class Projection:
    key: str
    tensor_name: str
    records: Path
    metadata: Path
    search: str
    filename: str


PROJECTIONS = [
    Projection(
        key="documento",
        tensor_name="Documento - BERTimbau [CLS]",
        records=ROOT / "projecao" / "documento" / "records_documento_768_base_CLS.tsv",
        metadata=ROOT / "projecao" / "documento" / "meta_documento_768_base_CLS.tsv",
        search="Brasil",
        filename="projector_documento.png",
    ),
    Projection(
        key="token",
        tensor_name="Tokens - BERTimbau pooling",
        records=ROOT / "projecao" / "token" / "DOALL_records_token_768_base_POOL.tsv",
        metadata=ROOT / "projecao" / "token" / "DOALL_meta_token_768_base_POOL.tsv",
        search="Brasil",
        filename="projector_token.png",
    ),
    Projection(
        key="token_documento",
        tensor_name="Tokens e Documento - BERTimbau pooling",
        records=ROOT / "projecao" / "token_documento" / "DOALL_records_token_documento_768_base_POOL.tsv",
        metadata=ROOT / "projecao" / "token_documento" / "DOALL_meta_token_documento_768_base_POOL.tsv",
        search="Brasil",
        filename="projector_token_documento.png",
    ),
    Projection(
        key="sentenca_documento",
        tensor_name="Sentenca e Documento - BERTimbau pooling",
        records=ROOT / "projecao" / "sentenca_documento" / "DOALL_records_sentenca_documento_768_base.tsv",
        metadata=ROOT / "projecao" / "sentenca_documento" / "DOALL_meta_sentenca_documento_768_base.tsv",
        search="Brasil",
        filename="projector_sentenca_documento.png",
    ),
]


def prepare_logdir(projection: Projection) -> Path:
    run_dir = LOG_ROOT / projection.key
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(projection.records, run_dir / "records.tsv")
    shutil.copyfile(projection.metadata, run_dir / "metadata.tsv")
    config = (
        "embeddings {\n"
        f'  tensor_name: "{projection.tensor_name}"\n'
        '  tensor_path: "records.tsv"\n'
        '  metadata_path: "metadata.tsv"\n'
        "}\n"
    )
    (run_dir / "projector_config.pbtxt").write_text(config, encoding="utf-8")
    return run_dir


def wait_for_tensorboard(port: int) -> None:
    url = f"http://127.0.0.1:{port}/data/plugin/projector/info?run=."
    last_error: Exception | None = None
    for _ in range(60):
        try:
            urllib.request.urlopen(url, timeout=1).read()
            return
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"TensorBoard did not become ready on port {port}: {last_error}")


def capture_projection(projection: Projection, port: int) -> Path:
    run_dir = prepare_logdir(projection)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCREENSHOT_DIR / projection.filename

    process = subprocess.Popen(
        [
            str(PYTHON),
            "-m",
            "tensorboard.main",
            "--logdir",
            str(run_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_for_tensorboard(port)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=False,
                args=["--ignore-gpu-blocklist", "--enable-webgl", "--enable-gpu"],
            )
            page = browser.new_page(viewport={"width": 1440, "height": 980}, device_scale_factor=1)
            page.goto(f"http://127.0.0.1:{port}/#projector", wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(12000)
            page.mouse.click(1195, 175)
            page.keyboard.type(projection.search)
            page.wait_for_timeout(6000)
            page.screenshot(path=str(out_path), full_page=True)
            browser.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    return out_path


def main() -> None:
    for index, projection in enumerate(PROJECTIONS):
        path = capture_projection(projection, 6010 + index)
        print(path)


if __name__ == "__main__":
    main()
