"""CLI 통합 테스트.

실제 ultralytics 모델을 다운로드하지 않기 위해 --help 동작만 검증한다.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_cli_help() -> None:
    """`python main.py --help`이 0 종료, 사용법에 --input 포함."""
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=30,
    )
    assert result.returncode == 0, (
        f"main.py --help failed: stderr={result.stderr}"
    )
    assert "--input" in result.stdout
    assert "--output" in result.stdout


def test_cli_help_lists_model_arg() -> None:
    """--model 옵션도 --help에 노출되어야 한다."""
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=30,
    )
    assert result.returncode == 0
    assert "--model" in result.stdout
