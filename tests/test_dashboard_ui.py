from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import pytest


def test_dashboard_has_claude_style_research_console_features():
    html = Path("dashboard/index.html").read_text(encoding="utf-8")
    required = [
        "训练动态",
        "指标浏览器",
        "动态采样器",
        "linear",
        "log",
        "chartTooltip",
        "metricSearch",
        "samplerTable",
        "feedStream",
        "seriesToggle",
        "crosshair",
        "Claude 风格",
        "总体进度",
        "Fold 进度",
        "Epoch 进度",
        "预计剩余",
        "translate3d",
        "历史 Champion",
    ]
    for token in required:
        assert token in html, token
    assert "ensureChartScaffold" in html
    assert "plotLayer" in html
    assert "panChartLayer" in html
    chart_fn = html.split("function chart(", 1)[1].split("function statusLabel", 1)[0]
    assert "svg.innerHTML=''" not in chart_fn


def test_dashboard_keeps_existing_runtime_contracts():
    html = Path("dashboard/index.html").read_text(encoding="utf-8")
    assert "state.json" in html
    assert "history.json" in html
    assert "leaderboard" in html
    assert "mps_driver_gb" in html
    assert "available_gb" in html


def test_dashboard_merges_model_explorer_into_monitor_and_keeps_paginated_data_view():
    html = Path("dashboard/index.html").read_text(encoding="utf-8")
    required = [
        'data-view="overviewView">监控',
        'data-view="dataView"',
        'id="overviewView"',
        'id="dataView"',
        'id="modelSelect"',
        'id="modelRuns"',
        'id="dataTable"',
        'id="dataPrev"',
        'id="dataNext"',
        'id="dataPage"',
        "Legacy baseline",
        "renderModelExplorer",
        "renderDataExplorer",
        "PAGE_SIZE",
    ]
    for token in required:
        assert token in html, token
    assert 'data-view="modelsView"' not in html
    assert 'id="modelsView"' not in html
    overview = html.split('id="overviewView"', 1)[1].split('id="dataView"', 1)[0]
    assert 'id="modelSelect"' in overview
    assert 'id="modelRuns"' in overview
    assert "Current contenders" not in overview


def test_dashboard_keeps_rbt3_only_as_selectable_legacy_history():
    html = Path("dashboard/index.html").read_text(encoding="utf-8")
    assert "RBT3 是 Legacy baseline" in html
    assert "RBT3 · legacy only · no retraining" not in html
    overview = html.split('id="overviewView"', 1)[1].split('id="dataView"', 1)[0]
    assert "Legacy baseline F1" not in overview
    assert 'id="modelKpi"' in overview


def test_dashboard_renders_readable_runtime_config_and_live_training_stream():
    html = Path("dashboard/index.html").read_text(encoding="utf-8")
    required = [
        'id="runtimeConfigChips"',
        'id="trainingStream"',
        "renderRuntimeConfig",
        "renderTrainingStream",
        "最近 160 个点",
        "当前 Epoch",
        "Step",
    ]
    for token in required:
        assert token in html, token
    assert "runtimeConfig').textContent=cur.runtime_config" not in html


def test_dashboard_charts_use_recent_window_and_dynamic_y_range():
    html = Path("dashboard/index.html").read_text(encoding="utf-8")
    assert "slice(-160)" in html
    assert "dynamicPad" in html
    assert "max:1" not in html.split("function renderCharts", 1)[1].split("function buildMetricBrowser", 1)[0]


def test_dashboard_inline_javascript_parses_with_node():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    html = Path("dashboard/index.html").read_text(encoding="utf-8")
    match = re.search(r"<script>(.*)</script>", html, flags=re.S)
    assert match, "inline script not found"
    with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8") as handle:
        handle.write(match.group(1))
        handle.flush()
        result = subprocess.run([node, "--check", handle.name], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
