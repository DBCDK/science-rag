#!/usr/bin/env python3
"""
:mod:`science_rag.service._dbc_optional` -- optional dbc_pyutils integration

======================
Optional dbc_pyutils
======================

``dbc_pyutils`` is only installed via the ``dbc`` dependency group (DBC
cluster hardware / CI), not by a plain ``uv sync``. This module is the single
seam that gates on its presence: everywhere else in the ``service`` package
imports from here instead of importing ``dbc_pyutils`` directly, so nothing
outside this file needs an ``ImportError`` handler.

When ``dbc_pyutils`` is unavailable, ``DBC_AVAILABLE`` is ``False`` and every
other symbol below is ``None`` — callers branch on ``DBC_AVAILABLE`` rather
than probing the symbols themselves.
"""

try:
    from dbc_pyutils import Statistics
    from dbc_pyutils import build_info
    from dbc_pyutils import create_instance_id
    from dbc_pyutils import setup_logging
    from dbc_pyutils.base_handler_fastapi import install_base_handler
    from dbc_pyutils.metric_handler_fastapi import PrometheusMiddleware
    from dbc_pyutils.metric_handler_fastapi import metrics_endpoint

    DBC_AVAILABLE = True
except ImportError:
    Statistics = None
    build_info = None
    create_instance_id = None
    setup_logging = None
    install_base_handler = None
    PrometheusMiddleware = None
    metrics_endpoint = None

    DBC_AVAILABLE = False

__all__ = [
    "DBC_AVAILABLE",
    "Statistics",
    "build_info",
    "create_instance_id",
    "setup_logging",
    "install_base_handler",
    "PrometheusMiddleware",
    "metrics_endpoint",
]
