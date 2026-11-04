import asyncio

from evaluation.http_load import running_server
from evaluation.serving_checks import check_restart, check_service


def test_real_http_two_workers_ingestion_and_restart(tmp_path):
    root = tmp_path / "snapshots"
    with running_server(root, "test-hash", 2, tmp_path / "first.log") as (url, token, pid):
        expected = asyncio.run(check_service(url, token))
    with running_server(root, "test-hash", 2, tmp_path / "restart.log") as (url, token, pid):
        actual = asyncio.run(check_restart(url, token, expected))
        assert actual["restored_count"] == 13
