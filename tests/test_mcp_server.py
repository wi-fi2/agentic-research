import asyncio

import mcp_server


def test_tools_registered():
    names = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert names == {"research", "list_runs", "get_report"}


def test_get_report_unknown_run(tmp_path, monkeypatch):
    from app import store
    monkeypatch.setattr(store.settings, "DB_PATH", str(tmp_path / "t.db"), raising=False)
    monkeypatch.setattr(store, "_conn", None, raising=False)
    mcp_server._ready = False
    try:
        mcp_server.get_report("nope")
    except ValueError as e:
        assert "No run" in str(e)
    else:
        raise AssertionError("expected ValueError")
