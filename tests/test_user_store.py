"""UserStore audit log filters."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from common.user_store import UserStore, parse_log_bound


def test_parse_log_bound_date_only_uses_shanghai():
    start = parse_log_bound("2026-09-13")
    end = parse_log_bound("2026-09-13", end=True)
    assert start.startswith("2026-09-12T16:00:00")
    assert end.startswith("2026-09-13T15:59:59")


def test_list_logs_filters_user_type_time(tmp_path):
    store = UserStore(tmp_path / "u.sqlite")
    store.write_log("admin", "登录", "a")
    store.write_log("herb", "登录", "b")
    store.write_log("admin", "用户管理", "c")
    store.write_log("adm%in", "登录", "wild")

    rows = store._conn.execute(
        "SELECT id FROM audit_logs ORDER BY rowid"
    ).fetchall()
    t0 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    for i, row in enumerate(rows):
        stamp = (t0 + timedelta(days=i)).isoformat()
        store._conn.execute(
            "UPDATE audit_logs SET created_at = ? WHERE id = ?",
            (stamp, row["id"]),
        )
    store._conn.commit()

    by_user = store.list_logs(username="herb")
    assert by_user["total"] == 1
    assert by_user["items"][0]["username"] == "herb"

    by_type = store.list_logs(operate_type="登录")
    assert by_type["total"] == 3
    assert {x["username"] for x in by_type["items"]} == {"admin", "herb", "adm%in"}

    escaped = store.list_logs(username="adm%in")
    assert escaped["total"] == 1
    assert escaped["items"][0]["username"] == "adm%in"

    mid = (t0 + timedelta(days=1)).isoformat()
    ranged = store.list_logs(start=mid)
    assert ranged["total"] == 3
    assert all(x["created_at"] >= mid for x in ranged["items"])

    until = t0.isoformat()
    early = store.list_logs(end=until)
    assert early["total"] == 1
    assert early["items"][0]["username"] == "admin"
    assert early["items"][0]["operate_type"] == "登录"

    page = store.list_logs(operate_type="登录", limit=1, offset=0)
    page2 = store.list_logs(operate_type="登录", limit=1, offset=1)
    assert page["total"] == page2["total"] == 3
    assert page["items"][0]["id"] != page2["items"][0]["id"]

    types = set(store.list_logs()["operate_types"])
    assert "登录" in types
    assert "用户管理" in types

    with pytest.raises(ValueError, match="开始时间"):
        store.list_logs(start="2026-09-10", end="2026-09-01")
    store.close()
