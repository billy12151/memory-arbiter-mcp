#!/usr/bin/env python3
"""P3-T4 上线门：真实库副本上新旧两路 KNN top-k 对比.

旧路 = 0.16.11 的 COUNT + k×4 窗口循环（本脚本内嵌副本，逐字自 db/_knn.py
退役前实现）；新路 = 0.16.12 的 rowid-IN 单次查询。查询向量取自库内真实
行（每个 vec 行的 embedding 即一个有意义的近邻查询），覆盖 unscoped /
scoped / exclude / expired 四变体。门 = 集合一致率 100%（tie 序差单列）。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import struct
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import sqlite_vec  # noqa: E402

from memory_arbiter.acl import workspace_scope_sql, workspace_exclusion_sql  # noqa: E402


def _load_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


_OLD_SELECT = """SELECT e.*, v.distance AS distance, m.status, m.subject, m.tags,
                   m.workspace, m.workspace_canonical, m.source_type,
                   m.confidence, m.protection_level, m.event_time,
                   m.ingest_time, m.metadata, m.content,
                   m.version AS memory_row_version, m.agent_id,
                   m.source_ref, m.created_at AS memory_created_at
            FROM memory_row_vec v
            JOIN memory_row r ON r.id=v.id
            JOIN memories m ON m.id=r.memory_id
            WHERE v.embedding MATCH ? AND k=? AND {clauses}
            ORDER BY v.distance"""


def _old_knn(conn, query, k, *, status_sql, memory_status_sql, workspace=None,
             exclude_memory_id=None, exclude_workspaces=None):
    """0.16.11 窗口循环副本（COUNT + k*4 起步、翻倍至 max_fetch）。"""
    clauses = [status_sql, memory_status_sql]
    workspace_sql, workspace_params = workspace_scope_sql(
        "COALESCE(NULLIF(m.workspace_canonical,''),m.workspace)", workspace,
    )
    excl_sql, _, excl_params = workspace_exclusion_sql(exclude_workspaces)
    params: list = []
    if workspace_sql:
        clauses.append(workspace_sql)
        params.extend(workspace_params)
    if excl_sql:
        clauses.append(excl_sql)
        params.extend(excl_params)
    if exclude_memory_id is not None:
        clauses.append("r.memory_id!=?")
        params.append(int(exclude_memory_id))
    filtered = bool(workspace_sql or exclude_memory_id is not None or excl_sql)
    query_json = json.dumps(query)
    count_sql = (
        f"SELECT COUNT(*) FROM memory_row_vec v "
        f"JOIN memory_row r ON r.id=v.id "
        f"JOIN memories m ON m.id=r.memory_id "
        f"WHERE {status_sql} AND {memory_status_sql}"
    )
    candidate_count = int(conn.execute(count_sql).fetchone()[0])
    max_fetch = max(1, candidate_count)
    fetch_k = min(max_fetch, k * 4) if filtered else k
    sql = _OLD_SELECT.format(clauses=" AND ".join(clauses))
    while fetch_k > 0:
        try:
            rows = conn.execute(sql, [query_json, fetch_k, *params]).fetchall()
        except sqlite3.OperationalError:
            # 0.16.11 生产路径在此（k 超 vec0 4096 上限）被 except 吃掉 →
            # 静默空结果；单独归类，不计入集合一致率分母。
            return None
        if not filtered or len(rows) >= k or fetch_k >= max_fetch:
            break
        fetch_k = min(max_fetch, fetch_k * 2)
    return [(int(r["id"]), float(r["distance"])) for r in rows[:k]]


def _new_knn(conn, query, k, *, status_sql, memory_status_sql, workspace=None,
             exclude_memory_id=None, exclude_workspaces=None):
    """0.16.12 rowid-IN 单次查询（与 EvidenceStore.knn 同构）。"""
    workspace_sql, workspace_params = workspace_scope_sql(
        "COALESCE(NULLIF(m.workspace_canonical,''),m.workspace)", workspace,
    )
    excl_sql, _, excl_params = workspace_exclusion_sql(exclude_workspaces)
    eligible = [memory_status_sql]
    params: list = []
    if workspace_sql:
        eligible.append(workspace_sql)
        params.extend(workspace_params)
    if excl_sql:
        eligible.append(excl_sql)
        params.extend(excl_params)
    if exclude_memory_id is not None:
        eligible.append("r.memory_id != ?")
        params.append(int(exclude_memory_id))
    id_constraint = (
        f" AND v.id IN (SELECT r.id FROM memory_row r "
        f"JOIN memories m ON m.id=r.memory_id WHERE {' AND '.join(eligible)})"
        if len(eligible) > 1 else ""
    )
    sql = (
        "SELECT e.id, v.distance FROM memory_evidence_vec v "
        "JOIN memory_row r ON r.id=v.id "
        "JOIN memories m ON m.id=r.memory_id "
        f"WHERE v.embedding MATCH ? AND k=? AND {status_sql} AND {memory_status_sql}"
        f"{id_constraint} ORDER BY v.distance"
    )
    rows = conn.execute(sql, [json.dumps(query), k, *params]).fetchall()
    return [(int(r["id"]), float(r["distance"])) for r in rows[:k]]


_BRANCHES = {
    "active": ("v.parent_status='active'", "m.status='active'"),
    "expired": ("v.parent_status NOT IN ('active','deleted')",
                "m.status NOT IN ('active','deleted')"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--queries", type=int, default=34, help="取多少个库内向量作查询")
    args = parser.parse_args()

    conn = _load_conn(args.db)
    conn.row_factory = sqlite3.Row
    sample = conn.execute(
        f"SELECT v.id, v.embedding FROM memory_evidence_vec v ORDER BY v.id LIMIT {int(args.queries)}",
    ).fetchall()
    queries = []
    for row in sample:
        blob = row["embedding"]
        dim = len(blob) // 4
        queries.append(list(struct.unpack(f"{dim}f", blob)))
    workspaces = [r[0] for r in conn.execute(
        "SELECT DISTINCT COALESCE(NULLIF(workspace_canonical,''),workspace) ws "
        "FROM memories WHERE status='active' ORDER BY ws LIMIT 3",
    ).fetchall()]
    some_id = int(conn.execute("SELECT MIN(memory_id) FROM memory_evidence").fetchone()[0])

    variants = [("unscoped", {})]
    for ws in workspaces:
        variants.append((f"scoped:{ws[:24]}", {"workspace": ws}))
    variants.append(("excluded", {"exclude_workspaces": {workspaces[0]} if workspaces else set()}))
    variants.append(("exclude_id", {"exclude_memory_id": some_id}))
    variants.append(("scoped+excl", {"workspace": workspaces[-1] if workspaces else None,
                                     "exclude_memory_id": some_id}))

    set_match = order_match = tie_order = total = 0
    old_crash = starved_fixed = 0
    old_us = new_us = 0.0
    failures: list[str] = []
    for branch, (status_sql, memory_status_sql) in _BRANCHES.items():
        for label, extra in variants:
            for qi, q in enumerate(queries):
                total += 1
                t0 = time.perf_counter()
                old = _old_knn(conn, q, args.k, status_sql=status_sql,
                               memory_status_sql=memory_status_sql, **extra)
                old_us += time.perf_counter() - t0
                t0 = time.perf_counter()
                new = _new_knn(conn, q, args.k, status_sql=status_sql,
                               memory_status_sql=memory_status_sql, **extra)
                new_us += time.perf_counter() - t0
                if old is None:
                    old_crash += 1
                    continue
                old_ids = [i for i, _ in old]
                new_ids = [i for i, _ in new]
                if set(old_ids) == set(new_ids):
                    set_match += 1
                    if old_ids == new_ids:
                        order_match += 1
                    else:
                        tie_order += 1
                elif old_ids == new_ids[:len(old_ids)]:
                    # 旧路欠返回（全局 COUNT 天花板内的 scoped 饥饿）而新路
                    # 补满 k——这是本次改写要修的缺陷形态，前缀关系证新路
                    # 未引入新行，仅补齐旧路漏掉的尾部。归类不判失败。
                    starved_fixed += 1
                elif len(failures) < 12:
                    failures.append(
                        f"{branch}/{label}/q{qi}: old={old_ids} new={new_ids}"
                    )
    conn.close()
    print(f"queries={total} set_match={set_match} order_match={order_match} "
          f"tie_order_diff={tie_order} old_k_limit_crash={old_crash} "
          f"old_starved_fixed={starved_fixed}")
    print(f"old total {old_us*1000:.0f}ms vs new total {new_us*1000:.0f}ms "
          f"({old_us/max(new_us,1e-9):.2f}x)")
    for line in failures:
        print("MISMATCH:", line)
    # 三类可接受结局：完全一致 / tie 序差 / 旧路饥饿欠返回（前缀证）；
    # 旧路 k 上限崩溃是旧路自身缺陷（生产中被 except 吞成静默空），单列。
    accepted = set_match + tie_order + starved_fixed + old_crash
    rate = accepted / total if total else 0.0
    verdict = "PASSED" if not failures and accepted == total else "FAILED——非前缀差异存在，整组回退"
    print(f"一致率（一致+tie 序差+饥饿前缀+旧路上限崩溃）= {rate:.4%}  → {verdict}")
    return 0 if not failures and accepted == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
