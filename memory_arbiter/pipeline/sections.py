"""Section split/catalog/rebuild pipeline for MemoryTools (Phase 4 extraction)."""
# mypy: disable-error-code="no-any-return,type-arg,arg-type,index"
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Optional, Tuple, TYPE_CHECKING

from ..db import MemoryDB
from ..models import MemoryRecord

if TYPE_CHECKING:
    from ..tools import MemoryTools


class SectionPipeline:
    def __init__(self, tools: "MemoryTools"):
        self._tools = tools

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tools, name)

    @staticmethod
    def _catalog_entry(s: dict) -> dict:
        """Unified section-catalog schema (same shape in zero-match and partial branches)."""
        return {
            "section_id": s["id"],
            "title": s.get("title"),
            "title_path": s.get("title_path"),
            "summary": s.get("summary"),
            "embedding_truncated": bool(s.get("embedding_truncated")),
            "embedding_original_tokens": s.get("embedding_original_tokens", 0),
            "embedding_used_tokens": s.get("embedding_used_tokens", 0),
        }

    def _attach_sections(
        self,
        results: list[dict[str, Any]],
        query_embedding: Optional[list[float]],
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        """Post-process search results: attach section enhancement for active-split memories.

        v0.8.0 protocol (design doc §6.3). Each result carries a top-level
        ``content`` that is the directly-consumable complete content unit, and
        a ``content_scope`` tag the caller must use to interpret it:

          * full_memory      — the whole memory content
          * matched_sections — the matched sections' full text, joined

        Branch matrix:
          | coverage ≥ threshold            | full_memory   | (matched refs)      | –            |
          | 0 < coverage < threshold        | matched_sections | full section bodies | catalog(unmatched) |
          | memory hit, section zero-match  | full_memory   | –                   | –            |
          | invariant broken / vec gate down| full_memory   | –                   | optional     |

        Ordinary ``matched_sections`` never carry embedding budget diagnostics;
        those live only in catalog/get/doctor (or debug_ranking).
        """
        if not results or not self.db.db_available:
            return results

        vec_state = self.db.get_vec_index_state()
        vec_gate_open = (
            vec_state.get("state") == "ready"
            and query_embedding is not None
            and self.db.state.sqlite_vec_available
        )
        if not query_embedding:
            vec_disabled_reason = "no_query_embedding"
        elif vec_state.get("state") in {"mismatch", "failed"}:
            vec_disabled_reason = (
                "embedding_space_mismatch"
                if vec_state.get("state") == "mismatch"
                else "embedding_migration_failed"
            )
        elif vec_state.get("state") != "ready":
            vec_disabled_reason = "gate_closed_state_not_ready"
        else:
            vec_disabled_reason = "vec_extension_unavailable"
        threshold = self.settings.section_vec_distance_threshold
        fulltext_threshold = self.settings.section_fulltext_threshold

        active_ids = [
            r.get("id") for r in results
            if r.get("split_status") == "active"
        ]
        if not active_ids:
            return results

        # Read all sections + section vec IDs in one snapshot
        sections_map: dict[int, list[dict]] = {}
        section_vec_ids_map: dict[int, set[int]] = {}
        current_mem_map: dict[int, dict] = {}
        try:
            with self.db.connection() as conn:
                for mid in active_ids:
                    mem = MemoryDB._fetch_memory(conn, mid)
                    if mem is None or mem.get("status") == "deleted":
                        continue
                    if mem.get("split_status") != "active":
                        continue
                    current_mem_map[mid] = mem
                    sections_map[mid] = MemoryDB._get_sections(conn, mid)
                    section_vec_ids_map[mid] = MemoryDB._get_section_vec_ids(conn, mid)
        except Exception as exc:
            warnings.append(f"attach_sections read failed: {exc}")
            return results

        for result in results:
            mid = result.get("id")
            if mid not in current_mem_map:
                continue

            # Normalise content: Channel-6 candidates carry content="" upstream,
            # but current_mem_map has the full text. Ensure the result content is
            # the real full text before branching.
            real_content = current_mem_map[mid].get("content") or ""
            if real_content and not result.get("content"):
                result["content"] = real_content
            full_content = result.get("content") or ""

            sections = sections_map.get(mid, [])
            total_sections = len(sections)
            sec_by_id: dict[int, dict] = {s["id"]: s for s in sections}

            # ---- Invariant guards: always return full memory, but flag the
            # corruption so it is detectable regardless of the vec gate.
            if total_sections == 0:
                result.setdefault("warnings", []).append("split_invariant_broken_empty_sections")
                result["content_scope"] = "full_memory"
                result["section_enhancement_applied"] = False
                result["content"] = full_content
                continue
            if total_sections == 1:
                result.setdefault("warnings", []).append("split_invariant_broken_too_few_sections")
                result["content_scope"] = "full_memory"
                result["section_enhancement_applied"] = False
                result["content"] = full_content
                continue
            section_ids = {s["id"] for s in sections}
            vec_ids = section_vec_ids_map.get(mid, set())
            if section_ids - vec_ids:
                result.setdefault("warnings", []).append("split_invariant_broken_missing_section_vec")
                result["content_scope"] = "full_memory"
                result["section_enhancement_applied"] = False
                result["content"] = full_content
                continue

            # ---- Vec gate closed → return full memory (explicit degrade).
            if not vec_gate_open:
                result.setdefault("warnings", []).append(f"vec_disabled={vec_disabled_reason}")
                result["content_scope"] = "full_memory"
                result["section_enhancement_applied"] = False
                result["content"] = full_content
                continue

            # ---- Section vec matching
            try:
                vec_hits = self.db.section_vec_distance_match(mid, query_embedding, threshold)
            except Exception:
                vec_hits = []

            matched_ids = {h["section_id"] for h in vec_hits}
            matched_count = len(matched_ids)

            # Build matched_sections with FULL section bodies, ordered by index.
            # Embedding diagnostics are deliberately omitted from ordinary search.
            def _matched_entry(h: dict) -> dict:
                s = sec_by_id.get(h["section_id"], {})
                body = full_content[s.get("start_offset", 0):s.get("end_offset", 0)] if s else ""
                return {
                    "section_id": h["section_id"],
                    "section_index": s.get("section_index"),
                    "title": s.get("title"),
                    "title_path": s.get("title_path"),
                    "summary": s.get("summary"),
                    "content": body,
                    "char_count": len(body),
                }

            if matched_count == 0:
                # Zero section match → return the FULL memory (design §6.3).
                # No preview, no truncation.
                result["content_scope"] = "full_memory"
                result["content"] = full_content
                result["section_enhancement_applied"] = True
                result["zero_section_match"] = True
                result["hint"] = (
                    f"已拆分为 {total_sections} 段，零段落命中阈值，已返回完整全文"
                )
            elif matched_count / total_sections >= fulltext_threshold:
                # Coverage ≥ threshold → return full memory.
                ordered = sorted(vec_hits, key=lambda h: (sec_by_id.get(h["section_id"], {}).get("section_index", 0)))
                result["content_scope"] = "full_memory"
                result["content"] = full_content
                result["section_enhancement_applied"] = True
                result["matched_sections"] = [_matched_entry(h) for h in ordered]
                pct = round(100 * matched_count / total_sections)
                result["hint"] = f"{pct}% 段落命中，建议直接看全文"
            else:
                # Partial match → join matched sections' full text by index.
                ordered = sorted(vec_hits, key=lambda h: (sec_by_id.get(h["section_id"], {}).get("section_index", 0)))
                matched = [_matched_entry(h) for h in ordered]
                joined = "\n\n".join(m["content"] for m in matched)
                result["content_scope"] = "matched_sections"
                result["content"] = joined
                result["section_enhancement_applied"] = True
                result["matched_sections"] = matched
                result["matched_section_count"] = len(matched)
                result["total_section_count"] = total_sections
                # Catalog of UNMATCHED sections (diagnostic fields allowed here).
                result["section_catalog"] = [
                    self._catalog_entry(s) for s in sections if s["id"] not in matched_ids
                ]
                result["hint"] = "已返回命中段落完整原文；未命中段落见 section_catalog"

        return results

    @staticmethod
    def _find_nth_occurrence(text: str, anchor: str, occurrence: int) -> int:
        """Find the start position of the n-th (0-based) occurrence of anchor in text."""
        start = 0
        for i in range(occurrence + 1):
            pos = text.find(anchor, start)
            if pos == -1:
                return -1
            if i == occurrence:
                return pos
            start = pos + 1
        return -1

    def _compute_offsets(
        self,
        content: str,
        sections_data: list[dict[str, Any]],
        trust_planner_offsets: bool = False,
    ) -> Optional[list[dict[str, Any]]]:
        """Compute global offsets for each section.

        Returns list of {start_offset, end_offset, ...section_data} (with the
        internal ``_planner_start_offset`` stripped) or None on failure.

        Two location strategies, selected by ``trust_planner_offsets``:

        * True  — the rules path. The deterministic fence-aware parser already
          knows each heading's real offset; it passes ``_planner_start_offset``
          per section. We trust it but STILL cross-check the anchor text appears
          at that offset (defense in depth against a parser bug). This avoids
          the silent mis-segmentation that a raw substring search caused when a
          heading line also appeared inside a fenced code block / body text
          (the search was not fence-aware, so it could locate an earlier,
          wrong occurrence that still passed continuity/coverage validation).

        * False — the Agent path. Only anchors + occurrence_index are trusted;
          any offset field the caller supplied is IGNORED (design §5.1: LLM
          offsets must be ignored), and each non-first section is located via
          ``_find_nth_occurrence``.
        """
        result: list[dict[str, Any]] = []
        for i, sec in enumerate(sections_data):
            if i == 0:
                local_start = 0
            else:
                anchor = sec.get("anchor_text")
                if not anchor:
                    return None
                planner_off = sec.get("_planner_start_offset")
                if trust_planner_offsets and isinstance(planner_off, int):
                    if planner_off < 0 or planner_off > len(content):
                        return None
                    # Anchored cross-check: the parser's offset must point at
                    # the anchor text exactly. Catches any planner drift.
                    if not content.startswith(anchor, planner_off):
                        return None
                    local_start = planner_off
                else:
                    occ = sec.get("occurrence_index", 0)
                    local_start = self._find_nth_occurrence(content, anchor, occ)
                    if local_start == -1:
                        return None
            # Drop the internal planner hint so it never reaches the DB / response.
            entry = {k: v for k, v in sec.items() if k != "_planner_start_offset"}
            entry["start_offset"] = local_start
            result.append(entry)

        # Derive end_offsets
        for i in range(len(result)):
            if i < len(result) - 1:
                result[i]["end_offset"] = result[i + 1]["start_offset"]
            else:
                result[i]["end_offset"] = len(content)

        # Validate
        offsets = [(r["start_offset"], r["end_offset"]) for r in result]
        for i in range(len(offsets)):
            if offsets[i][0] >= offsets[i][1]:
                return None
            if i > 0 and offsets[i][0] != offsets[i - 1][1]:
                return None
        if offsets[0][0] != 0 or offsets[-1][1] != len(content):
            return None
        # Check strict increase
        starts = [o[0] for o in offsets]
        if starts != sorted(set(starts)):
            return None
        # Coverage
        if "".join(content[s:e] for s, e in offsets) != content:
            return None

        return result

    @staticmethod
    def _rule_plan_sections(
        content: str, max_sections: int, max_section_chars: int,
    ) -> tuple[Optional[list[dict[str, Any]]], str]:
        """Build a publishable rule plan from fenced-code-safe Markdown headings.

        Design §7. Returns (sections, reason). When sections is non-None the
        plan is immediately publishable (count/size/coverage all satisfied).
        When sections is None, ``reason`` explains why the caller should fall
        back to an Agent split_request:

          * no_rule_structure            — no heading found
          * rule_section_count_out_of_range — <2 or >max_sections headings
          * rule_section_too_large       — some section slice > max_section_chars

        Each section carries title/anchor_text/occurrence_index/title_path and
        a ``_planner_start_offset`` (0 for the first section, the heading's real
        offset otherwise). v0.8.0: the publish helper now trusts this offset on
        the rules path (provenance='parser') and only cross-checks the anchor
        sits there — it no longer re-derives offsets via a non-fence-aware
        substring search, which silently mis-segmented when a heading line also
        appeared in a fenced code block. summary is left empty (rules path does
        not generate summaries; the Agent path does).
        """
        if not content:
            return None, "no_rule_structure"

        # Parse headings outside fenced code, capturing offset/level/raw/title.
        in_fence = False
        fence_marker: Optional[str] = None
        raw_headings: list[dict[str, Any]] = []
        pos = 0
        for line in content.splitlines(keepends=True):
            stripped = line.lstrip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                marker = stripped[:3]
                if not in_fence:
                    in_fence = True
                    fence_marker = marker
                elif marker == fence_marker:
                    in_fence = False
                    fence_marker = None
            elif not in_fence:
                m = re.match(r"^(#{1,6})\s+(.+?)(?:\s+#+)?\s*$", line.rstrip())
                if m:
                    raw_headings.append({
                        "offset": pos,
                        "level": len(m.group(1)),
                        "raw_line": line.rstrip("\n"),
                        "title": m.group(2).strip(),
                    })
            pos += len(line)

        if len(raw_headings) < 2:
            return None, "no_rule_structure"

        # occurrence_index: n-th occurrence of the same raw_line in document order.
        seen: dict[str, int] = {}
        # title_path via a heading-level stack.
        stack: list[tuple[int, str]] = []

        sections: list[dict[str, Any]] = []
        for idx, h in enumerate(raw_headings):
            raw = h["raw_line"]
            occ = seen.get(raw, 0)
            seen[raw] = occ + 1
            # Maintain the level stack for title_path.
            while stack and stack[-1][0] >= h["level"]:
                stack.pop()
            stack.append((h["level"], h["title"]))
            title_path = " / ".join(t for _, t in stack) if len(stack) > 1 else None

            if idx == 0:
                # First section: spans from 0 to the next heading. Its anchor is
                # the heading itself, but the publish helper treats the first
                # section as starting at offset 0 (anchor ignored for sec 0).
                anchor = raw
            else:
                anchor = raw
            sections.append({
                "title": h["title"],
                "summary": "",
                "anchor_text": anchor,
                "occurrence_index": occ,
                "title_path": title_path,
                # Trusted offset from the deterministic parser (method B fix).
                # Section 0 starts at 0 (preamble归入第一段); others start at
                # their heading. The publish helper verifies the anchor sits here.
                "_planner_start_offset": 0 if idx == 0 else h["offset"],
            })

        # Count gate.
        if len(sections) < 2 or len(sections) > max_sections:
            return None, "rule_section_count_out_of_range"

        # Size gate: compute each section's slice length from heading offsets.
        offsets = [h["offset"] for h in raw_headings]
        bounds = offsets + [len(content)]
        for i in range(len(bounds) - 1):
            if bounds[i + 1] - bounds[i] > max_section_chars:
                return None, "rule_section_too_large"

        return sections, ""

    @staticmethod
    def _split_snapshot_error(
        memory: dict[str, Any],
        decision_content_hash: Optional[str],
        decision_memory_version: Optional[int],
        decision_split_status: Optional[str],
        decision_split_revision: Optional[int],
        allowed_split_statuses: tuple[Optional[str], ...],
    ) -> Optional[str]:
        """Validate a caller's prepare snapshot against the current row."""
        if (
            not decision_content_hash
            or decision_memory_version is None
            or decision_split_revision is None
        ):
            return "decision snapshot fields are required"
        current_hash = hashlib.sha256(
            str(memory.get("content") or "").encode("utf-8")
        ).hexdigest()
        if (
            memory.get("status") != "active"
            or current_hash != decision_content_hash
            or int(memory.get("version") or 1) != int(decision_memory_version)
        ):
            return "memory_changed"
        if (
            memory.get("split_status") != decision_split_status
            or int(memory.get("split_revision") or 0) != int(decision_split_revision)
            or decision_split_status not in allowed_split_statuses
        ):
            return "split_revision_conflict"
        return None

    # ------------------------------------------------------------------
    #  v0.8.0: Unified publish helper (design doc §9.1)
    #
    #  Shared by the rules path (memory_write/edit auto-split) and the Agent
    #  path (memory_split publish/rebuild). Replaces the inline validate-then-
    #  write block that previously lived only inside memory_split.
    #
    #  Provenance is now an explicit caller argument ("parser" for rules,
    #  "agent" for memory_split) instead of inferred from anchor text — the
    #  old heuristic guessed "parser" when an anchor happened to equal a
    #  heading string, which conflated the two paths.
    #
    #  Failure semantics (design doc §5.3 / §9.2):
    #    * decision_kind="split": a real failure marks split_status=failed
    #      via CAS (_mark_split_failed), and returns an error.
    #    * decision_kind="rebuild": failures NEVER touch split_status — the
    #      old active sections stay intact. Only an error is returned.
    # ------------------------------------------------------------------

    def _publish_sections(
        self,
        memory_id: int,
        content: str,
        sections_data: list[dict[str, Any]],
        decision_content_hash: str,
        decision_memory_version: int,
        decision_split_status: Optional[str],
        decision_split_revision: int,
        decision_kind: str,
        provenance: str,
    ) -> dict[str, Any]:
        """Validate, embed, and atomically publish sections + section vectors.

        Returns a state.response() dict. On failure, the original content is
        untouched. ``decision_kind`` is "split" (initial publish from
        NULL/failed/declined) or "rebuild" (replace existing active sections).
        """
        mid = memory_id
        max_sections = self.settings.max_sections
        max_section_chars = self.settings.max_section_chars

        # 1) Count gate.
        if len(sections_data) < 2 or len(sections_data) > max_sections:
            return self.db.state.response({
                "error": f"sections count must be 2..{max_sections}, got {len(sections_data)}",
            }, ok=False)

        # 2) Compute + validate offsets (anchor→offset, continuity, coverage).
        #    Pure text computation — run it before touching the embedder so a
        #    bad anchor fails fast and (for split) records the failure reason.
        #    The rules path trusts the parser's offsets (verified); the Agent
        #    path re-derives from anchors and ignores any caller offset.
        offset_result = self._compute_offsets(
            content, sections_data, trust_planner_offsets=(provenance == "parser"),
        )
        if offset_result is None:
            if decision_kind == "split":
                self._mark_split_failed(
                    mid, decision_content_hash, decision_memory_version,
                    decision_split_revision, decision_split_status,
                    "validation", "offset computation failed",
                )
            return self.db.state.response({"error": "offset validation failed"}, ok=False)

        # 3) Section-size hard gate (design doc §6.2): a section slice that
        #    exceeds max_section_chars would embed only its front portion,
        #    producing a misleading section vector. Reject before embedding.
        for i, sec in enumerate(offset_result):
            slice_len = sec["end_offset"] - sec["start_offset"]
            if slice_len > max_section_chars:
                if decision_kind == "split":
                    self._mark_split_failed(
                        mid, decision_content_hash, decision_memory_version,
                        decision_split_revision, decision_split_status,
                        "validation", f"section {i} too large: {slice_len}>{max_section_chars}",
                    )
                return self.db.state.response({
                    "error": f"section_too_large: section {i} is {slice_len} chars (max {max_section_chars})",
                }, ok=False)

        # 4) Vec state + embedder must be ready before the expensive embedding.
        vec_state = self.db.get_vec_index_state()
        if vec_state.get("state") != "ready":
            return self.db.state.response({
                "error": "vec index not ready, complete migration first",
                "vec_index_state": vec_state,
            }, ok=False)
        embedder, _ = self._ensure_embedder()
        if embedder is None:
            return self.db.state.response({"error": "embedder unavailable"}, ok=False)

        # 5) Generate section embeddings (outside the write transaction).
        section_embeddings: list[tuple[int, list[float], int, int, int, bool]] = []
        for i, sec in enumerate(offset_result):
            title_path = sec.get("title_path") or sec.get("title") or ""
            body = content[sec["start_offset"]:sec["end_offset"]]
            try:
                er = embedder.embed_text(prefix=title_path, body=body, max_body_chars=max_section_chars)
                if not er.embedding:
                    raise RuntimeError(
                        f"section {i}: {getattr(embedder, 'last_encode_error', None) or 'encode returned empty embedding'}"
                    )
                section_embeddings.append((i, er.embedding, int(er.truncated), er.original_tokens, er.used_tokens, True))
            except Exception as exc:
                if decision_kind == "split":
                    self._mark_split_failed(
                        mid, decision_content_hash, decision_memory_version,
                        decision_split_revision, decision_split_status,
                        "embedding", f"section {i}: {exc}",
                    )
                return self.db.state.response({"error": f"section embedding failed at {i}: {exc}"}, ok=False)

        # 6) Atomic publish: re-CAS inside the write transaction, then swap.
        try:
            with self.db.write_transaction() as conn:
                cur = conn.execute(
                    "SELECT status, content, version, split_status, split_revision FROM memories WHERE id = ?",
                    (mid,),
                ).fetchone()
                if cur is None:
                    raise ValueError("memory_changed")
                if cur["status"] != "active":
                    raise ValueError("memory_changed")
                if hashlib.sha256(str(cur["content"]).encode("utf-8")).hexdigest() != decision_content_hash:
                    raise ValueError("memory_changed")
                if int(cur["version"]) != int(decision_memory_version):
                    raise ValueError("memory_changed")
                if cur["split_status"] != decision_split_status:
                    raise ValueError("split_revision_conflict")
                if int(cur["split_revision"]) != int(decision_split_revision):
                    raise ValueError("split_revision_conflict")
                if MemoryDB._get_meta(conn, "state") != "ready":
                    raise ValueError("vec_space_changed")
                active_space = MemoryDB._get_meta(conn, "active_space_id")
                if active_space != embedder.embedding_space_id:
                    raise ValueError("vec_space_changed")

                MemoryDB._delete_sections_for_memory(conn, mid)
                for i, sec in enumerate(offset_result):
                    em = section_embeddings[i]
                    section_id = MemoryDB._insert_section(
                        conn, mid, i,
                        title=sec.get("title"),
                        title_path=sec.get("title_path"),
                        summary=sec.get("summary"),
                        anchor_text=sec.get("anchor_text"),
                        occurrence_index=sec.get("occurrence_index", 0),
                        start_offset=sec["start_offset"],
                        end_offset=sec["end_offset"],
                        provenance=provenance,
                        embedding_truncated=em[2],
                        embedding_original_tokens=em[3],
                        embedding_used_tokens=em[4],
                    )
                    MemoryDB._store_section_vec(conn, section_id, em[1])
                updated = conn.execute(
                    "UPDATE memories SET split_status = 'active', "
                    "split_revision = split_revision + 1 "
                    "WHERE id = ? AND split_revision = ?",
                    (mid, int(decision_split_revision)),
                )
                if updated.rowcount != 1:
                    raise ValueError("split_revision_conflict")
        except ValueError as e:
            return self.db.state.response({"error": str(e)}, ok=False)

        return self.db.state.response({
            "split_active": True,
            "memory_id": mid,
            "section_count": len(offset_result),
        })

    def _after_write_split(
        self, memory_id: int,
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]], list[str]]:
        """Run the post-write split decision (design §6.1).

        Shared by memory_write and content memory_edit. Returns
        ``(split_block, split_request_or_None, warnings)``.

          * Below threshold / vec not ready / already active → split.required
            is False (capability reported when vec is off).
          * Rule plan publishable → enqueue background publish via the unified
            helper (provenance='parser'); split_block reports mode='rules_async'.
          * No publishable plan → return a full split_request for the Agent;
            split_status stays NULL (NOT failed, NOT pending).
          * Rule publish genuinely fails → split_block reports failed; the
            original content is untouched.
        """
        warnings: list[str] = []
        threshold = self.settings.split_threshold
        max_sections = self.settings.max_sections
        max_section_chars = self.settings.max_section_chars

        mem = self.db.get_memory(memory_id)
        if mem is None or mem.get("status") != "active":
            return ({"required": False, "applied": False, "mode": None,
                     "status": None, "reason": None, "action_required": None,
                     "extra_llm_call_required": False}, None, warnings)
        content = mem.get("content") or ""
        if len(content) < threshold:
            return ({"required": False, "applied": False, "mode": None,
                     "status": None, "reason": None, "action_required": None,
                     "extra_llm_call_required": False}, None, warnings)

        vec_state = self.db.get_vec_index_state()
        if vec_state.get("state") != "ready":
            warnings.append("split_skipped_vec_not_ready")
            return ({"required": False, "applied": False, "mode": None,
                     "status": None, "reason": "vec_not_ready",
                     "action_required": None, "extra_llm_call_required": False,
                     "split_capability": {"available": False, "reason": "vec_not_ready"},
                     }, None, warnings)

        # Already actively split for this content version → nothing to do.
        if mem.get("split_status") == "active":
            return ({"required": False, "applied": False, "mode": None,
                     "status": "active", "reason": None, "action_required": None,
                     "extra_llm_call_required": False}, None, warnings)

        plan, reason = self._rule_plan_sections(content, max_sections, max_section_chars)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        snapshot = {
            "memory_id": memory_id,
            "content": content,
            "plan": plan,
            "content_hash": content_hash,
            "memory_version": int(mem.get("version") or 1),
            "split_status": mem.get("split_status"),
            "split_revision": int(mem.get("split_revision") or 0),
        }

        if plan is not None:
            self._split_worker.enqueue(memory_id, snapshot)
            return ({"required": True, "applied": False, "mode": "rules_async",
                     "status": mem.get("split_status"), "reason": None,
                     "action_required": None, "extra_llm_call_required": False,
                     "reindex_pending": True, "section_count_estimated": len(plan)},
                    None, warnings)

        # No publishable rule plan → hand a full split_request to the Agent.
        split_request = {
            **snapshot,
            "char_count": len(content),
            "reason": reason,
            "split_schema": {
                "sections": [{
                    "title": "str",
                    "summary": "str",
                    "anchor_text": "str (除第一段外必填)",
                    "occurrence_index": "int (0-based)",
                    "title_path": "str (可选)",
                }],
            },
        }
        return ({"required": True, "applied": False, "mode": "agent_semantic",
                 "status": None, "reason": reason,
                 "action_required": "memory_split", "extra_llm_call_required": True},
                split_request, warnings)

    def memory_split(
        self,
        memory_id: int,
        split_decision: Optional[str] = None,
        decision_content_hash: Optional[str] = None,
        decision_memory_version: Optional[int] = None,
        decision_split_status: Optional[str] = None,
        decision_split_revision: Optional[int] = None,
        sections: Optional[list[dict]] = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Section split: prepare (return content for LLM) or publish (validate + atomically write).

        v0.6.0 is single-batch: prepare returns the full content in one go.  For
        ultra-long documents that exceed an external LLM's context, the caller
        should pre-chunk before ``memory_write`` (split across multiple
        memories) rather than relying on a server-side batch protocol.
        """
        mid = int(memory_id)
        memory = self.db.get_memory(mid)
        if not memory:
            return self.db.state.response({"error": "memory not found"}, ok=False)

        content = memory.get("content") or ""
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        memory_version = int(memory.get("version") or 1)
        split_status = memory.get("split_status")
        split_revision = int(memory.get("split_revision") or 0)

        # ---- PREPARE ----
        if split_decision is None:
            # v0.8: capability is bound to vec readiness, not split_enabled.
            vec_state = self.db.get_vec_index_state()
            if vec_state.get("state") != "ready":
                return self.db.state.response({
                    "error": "vec index not ready",
                    "vec_index_state": vec_state,
                }, ok=False)
            if len(content) <= self.settings.split_threshold:
                return self.db.state.response({"error": "content below threshold, no need to split"})

            # v0.8: prepare returns the full snapshot for the Agent to continue.
            # No user-confirmation gate — this is an internal continuation /
            # repair protocol. Active records return allowed_decision='rebuild'.
            return self.db.state.response({
                "content": content,
                "content_hash": content_hash,
                "memory_version": memory_version,
                "split_status": split_status,
                "split_revision": split_revision,
                "char_count": len(content),
                "allowed_decision": "rebuild" if split_status == "active" else "split",
                "split_schema": {
                    "sections": [{
                        "title": "str",
                        "summary": "str",
                        "anchor_text": "str (除第一段外必填)",
                        "occurrence_index": "int (0-based)",
                        "title_path": "str (可选)",
                    }],
                },
            })

        # ---- DECLINE ----
        if split_decision == "decline":
            snapshot_error = self._split_snapshot_error(
                memory,
                decision_content_hash,
                decision_memory_version,
                decision_split_status,
                decision_split_revision,
                (None, "failed", "declined"),
            )
            if snapshot_error:
                return self.db.state.response({"error": snapshot_error}, ok=False)
            with self.db.write_transaction() as conn:
                cur = conn.execute(
                    "SELECT status, content, version, split_status, split_revision "
                    "FROM memories WHERE id = ?", (mid,)
                ).fetchone()
                current = dict(cur) if cur is not None else {}
                snapshot_error = self._split_snapshot_error(
                    current,
                    decision_content_hash,
                    decision_memory_version,
                    decision_split_status,
                    decision_split_revision,
                    (None, "failed", "declined"),
                )
                if snapshot_error:
                    return self.db.state.response({"error": snapshot_error}, ok=False)
                updated = conn.execute(
                    "UPDATE memories SET split_status = 'declined', "
                    "split_revision = split_revision + 1 "
                    "WHERE id = ? AND split_revision = ?",
                    (mid, int(decision_split_revision)),
                )
                if updated.rowcount != 1:
                    return self.db.state.response({"error": "split_revision_conflict"}, ok=False)
            return self.db.state.response({"declined": True, "memory_id": mid})

        # ---- PUBLISH (split or rebuild) ----
        if split_decision in ("split", "rebuild"):
            allowed_statuses: tuple[Optional[str], ...] = (
                ("active",)
                if split_decision == "rebuild"
                else (None, "failed", "declined")
            )
            snapshot_error = self._split_snapshot_error(
                memory,
                decision_content_hash,
                decision_memory_version,
                decision_split_status,
                decision_split_revision,
                allowed_statuses,
            )
            if snapshot_error:
                return self.db.state.response({"error": snapshot_error}, ok=False)
            if not sections:
                return self.db.state.response({"error": "sections required for publish"}, ok=False)

            # Delegate to the unified publish helper (v0.8.0). The Agent
            # continuation/repair path is always provenance="agent"; the rules
            # path (memory_write/edit) calls the same helper with "parser".
            return self._publish_sections(
                mid, content, sections,
                str(decision_content_hash), int(decision_memory_version),
                decision_split_status, int(decision_split_revision),
                decision_kind=split_decision,
                provenance="agent",
            )

        return self.db.state.response({"error": f"unknown split_decision: {split_decision}"}, ok=False)

    def _mark_split_failed(
        self, mid: int, content_hash: str, version: int, revision: int,
        expected_status: Optional[str], stage: str, message: str,
    ) -> None:
        """Mark split as failed using CAS (best-effort)."""
        try:
            with self.db.write_transaction() as conn:
                cur = conn.execute(
                    "SELECT status, content, version, split_status, split_revision, metadata "
                    "FROM memories WHERE id = ?",
                    (mid,),
                ).fetchone()
                if cur is None:
                    return
                if cur["status"] != "active":
                    return
                if hashlib.sha256(str(cur["content"]).encode("utf-8")).hexdigest() != content_hash:
                    return
                if int(cur["version"]) != version or int(cur["split_revision"]) != revision:
                    return
                if str(cur["split_status"]) != str(expected_status):
                    return
                # Merge metadata
                try:
                    meta = json.loads(cur["metadata"] or "{}")
                except (TypeError, json.JSONDecodeError):
                    meta = {}
                if not isinstance(meta, dict):
                    meta = {}
                split_meta = meta.get("_split", {})
                split_meta["last_split_error"] = {"stage": stage, "message": message}
                meta["_split"] = split_meta
                conn.execute(
                    "UPDATE memories SET split_status = 'failed', "
                    "split_revision = split_revision + 1, "
                    "metadata = ? WHERE id = ?",
                    (json.dumps(meta, ensure_ascii=False), mid),
                )
        except Exception:
            pass

    def memory_rebuild_embeddings(
        self,
        memory_ids: Optional[list[int]] = None,
        dry_run: bool = True,
        batch_size: Optional[int] = 50,
        **_: Any,
    ) -> dict[str, Any]:
        """Rebuild embeddings after model switch or for repair."""
        vec_state = self.db.get_vec_index_state()
        state = vec_state.get("state", "unmanaged")

        if state == "unmanaged":
            return self.db.state.response({"error": "unmanaged: no managed embedder"}, ok=False)

        embedder, _ = self._ensure_embedder()
        if embedder is None:
            return self.db.state.response({"error": "embedder unavailable"}, ok=False)

        # Determine target memories
        migration_mode = state in ("mismatch", "failed")
        migration_cursor = vec_state.get("migration_cursor")
        cursor_value = int(migration_cursor) if migration_cursor is not None else -1
        if migration_mode:
            if vec_state.get("target_space_id") != embedder.embedding_space_id:
                return self.db.state.response({
                    "error": "current embedder does not match migration target",
                    "target_space_id": vec_state.get("target_space_id"),
                    "current_space_id": embedder.embedding_space_id,
                }, ok=False)
            # Migration mode: continue after the persisted contiguous cursor.
            with self.db.connection() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT m.id AS id FROM memories m "
                    "LEFT JOIN memories_vec v ON v.id = m.id "
                    "LEFT JOIN memory_sections s ON s.memory_id = m.id "
                    "WHERE (v.id IS NOT NULL OR s.id IS NOT NULL) AND m.id > ? "
                    "ORDER BY m.id",
                    (cursor_value,),
                ).fetchall()
                target_ids = [int(r["id"]) for r in rows]
        elif state == "ready":
            if not memory_ids:
                return self.db.state.response({
                    "error": "ready state: specify memory_ids for local repair"
                }, ok=False)
            target_ids = [int(mid) for mid in memory_ids]
        else:
            return self.db.state.response({"error": f"unexpected state: {state}"}, ok=False)

        if batch_size is not None:
            batch_size = max(1, int(batch_size))
            target_ids = target_ids[:batch_size]

        if dry_run:
            return self.db.state.response({
                "dry_run": True,
                "target_memory_ids": target_ids,
                "target_count": len(target_ids),
                "current_space_id": embedder.embedding_space_id,
                "active_space_id": vec_state.get("active_space_id"),
                "global_state": state,
            })

        # Execute rebuild
        succeeded = 0
        failed = 0
        processed = 0
        errors: list[dict] = []
        max_section_chars = self.settings.max_section_chars

        for mid in target_ids:
            processed += 1
            try:
                with self.db.connection() as conn:
                    mem = MemoryDB._fetch_memory(conn, mid)
                    if mem is None or mem.get("status") == "deleted":
                        # Cleanup
                        with self.db.write_transaction() as wconn:
                            MemoryDB._delete_sections_for_memory(wconn, mid)
                            wconn.execute("DELETE FROM memories_vec WHERE id = ?", (mid,))
                            if migration_mode:
                                MemoryDB._set_meta(wconn, "migration_cursor", str(mid))
                        succeeded += 1
                        continue

                    # Memory embedding
                    er = embedder.embed_text(
                        prefix=mem.get("subject") or "",
                        body=mem.get("content") or "",
                    )
                    if not er.embedding:
                        raise RuntimeError(
                            f"memory {mid}: {getattr(embedder, 'last_encode_error', None) or 'encode returned empty embedding'}"
                        )

                    # Section embeddings
                    sections = MemoryDB._get_sections(conn, mid)
                    sec_embeddings = []
                    content = mem.get("content") or ""
                    for sec in sections:
                        title_path = sec.get("title_path") or sec.get("title") or ""
                        body = content[sec["start_offset"]:sec["end_offset"]]
                        sec_er = embedder.embed_text(prefix=title_path, body=body, max_body_chars=max_section_chars)
                        if not sec_er.embedding:
                            raise RuntimeError(
                                f"section {sec['id']}: {getattr(embedder, 'last_encode_error', None) or 'encode returned empty embedding'}"
                            )
                        sec_embeddings.append((sec["id"], sec_er))

                    # Write
                    with self.db.write_transaction() as wconn:
                        wconn.execute("DELETE FROM memories_vec WHERE id = ?", (mid,))
                        wconn.execute(
                            "INSERT INTO memories_vec(id, parent_status, embedding) VALUES (?, ?, ?)",
                            (mid, mem["status"] or "deleted", json.dumps(er.embedding)),
                        )
                        for sid, ser in sec_embeddings:
                            wconn.execute("DELETE FROM memory_sections_vec WHERE id = ?", (sid,))
                            # v0.9.4: look up parent status via memory_sections JOIN (N17: COALESCE for orphan)
                            parent_row = wconn.execute(
                                "SELECT COALESCE(m.status, 'deleted') AS status "
                                "FROM memory_sections s "
                                "LEFT JOIN memories m ON m.id = s.memory_id "
                                "WHERE s.id = ?", (sid,)
                            ).fetchone()
                            parent_status = parent_row["status"] if parent_row else "deleted"
                            wconn.execute(
                                "INSERT INTO memory_sections_vec(id, parent_status, embedding) VALUES (?, ?, ?)",
                                (sid, parent_status, json.dumps(ser.embedding)),
                            )
                            wconn.execute(
                                "UPDATE memory_sections SET embedding_truncated = ?, "
                                "embedding_original_tokens = ?, embedding_used_tokens = ? "
                                "WHERE id = ?",
                                (int(ser.truncated), ser.original_tokens, ser.used_tokens, sid),
                            )
                        if migration_mode:
                            MemoryDB._set_meta(wconn, "migration_cursor", str(mid))
                    succeeded += 1
            except Exception as exc:
                failed += 1
                errors.append({"memory_id": mid, "error": str(exc)})
                if migration_mode:
                    try:
                        with self.db.write_transaction() as conn:
                            MemoryDB._set_meta(conn, "state", "failed")
                            MemoryDB._set_meta(conn, "last_error", f"memory_id={mid}: {exc}")
                    except Exception:
                        pass
                    break  # preserve a contiguous cursor for the next resume

        # Update vec state only after checking for targets beyond the cursor
        # produced by this batch.  Re-querying without a cursor would select
        # the vectors just rebuilt and make migration impossible to finish.
        if migration_mode and not errors:
            try:
                with self.db.write_transaction() as conn:
                    current_cursor_raw = MemoryDB._get_meta(conn, "migration_cursor")
                    current_cursor = int(current_cursor_raw) if current_cursor_raw is not None else -1
                    remaining = conn.execute(
                        "SELECT 1 FROM memories m "
                        "LEFT JOIN memories_vec v ON v.id = m.id "
                        "LEFT JOIN memory_sections s ON s.memory_id = m.id "
                        "WHERE (v.id IS NOT NULL OR s.id IS NOT NULL) AND m.id > ? "
                        "LIMIT 1",
                        (current_cursor,),
                    ).fetchone()
                    if remaining is None:
                        MemoryDB._set_meta(conn, "state", "ready")
                        MemoryDB._set_meta(conn, "active_space_id", embedder.embedding_space_id)
                        for key in (
                            "target_space_id", "migration_cursor", "migration_epoch",
                            "migration_lease_owner", "migration_lease_expires_at",
                            "last_error",
                        ):
                            MemoryDB._delete_meta(conn, key)
            except Exception:
                pass

        return self.db.state.response({
            "processed": processed,
            "succeeded": succeeded,
            "failed": failed,
            "errors": errors,
            "global_state": self.db.get_vec_index_state().get("state"),
        })
