#!/usr/bin/env python3
"""Generate eval/fixtures/conflict/pairs_claims.jsonl (gate-v2 G7b).

B-class (both sides carry claims, deterministic claims×claims channel) in
four shapes ×5 pairs; C-class (own carries claims, peer prose only,
claims×sentence KNN channel) in two shapes ×10 pairs. Grounding contract is
asserted at generation time: every claims.value MUST be a literal slice of
its own side's content.
"""
import json

pairs = []


def env(subject, content, claims=None):
    e = {"subject": subject, "content": content, "tags": ["claims-eval"]}
    if claims:
        e["claims"] = claims
    return e


def claim(attr, value):
    return {"attr": attr, "value": value}


# ── B 类：claims×claims（确定性）——真冲突 / 换算相等 / 版本豁免 / 不同属性 ──
for i in range(1, 6):
    pairs.append({
        "pair_id": f"claims-b-true-{i:02d}", "label": "true_conflict", "channel": "B",
        "left": env(f"网关配置甲{i}", f"服务{i}的网关超时为 5 秒，超时后触发熔断。",
                    [claim("网关超时", "网关超时为 5 秒")]),
        "right": env(f"网关配置乙{i}", f"服务{i}的网关超时为 3 秒，超时后触发熔断。",
                     [claim("网关超时", "网关超时为 3 秒")]),
    })

for i in range(1, 6):
    pairs.append({
        "pair_id": f"claims-b-equiv-{i:02d}", "label": "governed_negative", "channel": "B",
        "left": env(f"压测配置甲{i}", f"服务{i}的采样窗口为 500ms，滚动采集。",
                    [claim("采样窗口", "采样窗口为 500ms")]),
        "right": env(f"压测配置乙{i}", f"服务{i}的采样窗口为 0.5秒，滚动采集。",
                     [claim("采样窗口", "采样窗口为 0.5秒")]),
    })

for i in range(1, 6):
    pairs.append({
        "pair_id": f"claims-b-ver-{i:02d}", "label": "governed_negative", "channel": "B",
        "left": env(f"发布记录甲{i}", f"组件{i}的版本为 0.2.1，含修复补丁。",
                    [claim("版本", "版本为 0.2.1")]),
        "right": env(f"发布记录乙{i}", f"组件{i}的版本为 0.9.6，含修复补丁。",
                     [claim("版本", "版本为 0.9.6")]),
    })

for i in range(1, 6):
    pairs.append({
        "pair_id": f"claims-b-attr-{i:02d}", "label": "governed_negative", "channel": "B",
        "left": env(f"接口配置甲{i}", f"服务{i}的接口超时为 5 秒。",
                    [claim("接口超时", "接口超时为 5 秒")]),
        "right": env(f"接口配置乙{i}", f"服务{i}的重试次数为 5 次。",
                     [claim("重试次数", "重试次数为 5 次")]),
    })

# ── C 类：claims×句子（attr 向量 KNN 捞对方句子）——真冲突 / 无关负例 ──
for i in range(1, 11):
    pairs.append({
        "pair_id": f"claims-c-true-{i:02d}", "label": "true_conflict", "channel": "C",
        "left": env(f"上传规范甲{i}", f"项目{i}的上传方式为 ssh 直传内网机。",
                    [claim("上传方式", "上传方式为 ssh 直传内网机")]),
        "right": env(f"上传规范乙{i}", f"项目{i}的发布使用 https 拉取公网工件。"),
    })

for i in range(1, 11):
    pairs.append({
        "pair_id": f"claims-c-neg-{i:02d}", "label": "governed_negative", "channel": "C",
        "left": env(f"文档规范甲{i}", f"项目{i}的文档存放于 docs 目录。",
                    [claim("文档目录", "文档存放于 docs 目录")]),
        "right": env(f"会议纪要乙{i}", f"项目{i}的周会在周四下午三点召开。"),
    })

# grounding 自检（方案 G7b：value 必须是正文子串）
for p in pairs:
    for side in ("left", "right"):
        content = p[side]["content"]
        for c in p[side].get("claims") or []:
            assert c["value"] in content, (p["pair_id"], c["value"])

OUT = "eval/fixtures/conflict/pairs_claims.jsonl"
with open(OUT, "w", encoding="utf-8") as fh:
    for p in pairs:
        fh.write(json.dumps(p, ensure_ascii=False) + "\n")
print("wrote", len(pairs), "pairs to", OUT)
