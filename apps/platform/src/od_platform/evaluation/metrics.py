#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : metrics.py
# @Project   : ODPlatform
# @Function  : 指标整理（纯函数, 不 import ultralytics, 可离线单测）

"""把框架吐出来的指标**整理成人能用的口径**。

★ 为什么要有这一层

    ``ultralytics`` 给的是"每类 AP 的数组 + 一堆标量", 而"看指标"真正需要的是:

        1. **每类一行**（AP50 / AP50-95 / P / R）—— 且**没有真值的类必须标成"无真值"**,
           不能顺手记成 0: 把它记成 0 会让人以为"这类识别得很差", 而事实是"这类没参与评测"。
        2. **多套平均口径** —— mAP 是"按类平均"的, 一个框数少、又难的小类
           （足球比赛里的 ball: 327 框 vs player 7427 框）会把平均值明显拖下来。
           所以除了全类平均, 还要能出"不含某些类"的平均, 让两件事分开说清。
        3. **置信度扫描** —— 部署时要选一个 conf 阈值。选它的依据不是"置信度高",
           而是"在满足精确率要求的前提下召回率最大"。

    ★ 本模块刻意**不 import ultralytics**: 输入是普通的 list / float,
      于是判据可以完全离线单测, 不需要权重、不需要 GPU、不需要数据。

★ 关于"均值"的口径差异（很容易被问倒, 先说清）

    ``map50`` 的分母是**有真值的类**（框架的约定, 本模块的 ``mean_ap`` 也用同一口径,
    所以不排除任何类时两者应当相等 —— 有一条测试专门守这个一致性）。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# conf 扫描里 F1 的分母保护: P+R 太小时不给 F1, 免得报出一个由噪声放大的数
F1_MIN_SUM: float = 1e-9

NO_GROUND_TRUTH = "—"          # 报告里表示"该类没有真值, 未参与评测"


def f1_of(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    """调和平均 F1; 任一为空或无意义时返回 None（不返回 0 —— 0 是"算出来很差"）。"""
    if precision is None or recall is None:
        return None
    total = float(precision) + float(recall)
    if total <= F1_MIN_SUM:
        return None
    return 2.0 * float(precision) * float(recall) / total


def per_class_rows(names: Sequence[str], *, ap_class_index: Iterable[int],
                   ap50: Iterable[float], ap: Iterable[float],
                   precision: Iterable[float], recall: Iterable[float],
                   instances: Sequence[int]) -> List[Dict[str, Any]]:
    """把框架的"按位置"数组摊成"每类一行"。

    ★ 关键点: ``ap_class_index`` 只列出**有真值的类**。没有真值的类在这里被显式标成
      ``evaluated=False`` + ``NO_GROUND_TRUTH``, 而不是补 0 —— 补 0 会把
      "没评测"说成"评测得很差"。
    """
    names = list(names)
    indices = [int(i) for i in ap_class_index]
    arrays = {
        "AP50": [float(v) for v in ap50],
        "AP50_95": [float(v) for v in ap],
        "precision": [float(v) for v in precision],
        "recall": [float(v) for v in recall],
    }
    by_index: Dict[int, Dict[str, float]] = {}
    for position, class_index in enumerate(indices):
        row: Dict[str, float] = {}
        for key, values in arrays.items():
            if position < len(values):
                row[key] = values[position]
        by_index[class_index] = row

    rows: List[Dict[str, Any]] = []
    for class_index, name in enumerate(names):
        count = int(instances[class_index]) if class_index < len(instances) else 0
        if class_index in by_index:
            row = by_index[class_index]
            rows.append({
                "class_id": class_index, "class_name": name,
                "evaluated": True, "instances": count,
                "AP50": row.get("AP50"), "AP50_95": row.get("AP50_95"),
                "precision": row.get("precision"), "recall": row.get("recall"),
            })
        else:
            rows.append({
                "class_id": class_index, "class_name": name,
                "evaluated": False, "instances": count,
                "AP50": None, "AP50_95": None, "precision": None, "recall": None,
                "note": f"没有真值（{NO_GROUND_TRUTH}）, 未参与评测",
            })
    return rows


def mean_ap(rows: Sequence[Dict[str, Any]], *, exclude: Iterable[str] = (),
            include: Iterable[str] = ()) -> Dict[str, Any]:
    """按类平均 mAP —— 分母是"既在范围内、又有真值"的类。

    Args:
        rows: :func:`per_class_rows` 的输出
        exclude: 要排除的类名（例如小目标 ball）
        include: 只算这些类名（与 exclude 互斥; 同时给会报错, 免得口径含糊）

    ★ 没有真值的类不参与平均 —— 这是 mAP 的通行约定。把"没评测的类"塞进分母,
      等于用一个不存在的结果去拉低平均值。
    """
    exclude_set, include_set = set(exclude), set(include)
    if exclude_set and include_set:
        raise ValueError("exclude 与 include 不能同时给 —— 口径必须唯一")
    selected = []
    for row in rows:
        if not row.get("evaluated"):
            continue
        name = str(row.get("class_name"))
        if exclude_set and name in exclude_set:
            continue
        if include_set and name not in include_set:
            continue
        selected.append(row)

    label = "全类平均"
    if exclude_set:
        label = f"排除 {', '.join(sorted(exclude_set))} 后的平均"
    elif include_set:
        label = f"只算 {', '.join(sorted(include_set))} 的平均"

    result: Dict[str, Any] = {
        "label": label,
        "n_classes": len(selected),
        "classes": [row["class_name"] for row in selected],
    }
    for key in ("AP50", "AP50_95", "precision", "recall"):
        values = [float(row[key]) for row in selected if row.get(key) is not None]
        result[key] = (sum(values) / len(values)) if values else None
    return result


def conf_sweep_rows(thresholds: Sequence[float],
                    observations: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把"每个阈值跑一次的观测"整理成扫描表。

    Args:
        thresholds: 阈值列表（与 observations 一一对应）
        observations: 每个元素形如 ``{"precision":.., "recall":..}``

    ★ 为什么是"逐个阈值各跑一次"而不是从 P-R 曲线推:
      框架给的 PR 曲线的横轴是**召回率**, 不是置信度 —— 没有"conf→R"的对应关系,
      推不出来。所以扫描的代价就是 N 倍 val 时间, 这一点必须如实写在文档里,
      并允许 ``--no-conf-sweep`` 关掉。
    """
    rows: List[Dict[str, Any]] = []
    # strict=True: 阈值与观测必须一一对应。长度对不上是**调用方的编程错误**,
    # 静默截断会产出一张少了几行的扫描表, 那比当场报错更难查。
    for threshold, observed in zip(thresholds, observations, strict=True):
        precision = observed.get("precision")
        recall = observed.get("recall")
        rows.append({
            "conf": float(threshold),
            "precision": None if precision is None else float(precision),
            "recall": None if recall is None else float(recall),
            "f1": f1_of(precision, recall),
        })
    return rows


def recommend_threshold(rows: Sequence[Dict[str, Any]], *, target_precision: float,
                        decimals: int = 4) -> Dict[str, Any]:
    """推荐一个部署用的 conf 阈值。

    规则（说清楚, 免得"推荐"变成玄学）:

        1. 在 **精度 ≥ target_precision** 的阈值里, 取**召回率最大**的那个
           —— 也就是"精确率达标的前提下, 尽量多抓"。这比"取最大 F1"更贴合
           "宁可少报错、不可漏太多"的实际取向;
        2. 若并列（同一个 R）, 取**精确率更高**的; 再并列则取**阈值更低**的;
        3. **如果没有任何阈值达到精度要求**, 不硬推荐 —— 返回
           ``status="目标精度不可达"``, 并给出 F1 最高的那个阈值作为参考。

    Returns:
        含 status / conf / precision / recall / f1 / reason 的字典
    """
    usable = [row for row in rows
              if row.get("precision") is not None and row.get("recall") is not None]
    if not usable:
        return {"status": "无可用观测", "conf": None, "precision": None,
                "recall": None, "f1": None,
                "reason": "置信度扫描没有产出可用的精确率/召回率"}

    qualified = [row for row in usable if float(row["precision"]) >= target_precision]
    if qualified:
        best = max(qualified, key=lambda r: (float(r["recall"]),
                                            float(r["precision"]),
                                            -float(r["conf"])))
        return {
            "status": "达到目标精度",
            "conf": round(float(best["conf"]), decimals),
            "precision": round(float(best["precision"]), decimals),
            "recall": round(float(best["recall"]), decimals),
            "f1": None if best.get("f1") is None else round(float(best["f1"]), decimals),
            "reason": (f"在精度 ≥ {target_precision} 的阈值里取召回率最大者"
                       f"（扫描了 {len(usable)} 个阈值）"),
        }

    best_f1 = max(usable, key=lambda r: (float(r.get("f1") or 0.0),
                                         float(r["precision"])))
    return {
        "status": "目标精度不可达",
        "conf": round(float(best_f1["conf"]), decimals),
        "precision": round(float(best_f1["precision"]), decimals),
        "recall": round(float(best_f1["recall"]), decimals),
        "f1": None if best_f1.get("f1") is None else round(float(best_f1["f1"]), decimals),
        "reason": (f"扫描范围内没有任何阈值的精度达到 {target_precision}"
                   f"（最高 {round(max(float(r['precision']) for r in usable), decimals)}）; "
                   "上面给的是 F1 最高的阈值, 仅供参考, 不构成「达标」结论"),
    }


def pr_curve_payload(curves: Sequence[Tuple[Any, Any, Any, Any]],
                     names: Sequence[str], *, points: int) -> List[Dict[str, Any]]:
    """把框架的 P-R 曲线降采样成报告里的点集。

    框架给的每条曲线是 1000 个点（横轴召回率 0→1、纵轴每类精度）, 直接塞进 json
    会又大又难读。这里按 ``points`` 均匀采样, 并带上类名。
    """
    payload: List[Dict[str, Any]] = []
    for position, entry in enumerate(curves):
        if len(entry) < 2:
            continue
        x, y = entry[0], entry[1]
        try:
            import numpy as np
            xs = np.asarray(x, dtype=float).ravel()
            ys = np.asarray(y, dtype=float)
        except Exception as exc:                     # noqa: BLE001 —— 曲线是附加信息, 不该拖垮报告
            logger.debug("P-R 曲线不可解析（第 %d 条）: %s", position, exc)
            continue
        if xs.size == 0 or ys.size == 0:
            continue
        if ys.ndim == 1:
            ys = ys.reshape(1, -1)
        step = max(1, xs.size // max(points, 1))
        indices = list(range(0, xs.size, step))[:points]
        series: Dict[str, List[float]] = {}
        for row in range(min(ys.shape[0], len(names))):
            series[str(names[row])] = [round(float(ys[row][i]), 6) for i in indices
                                       if i < ys.shape[1]]
        payload.append({
            "recall": [round(float(xs[i]), 6) for i in indices],
            "precision_by_class": series,
        })
    return payload


# ---------------------------------------------------------------- 如实警告

def build_warnings(view: Any, rows: Sequence[Dict[str, Any]], settings: Any) -> List[str]:
    """生成"必须说在前面"的警告 —— 这些是答辩会被追问的点, 不能等别人发现。

    触发条件全部来自 EvalSettings, 本函数不写死任何数字。
    """
    warnings: List[str] = []

    n_images = int(getattr(view, "n_images", 0) or 0)
    if n_images < settings.min_images_for_reliable_map:
        warnings.append(
            f"验证集只有 {n_images} 张图（低于 {settings.min_images_for_reliable_map} 张）: "
            "在这点样本上 mAP 的统计抖动可能大于结论本身, 报数时建议给区间或说明稳定性, "
            "不要给出小数点后一位当精确值")

    evaluated = [row for row in rows if row.get("evaluated")]
    weak = [row for row in evaluated
            if int(row.get("instances") or 0) < settings.min_instances_for_reliable_ap]
    if weak:
        detail = ", ".join(f"{row['class_name']}={row['instances']} 框" for row in weak)
        warnings.append(
            f"以下类别的真值框少于 {settings.min_instances_for_reliable_ap} 个（{detail}）: "
            "它们的每类 AP 不稳定, 且会明显影响按类平均的 mAP")

    counts = [int(row.get("instances") or 0) for row in evaluated if row.get("instances")]
    if counts:
        smallest, largest = min(counts), max(counts)
        if smallest > 0:
            ratio = largest / smallest
            if ratio > settings.imbalance_ratio_warn:
                warnings.append(
                    f"类别严重不平衡: 最多类 {largest} 框 / 最少类 {smallest} 框 = "
                    f"{ratio:.1f} 倍（阈值 {settings.imbalance_ratio_warn}）—— 按类平均的 mAP "
                    "会被框少的那几类拖低, 建议同时报「排除小类后的平均」这套口径")

    missing = [row for row in rows if not row.get("evaluated")]
    if missing:
        warnings.append(
            f"有 {len(missing)} 个类没有真值（{', '.join(str(r['class_name']) for r in missing)}）: "
            "它们没有参与评测, 报告里标为「无真值」而不是 0 分")

    if getattr(view, "limited_to", 0):
        warnings.append(
            f"本次只评估了前 {view.limited_to} 张图（--limit）: 指标**不代表整个 split**, "
            "仅可用于验证链路是否通")

    copied = int(getattr(view, "copied_files", 0) or 0)
    if copied:
        warnings.append(
            f"数据集视图里有 {copied} 个文件未能硬链接（跨卷）, 已退化为复制 —— "
            "这些是真实占用的磁盘")

    return warnings
