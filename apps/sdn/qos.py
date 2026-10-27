"""Klasifikasi Quality of Service (QoS) dari hasil metrik routing.

Skor QoS (0-100) dihitung dari kombinasi tertimbang Delay, Packet Loss, dan
Throughput, dinormalisasi terhadap kondisi terburuk/terbaik yang mungkin pada
topologi aktif (bukan standar ITU-T formal, melainkan skema relatif untuk
kebutuhan pembelajaran/simulasi).
"""

QOS_WEIGHTS = {"delay": 0.4, "loss": 0.3, "throughput": 0.3}

QOS_CLASSES = [
    (85, "Excellent"),
    (70, "Good"),
    (50, "Fair"),
    (0, "Poor"),
]


def _clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, v))


def topology_reference(G):
    """Hitung nilai acuan (referensi) dari topologi aktif untuk normalisasi skor QoS."""
    delays = [d["delay"] for _, _, d in G.edges(data=True)] or [1.0]
    bandwidths = [d["bandwidth"] for _, _, d in G.edges(data=True)] or [1.0]
    n_nodes = max(G.number_of_nodes(), 2)
    return {
        "max_delay": sum(sorted(delays, reverse=True)[: n_nodes - 1]) or max(delays),
        "max_bandwidth": max(bandwidths),
    }


def classify(metrics, ref):
    """metrics: dict hasil compute_path_metrics/compute_mst_metrics (harus valid=True)."""
    if not metrics or not metrics.get("valid"):
        return {"score": None, "class": None}

    delay = metrics.get("delay_ms") or 0
    loss = metrics.get("packet_loss_pct") or 0
    throughput = metrics.get("throughput_mbps") or 0

    delay_score = _clamp(100 - (delay / max(ref["max_delay"], 1e-6)) * 100)
    loss_score = _clamp(100 - loss * 8)          # tiap 1% loss memotong 8 poin
    throughput_score = _clamp((throughput / max(ref["max_bandwidth"], 1e-6)) * 100)

    overall = (
        QOS_WEIGHTS["delay"] * delay_score
        + QOS_WEIGHTS["loss"] * loss_score
        + QOS_WEIGHTS["throughput"] * throughput_score
    )
    overall = round(overall, 1)

    qos_class = next(label for threshold, label in QOS_CLASSES if overall >= threshold)
    return {
        "score": overall,
        "class": qos_class,
        "components": {
            "delay_score": round(delay_score, 1),
            "loss_score": round(loss_score, 1),
            "throughput_score": round(throughput_score, 1),
        },
    }
