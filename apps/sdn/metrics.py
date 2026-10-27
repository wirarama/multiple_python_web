"""Perhitungan metrik evaluasi jalur: Delay, Hop Count, Throughput (simulasi), Packet Loss (simulasi).

Throughput dan Packet Loss ditandai '(simulasi)' karena tidak diukur dari trafik nyata,
melainkan diestimasi dari atribut link (bandwidth, loss dasar) menggunakan model
sederhana yang merepresentasikan efek jumlah hop dan kongesti pada jaringan SDN.
"""
import random
from graph_utils import path_edges


def compute_path_metrics(G, path, seed=None, congestion=0.0):
    """Hitung metrik untuk sebuah path (list of node).

    congestion: faktor beban jaringan tambahan [0..1] yang mensimulasikan trafik
                bersaing pada link yang sama (0 = jaringan sepi, 1 = sangat padat).
    """
    if not path or len(path) < 2:
        return {
            "valid": False,
            "path": path or [],
            "hop_count": 0,
            "delay_ms": None,
            "throughput_mbps": None,
            "packet_loss_pct": None,
        }

    rng = random.Random(seed)
    edges = path_edges(path)

    # --- Delay: jumlah delay tiap link sepanjang jalur (ms) ---
    total_delay = sum(G[u][v]["delay"] for u, v in edges)

    # --- Hop Count: jumlah lompatan (edge) sepanjang jalur ---
    hop_count = len(edges)

    # --- Throughput (simulasi): dibatasi oleh link dengan bandwidth terkecil
    #     (bottleneck), dikurangi efek jumlah hop dan kongesti simulasi ---
    min_bandwidth = min(G[u][v]["bandwidth"] for u, v in edges)
    hop_penalty = max(0.0, 1 - 0.03 * (hop_count - 1))          # tiap hop tambahan -3%
    congestion_penalty = max(0.0, 1 - congestion)
    jitter = rng.uniform(0.92, 1.0)                              # variasi acak simulasi trafik
    throughput = min_bandwidth * hop_penalty * congestion_penalty * jitter
    throughput = max(0.0, round(throughput, 2))

    # --- Packet Loss (simulasi): gabungan loss dasar tiap link (1 - prod(1-loss_i))
    #     ditambah kontribusi kongesti dan variasi acak kecil ---
    prob_success = 1.0
    for u, v in edges:
        prob_success *= (1 - G[u][v]["loss"])
    base_loss = 1 - prob_success
    congestion_loss = congestion * 0.02 * hop_count
    noise = rng.uniform(0.0, 0.002)
    packet_loss = min(1.0, base_loss + congestion_loss + noise)

    return {
        "valid": True,
        "path": path,
        "hop_count": hop_count,
        "delay_ms": round(total_delay, 3),
        "throughput_mbps": throughput,
        "packet_loss_pct": round(packet_loss * 100, 3),
    }


def compute_mst_metrics(G, mst_edges, seed=None, congestion=0.0):
    """Metrik khusus untuk hasil Minimum Spanning Tree (bukan satu jalur, tapi pohon)."""
    rng = random.Random(seed)
    if not mst_edges:
        return {"valid": False}

    total_weight = sum(G[u][v]["delay"] for u, v in mst_edges)
    avg_bandwidth = sum(G[u][v]["bandwidth"] for u, v in mst_edges) / len(mst_edges)
    jitter = rng.uniform(0.92, 1.0)
    throughput = avg_bandwidth * max(0.0, 1 - congestion) * jitter

    prob_success = 1.0
    for u, v in mst_edges:
        prob_success *= (1 - G[u][v]["loss"])
    packet_loss = min(1.0, (1 - prob_success) + congestion * 0.01 * len(mst_edges))

    return {
        "valid": True,
        "edges": [[u, v] for u, v in mst_edges],
        "hop_count": len(mst_edges),
        "delay_ms": round(total_weight, 3),
        "throughput_mbps": round(throughput, 2),
        "packet_loss_pct": round(packet_loss * 100, 3),
    }
