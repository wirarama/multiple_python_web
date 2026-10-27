"""
Simulator Modul Deteksi Intrusi (Intrusion Detection Simulation) pada topologi
server & client SDN.

Membangkitkan skenario client-server (server sebagai target, client berperan
sebagai pengguna umum/normal atau penyerang, dengan proporsi yang dapat
diatur), lalu mensimulasikan traffic/flow acak berdasarkan jenis intrusion
yang dicentang di UI beserta potensi/intensitasnya:

    - DoS & DDoS                                            (dos_ddos)
    - Reconnaissance / Probing                               (probing)
    - Eksploitasi Akses Masuk -- Remote to Local (R2L)        (r2l)
    - Peningkatan Hak Akses Ilegal -- User to Root (U2R)      (u2r)

Tiap flow diberi label kelas sebenarnya (ground truth) lalu dievaluasi oleh
detector berbasis aturan sederhana (bukan model ML sungguhan) yang peluang
mendeteksinya naik seiring severity/potensi serangan -- untuk keperluan
demonstrasi/praktikum SDN & keamanan jaringan, dengan gaya yang sama seperti
modul simulasi lain pada proyek ini (topology_generator.py, metrics.py,
algorithms/load_balancing.py): bukan pengukuran trafik nyata, melainkan data
sintetis yang parameternya dapat diatur dari UI.
"""
import random

# --------------------------------------------------------------- registry --
# Fitur trafik gaya NSL-KDD/KDD-Cup: tiap flow membawa satu vektor fitur penuh
# (lihat _baseline_features), namun tiap jenis intrusion hanya "menonjol" pada
# subset fitur tertentu -- persis seperti dataset intrusion detection nyata,
# di mana klasifikasinya bertumpu pada fitur-fitur kunci berikut (KEY_FEATURES).
ATTACK_TYPES = {
    "dos_ddos": {
        "label": "DoS & DDoS",
        "description": (
            "Denial of Service / Distributed Denial of Service -- banjir "
            "request/paket ke server target untuk menghabiskan resource "
            "(bandwidth, koneksi, CPU) hingga layanan tidak dapat diakses. "
            "Mengandalkan volume trafik, sehingga bobot deteksi tertinggi "
            "ada pada fitur trafik: count, srv_count, serror_rate, "
            "rerror_rate, src_bytes, dan wrong_fragment (Smurf/Ping of Death)."
        ),
    },
    "probing": {
        "label": "Reconnaissance / Probing",
        "description": (
            "Pemindaian port/service (scanning) untuk memetakan celah "
            "keamanan pada server target sebelum melancarkan serangan lanjutan. "
            "Penyerang menyapu banyak target/layanan, bukan membanjiri satu "
            "layanan, sehingga prediktor terkuatnya adalah fitur trafik "
            "berbasis host: dst_host_diff_srv_rate, "
            "dst_host_same_src_port_rate, serta anomali protocol_type/service "
            "(mis. lonjakan ICMP pada Ping Sweep)."
        ),
    },
    "r2l": {
        "label": "Eksploitasi Akses Masuk (Remote to Local / R2L)",
        "description": (
            "Percobaan mendapatkan akses lokal dari jaringan luar tanpa "
            "kredensial sah, mis. brute-force login atau eksploitasi service "
            "(SSH/FTP/RDP/Telnet). Bentuknya eksploitasi berbasis payload pada "
            "satu koneksi, bukan banjir trafik, sehingga deteksi bertumpu pada "
            "fitur konten: num_failed_logins, is_guest_login, dan duration "
            "(sesi R2L yang gagal cenderung sangat singkat)."
        ),
    },
    "u2r": {
        "label": "Peningkatan Hak Akses Ilegal (User to Root / U2R)",
        "description": (
            "Eskalasi hak akses dari user biasa menjadi root/administrator "
            "pada host yang sudah berhasil diakses (lanjutan dari R2L). Paling "
            "sulit diklasifikasikan karena hanya butuh 1-3 koneksi kecil, "
            "sehingga deteksi eksklusif mengandalkan fitur konten: root_shell, "
            "su_attempted, num_file_creations, dan num_root."
        ),
    },
}
ATTACK_ORDER = ["dos_ddos", "probing", "r2l", "u2r"]
DEFAULT_POTENSI = {k: 0.5 for k in ATTACK_ORDER}

# Fitur kunci (paling berbobot bagi model klasifikasi) per jenis intrusion --
# dipakai untuk menghitung severity tiap flow dan untuk menampilkan rata-rata
# fitur kunci per label pada ringkasan hasil (lihat summarize_flows()).
KEY_FEATURES = {
    "dos_ddos": ["count", "srv_count", "serror_rate", "rerror_rate", "src_bytes", "wrong_fragment"],
    "probing": ["dst_host_diff_srv_rate", "dst_host_same_src_port_rate"],
    "r2l": ["num_failed_logins", "is_guest_login", "duration"],
    "u2r": ["root_shell", "su_attempted", "num_file_creations", "num_root"],
}

QOS_WEIGHTS = {"recall": 0.5, "precision": 0.3, "false_alarm": 0.2}
QOS_CLASSES = [
    (85, "Excellent"),
    (70, "Good"),
    (50, "Fair"),
    (0, "Poor"),
]


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def _norm(v, lo, hi):
    """Normalisasi linear v ke rentang [0, 1] terhadap [lo, hi], dipakai utk
    menyusun severity dari fitur mentah (mis. count, num_failed_logins)."""
    if hi <= lo:
        return 0.0
    return _clamp((float(v) - lo) / (hi - lo))


def attack_type_list():
    """Daftar jenis intrusion untuk render checkbox + slider potensi di UI."""
    return [
        {
            "key": k,
            "label": ATTACK_TYPES[k]["label"],
            "description": ATTACK_TYPES[k]["description"],
            "default_potensi": DEFAULT_POTENSI[k],
        }
        for k in ATTACK_ORDER
    ]


# ------------------------------------------------------------- topologi ---
def generate_scenario(num_servers=4, num_clients=20, attacker_ratio=0.25,
                       area_width=1000.0, area_height=1000.0, seed=None):
    """Bangkitkan topologi client-server: server (target) & client tersebar
    acak pada suatu area. `attacker_ratio` (0-1) menentukan proporsi client
    yang berperan sebagai penyerang; sisanya berperan sebagai client umum
    (normal), mirip pembagian server/client pada algorithms/load_balancing.py
    namun di sini client dibedakan berdasarkan perannya (normal vs penyerang)."""
    num_servers = max(1, int(num_servers))
    num_clients = max(1, int(num_clients))
    area_width = max(10.0, float(area_width))
    area_height = max(10.0, float(area_height))
    attacker_ratio = _clamp(float(attacker_ratio), 0.0, 1.0)
    rng = random.Random(seed)

    servers = []
    for i in range(num_servers):
        servers.append({
            "id": f"SRV{i + 1}",
            "x": rng.uniform(0, area_width),
            "y": rng.uniform(0, area_height),
            "capacity": round(rng.uniform(80.0, 260.0), 1),
        })

    num_attackers = max(0, min(num_clients, round(num_clients * attacker_ratio)))
    roles = ["attacker"] * num_attackers + ["normal"] * (num_clients - num_attackers)
    rng.shuffle(roles)

    clients = []
    for i in range(num_clients):
        clients.append({
            "id": f"CLI{i + 1}",
            "x": rng.uniform(0, area_width),
            "y": rng.uniform(0, area_height),
            "role": roles[i],
        })

    actual_ratio = (num_attackers / num_clients) if num_clients else 0.0
    return {
        "servers": servers,
        "clients": clients,
        "area": {"width": area_width, "height": area_height},
        "attacker_ratio": round(actual_ratio, 4),
        "num_attackers": num_attackers,
        "num_normal": num_clients - num_attackers,
    }


# --------------------------------------------------------- fitur traffic --
# Vektor fitur setiap flow mengikuti gaya dataset intrusion detection klasik
# (NSL-KDD/KDD-Cup 99): setiap flow -- normal maupun serangan -- punya fitur
# yang SAMA (baseline di bawah), namun tiap jenis intrusion membuat subset
# fitur tertentu menonjol (lihat KEY_FEATURES) sementara fitur lainnya tetap
# bernilai wajar -- persis seperti pada data intrusion detection sungguhan.
def _baseline_features(rng):
    return {
        "duration": round(rng.uniform(0.5, 12.0), 2),
        "protocol_type": rng.choices(["tcp", "udp", "icmp"], weights=[0.75, 0.20, 0.05])[0],
        "service": rng.choice(["http", "ftp", "smtp", "ssh", "dns", "telnet", "private"]),
        "src_bytes": int(rng.uniform(50, 3000)),
        "dst_bytes": int(rng.uniform(50, 5000)),
        "wrong_fragment": 0,
        "count": rng.randint(1, 8),
        "srv_count": rng.randint(1, 8),
        "serror_rate": round(rng.uniform(0.0, 0.05), 3),
        "rerror_rate": round(rng.uniform(0.0, 0.05), 3),
        "dst_host_diff_srv_rate": round(rng.uniform(0.0, 0.1), 3),
        "dst_host_same_src_port_rate": round(rng.uniform(0.0, 0.1), 3),
        "num_failed_logins": 0,
        "is_guest_login": 0,
        "root_shell": 0,
        "su_attempted": 0,
        "num_file_creations": 0,
        "num_root": 0,
    }


def _gen_normal_features(rng):
    feats = _baseline_features(rng)
    severity = _clamp(rng.uniform(0.0, 0.1))
    return feats, severity


def _gen_dos_features(rng, intensity):
    """DoS/DDoS (banjir trafik): bobot tertinggi pada count & srv_count
    (lonjakan jumlah koneksi ke host/service yang sama dalam jendela waktu),
    serror_rate & rerror_rate (kegagalan handshake TCP SYN / flag REJ -- ciri
    SYN Flood), serta src_bytes & wrong_fragment (payload direkayasa sangat
    besar/fragmentasi ilegal -- ciri Smurf/Ping of Death)."""
    scale = 0.3 + intensity * 1.7
    feats = _baseline_features(rng)
    feats.update({
        "protocol_type": rng.choices(["tcp", "icmp", "udp"], weights=[0.55, 0.30, 0.15])[0],
        "duration": round(rng.uniform(0.01, 1.5), 2),
        "count": int(rng.uniform(120, 400) * scale),
        "srv_count": int(rng.uniform(100, 400) * scale),
        "serror_rate": round(_clamp(rng.uniform(0.55, 1.0) * (0.5 + 0.5 * intensity)), 3),
        "rerror_rate": round(_clamp(rng.uniform(0.2, 0.9) * intensity), 3),
        "src_bytes": int(rng.uniform(800, 8000) * scale),
        "wrong_fragment": rng.choices([0, 1, 2, 3], weights=[0.55, 0.2, 0.15, 0.1])[0] if intensity > 0.4 else 0,
    })
    severity = _clamp(
        0.35 * _norm(feats["count"], 10, 600)
        + 0.30 * ((feats["serror_rate"] + feats["rerror_rate"]) / 2)
        + 0.20 * _norm(feats["src_bytes"], 500, 8000)
        + 0.15 * (1.0 if feats["wrong_fragment"] > 0 else 0.0)
        + rng.uniform(-0.05, 0.05)
    )
    return feats, severity


def _gen_probing_features(rng, intensity):
    """Reconnaissance/Probing (pengintaian): penyerang menyapu banyak
    target/layanan alih-alih membanjiri satu layanan, sehingga bobot
    tertinggi pada dst_host_diff_srv_rate (persentase koneksi ke
    layanan/port berbeda pada satu mesin -- ciri Port Scanning/Nmap) dan
    dst_host_same_src_port_rate (source port statis -- ciri koneksi
    terprogram, bukan pengguna manusia); protocol_type & service turut
    dianomalikan (lonjakan ICMP -- ciri Ping Sweep)."""
    scale = 0.3 + intensity * 1.7
    feats = _baseline_features(rng)
    feats.update({
        "protocol_type": rng.choices(["icmp", "tcp", "udp"], weights=[0.5, 0.35, 0.15])[0],
        "service": rng.choice(["private", "other", "ftp", "http", "telnet"]),
        "duration": round(rng.uniform(0.005, 0.4), 3),
        "src_bytes": int(rng.uniform(0, 60)),
        "count": int(rng.uniform(15, 120) * scale),
        "dst_host_diff_srv_rate": round(_clamp(rng.uniform(0.4, 1.0) * intensity + rng.uniform(0, 0.15)), 3),
        "dst_host_same_src_port_rate": round(_clamp(rng.uniform(0.45, 1.0) * intensity + rng.uniform(0, 0.15)), 3),
    })
    icmp_signal = 0.8 if feats["protocol_type"] == "icmp" else 0.3
    severity = _clamp(
        0.40 * feats["dst_host_diff_srv_rate"]
        + 0.35 * feats["dst_host_same_src_port_rate"]
        + 0.25 * icmp_signal * intensity
        + rng.uniform(-0.05, 0.05)
    )
    return feats, severity


def _gen_r2l_features(rng, intensity):
    """R2L (eksploitasi akses masuk): tidak membanjiri jaringan -- eksploitasi
    berbasis payload pada satu koneksi -- sehingga bobot tertinggi pada
    num_failed_logins (indikator brute-force/penyusupan kata sandi jarak
    jauh) dan is_guest_login (upaya login akun tamu/default); duration ikut
    menjadi sinyal karena sesi R2L yang gagal cenderung sangat singkat
    dibanding koneksi remote sah yang panjang & simetris."""
    scale = 0.3 + intensity * 1.7
    feats = _baseline_features(rng)
    feats.update({
        "protocol_type": "tcp",
        "service": rng.choice(["ssh", "ftp", "telnet", "ftp_data"]),
        "duration": round(rng.uniform(0.1, 3.0), 2),
        "num_failed_logins": max(1, int(rng.uniform(3, 20) * scale)),
        "is_guest_login": 1 if rng.random() < _clamp(0.25 + intensity * 0.55) else 0,
    })
    severity = _clamp(
        0.5 * _norm(feats["num_failed_logins"], 1, 30)
        + 0.3 * feats["is_guest_login"]
        + 0.2 * _norm(3.0 - feats["duration"], 0, 3.0)
        + rng.uniform(-0.05, 0.05)
    )
    return feats, severity


def _gen_u2r_features(rng, intensity):
    """U2R (peningkatan hak akses ilegal): tersulit dideteksi krn hanya butuh
    1-3 koneksi kecil, sehingga deteksi eksklusif mengandalkan fitur konten:
    root_shell & su_attempted (eksekusi perintah level kernel/root secara
    mendadak oleh pengguna tanpa privilege), num_file_creations (skrip
    proksi/backdoor pasca buffer overflow), dan num_root (lonjakan operasi
    berkas yang mewajibkan hak akses root pasca-otentikasi sebagai user
    biasa)."""
    scale = 0.3 + intensity * 1.7
    feats = _baseline_features(rng)
    feats.update({
        "protocol_type": "tcp",
        "service": rng.choice(["telnet", "ssh", "ftp", "private"]),
        "duration": round(rng.uniform(0.5, 6.0), 2),
        "count": rng.randint(1, 3),
        "root_shell": 1 if rng.random() < _clamp(0.3 + intensity * 0.6) else 0,
        "su_attempted": 1 if rng.random() < _clamp(0.25 + intensity * 0.6) else 0,
        "num_file_creations": max(0, int(rng.uniform(0, 6) * scale)),
        "num_root": max(0, int(rng.uniform(0, 6) * scale)),
    })
    severity = _clamp(
        0.3 * feats["root_shell"]
        + 0.25 * feats["su_attempted"]
        + 0.25 * _norm(feats["num_file_creations"], 0, 8)
        + 0.2 * _norm(feats["num_root"], 0, 8)
        + rng.uniform(-0.05, 0.05)
    )
    return feats, severity


_ATTACK_GENERATORS = {
    "dos_ddos": _gen_dos_features,
    "probing": _gen_probing_features,
    "r2l": _gen_r2l_features,
    "u2r": _gen_u2r_features,
}


def _detect(label, severity, rng):
    """Detector berbasis aturan sederhana: peluang deteksi naik seiring
    severity (mensimulasikan IDS yang lebih mudah menangkap serangan yang
    lebih 'kentara'/intens); traffic normal tetap berpeluang kecil memicu
    false alarm (false positive)."""
    if label == "normal":
        p_detect = 0.03 + severity * 0.15
    else:
        p_detect = _clamp(0.3 + severity * 0.65)
    detected = rng.random() < p_detect
    confidence = round(_clamp(p_detect * rng.uniform(0.85, 1.15)), 3)
    return detected, confidence


# ------------------------------------------------------------- simulasi --
def simulate_traffic(cluster, selected_attacks, potensi, flows_per_client=6, seed=None):
    """Bangkitkan flow traffic acak client->server pada `cluster` (hasil
    generate_scenario). Tiap client mengirim `flows_per_client` flow ke server
    yang dipilih acak. Client berperan 'normal' selalu menghasilkan traffic
    normal; client 'attacker' menghasilkan traffic sesuai salah satu jenis
    intrusion pada `selected_attacks` (dibobot oleh nilai `potensi` tiap
    jenis, 0-1) -- bila `selected_attacks` kosong, seluruh traffic (termasuk
    dari client penyerang) dianggap normal, artinya tidak ada intrusion yang
    disimulasikan sama sekali.

    Mengembalikan (flows, summary): `flows` adalah daftar dict per flow
    (client, server, peran, label sebenarnya, fitur, severity, hasil
    deteksi), `summary` adalah agregat per label + confusion matrix + skor.
    """
    rng = random.Random(seed)
    servers = cluster["servers"]
    clients = cluster["clients"]
    flows_per_client = max(1, int(flows_per_client))

    selected = [a for a in (selected_attacks or []) if a in ATTACK_TYPES]
    weights = [max(0.02, _clamp(float((potensi or {}).get(a, 0.5)))) for a in selected]

    flows = []
    fid = 1
    for c in clients:
        for _ in range(flows_per_client):
            server = rng.choice(servers)
            if c["role"] == "attacker" and selected:
                atype = rng.choices(selected, weights=weights, k=1)[0]
                intensity = _clamp(float((potensi or {}).get(atype, 0.5)))
                feats, severity = _ATTACK_GENERATORS[atype](rng, intensity)
                label = atype
            else:
                feats, severity = _gen_normal_features(rng)
                label = "normal"
            detected, confidence = _detect(label, severity, rng)
            flows.append({
                "id": f"F{fid}",
                "client_id": c["id"],
                "server_id": server["id"],
                "client_role": c["role"],
                "label": label,
                "severity": round(severity, 3),
                "features": feats,
                "detected": detected,
                "confidence": confidence,
            })
            fid += 1

    summary = summarize_flows(flows, selected)
    return flows, summary


def summarize_flows(flows, selected_attacks):
    """Agregat jumlah/detection rate/severity per label, rata-rata fitur
    kunci tiap jenis intrusion (lihat KEY_FEATURES), confusion matrix
    (TP/FN/FP/TN, intrusion dianggap kelas positif), serta skor deteksi
    keseluruhan (0-100) & kelasnya (Excellent/Good/Fair/Poor, gaya sama
    dengan qos.py)."""
    per_label = {"normal": {"count": 0, "detected": 0, "severity_sum": 0.0, "feat_sums": {}}}
    for a in selected_attacks:
        per_label[a] = {"count": 0, "detected": 0, "severity_sum": 0.0, "feat_sums": {}}

    tp = fn = fp = tn = 0
    for f in flows:
        lab = f["label"]
        agg = per_label.setdefault(lab, {"count": 0, "detected": 0, "severity_sum": 0.0, "feat_sums": {}})
        agg["count"] += 1
        agg["severity_sum"] += f["severity"]
        if f["detected"]:
            agg["detected"] += 1
        for key in KEY_FEATURES.get(lab, []):
            val = f["features"].get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                agg["feat_sums"][key] = agg["feat_sums"].get(key, 0.0) + val
            elif isinstance(val, bool):
                agg["feat_sums"][key] = agg["feat_sums"].get(key, 0.0) + (1.0 if val else 0.0)
        if lab == "normal":
            fp += 1 if f["detected"] else 0
            tn += 0 if f["detected"] else 1
        else:
            tp += 1 if f["detected"] else 0
            fn += 0 if f["detected"] else 1

    per_label_out = []
    for lab, agg in per_label.items():
        cnt = agg["count"]
        det_rate = round(agg["detected"] / cnt * 100, 1) if cnt else None
        avg_sev = round(agg["severity_sum"] / cnt, 3) if cnt else None
        key_features_avg = [
            {"feature": key, "avg": round(agg["feat_sums"].get(key, 0.0) / cnt, 3)}
            for key in KEY_FEATURES.get(lab, [])
        ] if cnt else []
        per_label_out.append({
            "label": lab,
            "label_text": "Normal" if lab == "normal" else ATTACK_TYPES[lab]["label"],
            "count": cnt,
            "detected": agg["detected"],
            "detection_rate_pct": det_rate,
            "avg_severity": avg_sev,
            "key_features_avg": key_features_avg,
        })
    per_label_out.sort(key=lambda r: (r["label"] != "normal", r["label"]))

    total_attacks = tp + fn
    total_normal = tn + fp
    precision = (tp / (tp + fp)) if (tp + fp) else None
    recall = (tp / (tp + fn)) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and (precision + recall)) else None
    accuracy = ((tp + tn) / len(flows)) if flows else None
    false_alarm_rate = (fp / total_normal) if total_normal else 0.0

    score = None
    qos_class = None
    if flows:
        rec = recall if recall is not None else (1.0 if total_attacks == 0 else 0.0)
        prec = precision if precision is not None else (1.0 if (tp + fp) == 0 else 0.0)
        score = _clamp(
            QOS_WEIGHTS["recall"] * rec
            + QOS_WEIGHTS["precision"] * prec
            + QOS_WEIGHTS["false_alarm"] * (1 - false_alarm_rate)
        ) * 100
        for threshold, cls in QOS_CLASSES:
            if score >= threshold:
                qos_class = cls
                break

    return {
        "total_flows": len(flows),
        "total_attack_flows": total_attacks,
        "total_normal_flows": total_normal,
        "per_label": per_label_out,
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "precision": round(precision, 3) if precision is not None else None,
        "recall": round(recall, 3) if recall is not None else None,
        "f1_score": round(f1, 3) if f1 is not None else None,
        "accuracy": round(accuracy, 3) if accuracy is not None else None,
        "false_alarm_rate_pct": round(false_alarm_rate * 100, 2),
        "score": round(score, 1) if score is not None else None,
        "qos_class": qos_class,
    }
