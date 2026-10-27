"""Data contoh: rumah 10 ruang dengan instalasi, sensor, rule, dan penghuni."""
from models import (Circuit, Component, DEFAULT_POWER, DEFAULT_STANDBY,
                    House, Occupant, Room, Rule, SwitchLink, db)

ROOMS = [
    # name,               kind,       x,   y,   w,   h,  color,     window
    ("Teras",             "teras",    30,  30, 180, 210, "#fef3c7", 0.75),
    ("Ruang Tamu",        "ruang",   220,  30, 290, 210, "#e0f2fe", 0.30),
    ("Kamar Tidur Utama", "kamar",   520,  30, 250, 210, "#ede9fe", 0.22),
    ("Kamar Anak",        "kamar",   780,  30, 230, 210, "#fce7f3", 0.22),
    ("Garasi",            "garasi",   30, 250, 180, 200, "#e5e7eb", 0.15),
    ("Dapur",             "dapur",   220, 250, 230, 200, "#ffedd5", 0.25),
    ("Ruang Makan",       "ruang",   460, 250, 210, 200, "#dcfce7", 0.25),
    ("Kamar Mandi",       "kmandi",  680, 250, 150, 200, "#cffafe", 0.12),
    ("Ruang Kerja",       "kerja",   840, 250, 170, 200, "#f1f5f9", 0.28),
    ("Halaman Belakang",  "halaman",  30, 460, 980, 160, "#dcfce7", 1.00),
]


def seed_demo():
    house = House(name="Rumah Contoh - Simulator Smart Home",
                  voltage=220.0, power_factor=0.9, tariff=1444.70,
                  main_mcb_a=10.0, plan_w=1210, plan_h=650,
                  sim_minute=18 * 60, step_minutes=10, weather="cerah")
    db.session.add(house)
    db.session.flush()

    rooms = {}
    for name, kind, x, y, w, h, color, win in ROOMS:
        r = Room(house_id=house.id, name=name, kind=kind, x=x, y=y, w=w, h=h,
                 color=color, window_factor=win, temp_c=28.0, humidity=68.0)
        db.session.add(r)
        rooms[name] = r
    db.session.flush()

    # ------------------------------------------------------ sumber listrik
    def source(subtype, name, cap, rx, ry, enabled=True):
        c = Component(house_id=house.id, kind="source", subtype=subtype,
                      name=name, x=rx, y=ry, capacity_w=cap, enabled=enabled,
                      mode="manual", manual_state=enabled, params={})
        db.session.add(c)
        return c

    pln = source("pln", "PLN 2200 VA", 2200, 1110, 95)
    genset = source("genset", "Genset", 3000, 1110, 195, enabled=False)
    solar = source("solar", "Panel Surya", 800, 1110, 295, enabled=False)
    db.session.flush()

    # ------------------------------------------------------------ sirkuit
    def circuit(name, mcb, color, src=pln):
        c = Circuit(house_id=house.id, name=name, source_id=src.id,
                    mcb_a=mcb, color=color)
        db.session.add(c)
        return c

    c_lamp = circuit("Grup 1 - Penerangan", 4.0, "#f59e0b")
    c_plug = circuit("Grup 2 - Stop Kontak", 6.0, "#3b82f6")
    c_big = circuit("Grup 3 - Daya Besar", 10.0, "#ef4444")
    c_out = circuit("Grup 4 - Luar & Taman", 4.0, "#10b981")
    db.session.flush()

    # ----------------------------------------------------------- komponen
    def comp(kind, subtype, name, room, dx, dy, circ=None, power=None,
             mode="auto", combine="parallel", params=None, manual=False):
        r = rooms[room]
        c = Component(
            house_id=house.id, room_id=r.id,
            circuit_id=circ.id if circ else None,
            kind=kind, subtype=subtype, name=name,
            x=r.x + dx, y=r.y + dy,
            power_w=power if power is not None else DEFAULT_POWER.get(subtype, 0),
            standby_w=DEFAULT_STANDBY.get(subtype, 0.0),
            mode=mode, combine=combine, manual_state=manual,
            params=params or {})
        db.session.add(c)
        return c

    # --- perangkat
    d = {}
    d["lampu_teras"] = comp("device", "lampu", "Lampu Teras", "Teras", 90, 60, c_out, 18)
    d["cctv"] = comp("device", "cctv", "CCTV Depan", "Teras", 30, 40, c_out)
    d["lampu_tamu"] = comp("device", "lampu", "Lampu Tamu", "Ruang Tamu", 145, 60, c_lamp, 24)
    d["tv"] = comp("device", "tv", "TV LED", "Ruang Tamu", 60, 140, c_plug)
    d["ac_tamu"] = comp("device", "ac", "AC Tamu", "Ruang Tamu", 230, 40, c_big, 680,
                        combine="series", params={"setpoint": 25})
    d["kipas_tamu"] = comp("device", "kipas", "Kipas Angin", "Ruang Tamu", 230, 150, c_plug)
    d["router"] = comp("device", "router", "Router WiFi", "Ruang Tamu", 60, 40, c_plug)

    d["lampu_ku"] = comp("device", "lampu", "Lampu Utama", "Kamar Tidur Utama", 125, 60, c_lamp, 18)
    d["ac_ku"] = comp("device", "ac", "AC Utama", "Kamar Tidur Utama", 200, 40, c_big, 560,
                      combine="series", params={"setpoint": 24})
    d["charger"] = comp("device", "charger", "Charger HP", "Kamar Tidur Utama", 50, 150, c_plug)

    d["lampu_ka"] = comp("device", "lampu", "Lampu Anak", "Kamar Anak", 115, 60, c_lamp, 18)
    d["kipas_ka"] = comp("device", "kipas", "Kipas Anak", "Kamar Anak", 180, 150, c_plug)

    d["lampu_garasi"] = comp("device", "lampu", "Lampu Garasi", "Garasi", 90, 60, c_out, 20)
    d["mesin_cuci"] = comp("device", "mesin_cuci", "Mesin Cuci", "Garasi", 45, 150, c_big)
    d["pompa"] = comp("device", "pompa_air", "Pompa Air", "Garasi", 140, 150, c_big)

    d["lampu_dapur"] = comp("device", "lampu", "Lampu Dapur", "Dapur", 115, 55, c_lamp, 20)
    d["kulkas"] = comp("device", "kulkas", "Kulkas 2 Pintu", "Dapur", 40, 140, c_plug,
                       mode="manual", manual=True)
    d["rice"] = comp("device", "rice_cooker", "Rice Cooker", "Dapur", 110, 140, c_plug)
    d["exhaust"] = comp("device", "exhaust", "Exhaust Fan", "Dapur", 180, 140, c_plug)

    d["lampu_makan"] = comp("device", "lampu", "Lampu Makan", "Ruang Makan", 105, 55, c_lamp, 18)
    d["dispenser"] = comp("device", "dispenser", "Dispenser", "Ruang Makan", 45, 145, c_plug)

    d["lampu_mandi"] = comp("device", "lampu", "Lampu Mandi", "Kamar Mandi", 75, 55, c_lamp, 12)
    d["heater"] = comp("device", "water_heater", "Water Heater", "Kamar Mandi", 75, 145, c_big)

    d["lampu_kerja"] = comp("device", "lampu", "Lampu Kerja", "Ruang Kerja", 85, 55, c_lamp, 18)
    d["komputer"] = comp("device", "komputer", "Komputer", "Ruang Kerja", 45, 145, c_plug)
    d["ac_kerja"] = comp("device", "ac", "AC Kerja", "Ruang Kerja", 130, 145, c_big, 620,
                         combine="series", params={"setpoint": 25})

    d["lampu_taman"] = comp("device", "lampu", "Lampu Taman", "Halaman Belakang", 500, 90, c_out, 30)
    db.session.flush()

    # --- saklar & sensor
    s = {}
    s["ldr_luar"] = comp("switch", "ldr", "LDR Luar", "Teras", 140, 30, c_out,
                         power=0, params={"threshold": 120, "op": "<"})
    s["pir_teras"] = comp("switch", "pir", "PIR Teras", "Teras", 60, 30, c_out, power=0)
    s["hujan"] = comp("switch", "hujan", "Sensor Hujan", "Teras", 30, 150, c_out, power=0)

    s["sk_tamu"] = comp("switch", "saklar", "Saklar Tamu", "Ruang Tamu", 20, 90, c_lamp, power=0)
    s["pir_tamu"] = comp("switch", "pir", "PIR Tamu", "Ruang Tamu", 145, 20, c_lamp, power=0)
    s["ir_tv"] = comp("switch", "infrared", "Remote Tamu", "Ruang Tamu", 110, 145, c_plug, power=0)
    s["suhu_tamu"] = comp("switch", "suhu", "Suhu Tamu", "Ruang Tamu", 265, 100, c_lamp,
                          power=0, params={"threshold": 28.5, "op": ">"})

    s["sk_ku"] = comp("switch", "saklar", "Saklar Utama", "Kamar Tidur Utama", 20, 90, c_lamp, power=0)
    s["suhu_ku"] = comp("switch", "suhu", "Suhu K. Utama", "Kamar Tidur Utama", 215, 100, c_lamp,
                        power=0, params={"threshold": 28.0, "op": ">"})
    s["ir_ku"] = comp("switch", "infrared", "Remote Utama", "Kamar Tidur Utama", 175, 150, c_lamp, power=0)

    s["sk_ka"] = comp("switch", "saklar", "Saklar Anak", "Kamar Anak", 20, 90, c_lamp, power=0)
    s["pir_ka"] = comp("switch", "pir", "PIR Anak", "Kamar Anak", 115, 20, c_lamp, power=0)

    s["pintu_garasi"] = comp("switch", "pintu", "Pintu Garasi", "Garasi", 20, 30, c_out, power=0)
    s["timer_pompa"] = comp("switch", "timer", "Timer Pompa", "Garasi", 140, 100, c_big,
                            power=0, params={"on_hour": 5, "off_hour": 6})

    s["sk_dapur"] = comp("switch", "saklar", "Saklar Dapur", "Dapur", 20, 90, c_lamp, power=0)
    s["asap"] = comp("switch", "asap", "Asap Dapur", "Dapur", 115, 20, c_lamp, power=0)
    s["lembab_dapur"] = comp("switch", "kelembaban", "Lembab Dapur", "Dapur", 185, 90, c_plug,
                             power=0, params={"threshold": 78, "op": ">"})

    s["sk_makan"] = comp("switch", "saklar", "Saklar Makan", "Ruang Makan", 20, 90, c_lamp, power=0)
    s["pir_mandi"] = comp("switch", "pir", "PIR Mandi", "Kamar Mandi", 20, 90, c_lamp, power=0)
    s["tb_heater"] = comp("switch", "tombol", "Tombol Heater", "Kamar Mandi", 120, 145, c_big,
                          power=0, params={})
    s["sk_kerja"] = comp("switch", "saklar", "Saklar Kerja", "Ruang Kerja", 20, 90, c_lamp, power=0)
    s["suhu_kerja"] = comp("switch", "suhu", "Suhu Kerja", "Ruang Kerja", 130, 90, c_lamp,
                           power=0, params={"threshold": 28.5, "op": ">"})
    s["ir_kerja"] = comp("switch", "infrared", "Remote Kerja", "Ruang Kerja", 90, 145, c_lamp, power=0)

    s["ldr_taman"] = comp("switch", "ldr", "LDR Taman", "Halaman Belakang", 430, 60, c_out,
                          power=0, params={"threshold": 150, "op": "<"})
    s["pir_taman"] = comp("switch", "pir", "PIR Halaman", "Halaman Belakang", 570, 60, c_out, power=0)
    db.session.flush()

    # ------------------------------------------------------- pengkabelan
    def link(sw, dev, inverted=False):
        db.session.add(SwitchLink(switch_id=s[sw].id, device_id=d[dev].id,
                                  inverted=inverted))

    # paralel: salah satu aktif -> perangkat menyala
    link("ldr_luar", "lampu_teras")
    link("pir_teras", "lampu_teras")
    link("hujan", "lampu_teras")
    link("ldr_luar", "lampu_garasi")
    link("ldr_luar", "lampu_taman")
    link("pir_taman", "lampu_taman")
    link("pintu_garasi", "lampu_garasi")

    link("sk_tamu", "lampu_tamu")
    link("pir_tamu", "lampu_tamu")
    link("ir_tv", "tv")
    link("sk_tamu", "kipas_tamu")

    link("sk_ku", "lampu_ku")
    link("sk_ka", "lampu_ka")
    link("pir_ka", "lampu_ka")
    link("sk_dapur", "lampu_dapur")
    link("lembab_dapur", "exhaust")
    link("asap", "exhaust")
    link("sk_makan", "lampu_makan")
    link("pir_mandi", "lampu_mandi")
    link("tb_heater", "heater")
    link("sk_kerja", "lampu_kerja")
    link("timer_pompa", "pompa")

    # seri: semua syarat harus aktif (remote ON + suhu tinggi)
    link("ir_tv", "ac_tamu")
    link("suhu_tamu", "ac_tamu")
    link("ir_ku", "ac_ku")
    link("suhu_ku", "ac_ku")
    link("ir_kerja", "ac_kerja")
    link("suhu_kerja", "ac_kerja")

    # -------------------------------------------------------------- rule
    def rule(name, logic, conds, acts, prio=10, enabled=True):
        db.session.add(Rule(house_id=house.id, name=name, logic=logic,
                            conditions=conds, actions=acts, priority=prio,
                            enabled=enabled))

    rule("Lampu tamu otomatis saat malam & ada orang", "AND",
         [{"subject": "time_of_day", "ref": None, "op": "==", "value": "malam"},
          {"subject": "occupancy", "ref": rooms["Ruang Tamu"].id, "op": ">", "value": 0}],
         [{"target": "device", "ref": d["lampu_tamu"].id, "act": "on"}], 10)

    rule("Matikan hiburan saat rumah kosong", "AND",
         [{"subject": "occupant_total", "ref": None, "op": "==", "value": 0}],
         [{"target": "device", "ref": d["tv"].id, "act": "off"},
          {"target": "device", "ref": d["ac_tamu"].id, "act": "off"},
          {"target": "device", "ref": d["komputer"].id, "act": "off"},
          {"target": "device", "ref": d["ac_kerja"].id, "act": "off"}], 20)

    rule("Kipas anak menyala saat kamar panas", "AND",
         [{"subject": "room_temp", "ref": rooms["Kamar Anak"].id, "op": ">", "value": 29},
          {"subject": "occupancy", "ref": rooms["Kamar Anak"].id, "op": ">", "value": 0}],
         [{"target": "device", "ref": d["kipas_ka"].id, "act": "on"}], 10)

    rule("Lampu taman menyala saat hujan atau gelap", "OR",
         [{"subject": "weather", "ref": None, "op": "==", "value": "hujan"},
          {"subject": "outdoor_lux", "ref": None, "op": "<", "value": 200}],
         [{"target": "device", "ref": d["lampu_taman"].id, "act": "on"}], 10)

    rule("Proteksi beban: matikan water heater bila daya > 2000 W", "AND",
         [{"subject": "total_power", "ref": None, "op": ">", "value": 2000}],
         [{"target": "device", "ref": d["heater"].id, "act": "off"},
          {"target": "device", "ref": d["mesin_cuci"].id, "act": "off"}], 90)

    rule("Alarm asap: exhaust & lampu dapur menyala", "AND",
         [{"subject": "sensor", "ref": s["asap"].id, "op": "==", "value": True}],
         [{"target": "device", "ref": d["exhaust"].id, "act": "on"},
          {"target": "device", "ref": d["lampu_dapur"].id, "act": "on"}], 95)

    rule("Mode tidur: lampu utama mati setelah pukul 23", "AND",
         [{"subject": "hour", "ref": None, "op": ">=", "value": 23}],
         [{"target": "device", "ref": d["lampu_tamu"].id, "act": "off"},
          {"target": "device", "ref": d["lampu_makan"].id, "act": "off"}], 30,
         enabled=False)

    # ---------------------------------------------------------- penghuni
    R = {k: v.id for k, v in rooms.items()}

    def sched(pairs):
        out = {}
        for h_start, h_end, target in pairs:
            for hh in range(h_start, h_end):
                out[str(hh % 24)] = target
        return out

    db.session.add(Occupant(
        house_id=house.id, name="Ayah", role="orang tua", color="#2563eb",
        room_id=R["Ruang Tamu"], activity="santai", auto_move=True,
        schedule=sched([(0, 5, R["Kamar Tidur Utama"]),
                        (5, 6, R["Kamar Mandi"]),
                        (6, 7, R["Ruang Makan"]),
                        (7, 17, "away"),
                        (17, 19, R["Ruang Tamu"]),
                        (19, 20, R["Ruang Makan"]),
                        (20, 22, R["Ruang Tamu"]),
                        (22, 24, R["Kamar Tidur Utama"])])))
    db.session.add(Occupant(
        house_id=house.id, name="Ibu", role="orang tua", color="#db2777",
        room_id=R["Dapur"], activity="memasak", auto_move=True,
        schedule=sched([(0, 5, R["Kamar Tidur Utama"]),
                        (5, 7, R["Dapur"]),
                        (7, 9, R["Ruang Kerja"]),
                        (9, 11, R["Garasi"]),
                        (11, 13, R["Dapur"]),
                        (13, 16, R["Ruang Tamu"]),
                        (16, 19, R["Dapur"]),
                        (19, 20, R["Ruang Makan"]),
                        (20, 22, R["Ruang Tamu"]),
                        (22, 24, R["Kamar Tidur Utama"])])))
    db.session.add(Occupant(
        house_id=house.id, name="Anak", role="pelajar", color="#16a34a",
        room_id=R["Kamar Anak"], activity="belajar", auto_move=True,
        schedule=sched([(0, 6, R["Kamar Anak"]),
                        (6, 7, R["Kamar Mandi"]),
                        (7, 14, "away"),
                        (14, 16, R["Ruang Tamu"]),
                        (16, 18, R["Kamar Anak"]),
                        (18, 19, R["Kamar Mandi"]),
                        (19, 20, R["Ruang Makan"]),
                        (20, 21, R["Ruang Tamu"]),
                        (21, 24, R["Kamar Anak"])])))

    layout_components(house)
    db.session.commit()
    return house


def layout_components(house):
    """Susun ulang posisi komponen menjadi grid rapi di dalam tiap kamar."""
    import math
    for room in Room.query.filter_by(house_id=house.id):
        comps = (Component.query.filter_by(house_id=house.id, room_id=room.id)
                 .order_by(Component.kind.desc(), Component.id).all())
        n = len(comps)
        if not n:
            continue
        max_rows = max(1, int((room.h - 46) // 54))
        cols = max(1, math.ceil(n / max_rows))
        rows = math.ceil(n / cols)
        step_x = min(130.0, (room.w - 24) / cols)
        step_y = min(70.0, (room.h - 50) / rows)
        grid_w = step_x * cols
        x0 = room.x + (room.w - grid_w) / 2
        y0 = room.y + 46
        for i, c in enumerate(comps):
            c.x = round(x0 + step_x * (i % cols) + step_x / 2, 1)
            c.y = round(y0 + step_y * (i // cols) + step_y / 2 - 4, 1)
