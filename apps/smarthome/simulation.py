"""Mesin simulasi rumah pintar + instalasi listrik.

Satu langkah simulasi (tick) mengerjakan urutan berikut:
 1. memajukan jam simulasi;
 2. menghitung kondisi lingkungan (cuaca, cahaya luar, suhu luar);
 3. memindahkan penghuni sesuai jadwal;
 4. memperbarui fisika ruangan (cahaya, suhu, kelembaban);
 5. mengevaluasi saklar & sensor;
 6. mengevaluasi rule otomasi -> override;
 7. menentukan status akhir tiap perangkat (logika seri/paralel);
 8. menghitung daya, arus, proteksi MCB;
 9. mengakumulasi energi (kWh) dan biaya.
"""
import math
import random

from models import (Circuit, Component, EnergySample, EventLog, Occupant,
                    Room, Rule, SwitchLink, WEATHERS, band_of, db)

# perangkat yang melepas panas ke ruangan (faktor terhadap daya)
HEAT_FACTOR = {
    "lampu": 0.7, "tv": 0.5, "komputer": 0.8, "microwave": 0.6,
    "rice_cooker": 0.9, "water_heater": 0.9, "setrika": 0.9,
    "kulkas": 0.6, "dispenser": 0.6, "mesin_cuci": 0.3, "lainnya": 0.4,
}
LUMEN_PER_W = 85.0          # efisiensi lampu LED
MAX_LOG = 400
MAX_SAMPLES = 600


# --------------------------------------------------------------- lingkungan
def outdoor_lux(hour, weather):
    factor = WEATHERS.get(weather, WEATHERS["cerah"])[0]
    base = math.sin(math.pi * (hour - 6.0) / 12.0)
    if base <= 0:
        return 0.0
    return base * 95000.0 * factor


def outdoor_temp(hour, weather, offset=0.0):
    delta = WEATHERS.get(weather, WEATHERS["cerah"])[1]
    # siklus harian penuh: puncak pukul 14.30, terdingin pukul 02.30
    base = 27.0 + 4.2 * math.cos(2 * math.pi * (hour - 14.5) / 24.0)
    return base + delta + offset


def outdoor_humidity(weather):
    return WEATHERS.get(weather, WEATHERS["cerah"])[2]


def compare(left, op, right):
    try:
        if isinstance(right, str) and not isinstance(left, str):
            right = float(right)
        if isinstance(left, bool) or isinstance(right, bool):
            left, right = bool(left), bool(right)
    except (TypeError, ValueError):
        left, right = str(left), str(right)
    try:
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
    except TypeError:
        return False
    return False


def log(house, message, level="info", category="sim"):
    db.session.add(EventLog(house_id=house.id, sim_time=house.clock,
                            level=level, category=category, message=message))


# ------------------------------------------------------------------ sensor
def eval_switch(sw, ctx):
    """Kembalikan status logika sebuah saklar / sensor."""
    if not sw.enabled or sw.mode == "off":
        return False
    if sw.mode == "manual":
        return bool(sw.manual_state)

    p = sw.params or {}
    st = sw.subtype
    rid = sw.room_id

    if st in ("tombol", "saklar", "infrared", "pintu", "asap"):
        return bool(sw.manual_state)
    if st == "pir":
        return ctx["occupancy"].get(rid, 0) > 0
    if st == "ldr":
        return compare(ctx["room_lux"].get(rid, 0.0),
                       p.get("op", "<"), float(p.get("threshold", 80)))
    if st == "suhu":
        return compare(ctx["room_temp"].get(rid, 28.0),
                       p.get("op", ">"), float(p.get("threshold", 29)))
    if st == "kelembaban":
        return compare(ctx["room_humidity"].get(rid, 65.0),
                       p.get("op", ">"), float(p.get("threshold", 80)))
    if st == "hujan":
        return ctx["weather"] in ("hujan", "badai")
    if st == "timer":
        on_h = float(p.get("on_hour", 18))
        off_h = float(p.get("off_hour", 6))
        h = ctx["hour"]
        if on_h <= off_h:
            return on_h <= h < off_h
        return h >= on_h or h < off_h
    return bool(sw.manual_state)


# -------------------------------------------------------------------- rule
def subject_value(cond, ctx):
    subj = cond.get("subject")
    ref = cond.get("ref")
    try:
        ref = int(ref) if ref not in (None, "", "null") else None
    except (TypeError, ValueError):
        ref = None

    if subj == "time_of_day":
        return ctx["time_of_day"]
    if subj == "hour":
        return ctx["hour"]
    if subj == "weather":
        return ctx["weather"]
    if subj == "outdoor_temp":
        return ctx["outdoor_temp"]
    if subj == "outdoor_lux":
        return ctx["outdoor_lux"]
    if subj == "room_temp":
        return ctx["room_temp"].get(ref, 0.0)
    if subj == "room_lux":
        return ctx["room_lux"].get(ref, 0.0)
    if subj == "room_humidity":
        return ctx["room_humidity"].get(ref, 0.0)
    if subj == "occupancy":
        return ctx["occupancy"].get(ref, 0)
    if subj == "occupant_total":
        return ctx["occupant_total"]
    if subj == "sensor":
        return bool(ctx["switch_states"].get(ref, False))
    if subj == "device":
        return bool(ctx["device_states"].get(ref, False))
    if subj == "total_power":
        return ctx["total_w"]
    if subj == "energy_kwh":
        return ctx["energy_kwh"]
    return None


def eval_rule(rule, ctx):
    conds = rule.conditions or []
    if not conds:
        return False
    results = []
    for c in conds:
        left = subject_value(c, ctx)
        if left is None:
            results.append(False)
            continue
        results.append(compare(left, c.get("op", "=="), c.get("value")))
    return all(results) if (rule.logic or "AND").upper() == "AND" else any(results)


# ------------------------------------------------------------- resolusi alat
def resolve_device(dev, sw_states, links_by_device, overrides):
    if not dev.enabled:
        return False
    if dev.id in overrides:
        return bool(overrides[dev.id])
    if dev.mode == "off":
        return False
    if dev.mode == "manual":
        return bool(dev.manual_state)
    links = links_by_device.get(dev.id, [])
    if not links:
        return bool(dev.manual_state)
    vals = []
    for lk in links:
        v = bool(sw_states.get(lk.switch_id, False))
        vals.append((not v) if lk.inverted else v)
    return any(vals) if dev.combine == "parallel" else all(vals)


# -------------------------------------------------------------------- tick
def tick(house, advance=True):
    """Jalankan satu langkah simulasi. Kembalikan ringkasan hasil."""
    step = max(1, int(house.step_minutes or 10))
    if advance:
        house.sim_minute = int(house.sim_minute) + step
        if house.sim_minute >= 1440:
            house.sim_minute -= 1440
            house.sim_day = (house.sim_day or 1) + 1
            if house.auto_weather:
                house.weather = random.choice(list(WEATHERS.keys()))
    dt_h = step / 60.0

    rooms = Room.query.filter_by(house_id=house.id).all()
    comps = Component.query.filter_by(house_id=house.id).all()
    circuits = Circuit.query.filter_by(house_id=house.id).all()
    occupants = Occupant.query.filter_by(house_id=house.id).all()
    links = (SwitchLink.query
             .join(Component, SwitchLink.device_id == Component.id)
             .filter(Component.house_id == house.id).all())

    sources = [c for c in comps if c.kind == "source"]
    switches = [c for c in comps if c.kind == "switch"]
    devices = [c for c in comps if c.kind == "device"]
    room_by_id = {r.id: r for r in rooms}
    comp_by_id = {c.id: c for c in comps}
    circ_by_id = {c.id: c for c in circuits}

    links_by_device = {}
    for lk in links:
        links_by_device.setdefault(lk.device_id, []).append(lk)

    # ---- 1. lingkungan
    hour = house.hour
    o_lux = outdoor_lux(hour, house.weather)
    o_temp = outdoor_temp(hour, house.weather, house.outdoor_temp_offset or 0)
    o_hum = outdoor_humidity(house.weather)

    # ---- 2. penghuni bergerak sesuai jadwal
    occupancy = {r.id: 0 for r in rooms}
    for oc in occupants:
        if oc.auto_move and oc.schedule:
            key = str(int(hour))
            target = (oc.schedule or {}).get(key)
            if target == "away":
                oc.away, oc.room_id = True, None
            elif target is not None:
                try:
                    tid = int(target)
                except (TypeError, ValueError):
                    tid = None
                if tid in room_by_id:
                    oc.away, oc.room_id = False, tid
        if not oc.away and oc.room_id in occupancy:
            occupancy[oc.room_id] += 1
    occupant_total = sum(occupancy.values())

    # ---- 3. fisika ruangan (memakai status perangkat langkah sebelumnya)
    dev_by_room = {}
    for d in devices:
        dev_by_room.setdefault(d.room_id, []).append(d)

    for r in rooms:
        rdev = dev_by_room.get(r.id, [])
        lamp_lux = 0.0
        heat_w = 0.0
        ac_setpoint = None
        fan_w = 0.0
        for d in rdev:
            if not d.state:
                continue
            if d.subtype == "lampu":
                area = max(4.0, (r.w * r.h) / 900.0)     # ~m2 skala denah
                lamp_lux += (d.power_w * LUMEN_PER_W) / area
            elif d.subtype == "ac":
                sp = float((d.params or {}).get("setpoint", 24))
                ac_setpoint = sp if ac_setpoint is None else min(ac_setpoint, sp)
            elif d.subtype in ("kipas", "exhaust"):
                fan_w += d.power_w
            heat_w += d.power_w * HEAT_FACTOR.get(d.subtype, 0.3)

        r.lux = o_lux * (r.window_factor or 0.2) + lamp_lux

        target = o_temp + heat_w / 110.0 + occupancy.get(r.id, 0) * 0.45
        if fan_w > 0:
            target -= 1.2
        if ac_setpoint is not None:
            target = min(target, ac_setpoint)
        alpha = 1.0 - math.exp(-dt_h / 0.75)             # tau = 45 menit
        r.temp_c = r.temp_c + (target - r.temp_c) * alpha

        h_target = o_hum - (8.0 if ac_setpoint is not None else 0.0)
        r.humidity = r.humidity + (h_target - r.humidity) * alpha

    ctx = {
        "hour": hour, "time_of_day": band_of(hour), "weather": house.weather,
        "outdoor_temp": round(o_temp, 2), "outdoor_lux": round(o_lux, 1),
        "occupancy": occupancy, "occupant_total": occupant_total,
        "room_temp": {r.id: r.temp_c for r in rooms},
        "room_lux": {r.id: r.lux for r in rooms},
        "room_humidity": {r.id: r.humidity for r in rooms},
        "switch_states": {}, "device_states": {d.id: d.state for d in devices},
        # daya total pada langkah sebelumnya, dipakai syarat rule "total_power"
        "total_w": round(sum(c.load_w or 0.0 for c in circuits if not c.tripped), 1),
        "energy_kwh": house.energy_kwh or 0.0,
    }

    # ---- 4. evaluasi saklar & sensor
    sw_states = {}
    for sw in switches:
        sw_states[sw.id] = eval_switch(sw, ctx)
        sw.state = sw_states[sw.id]
    ctx["switch_states"] = sw_states

    # ---- 5. evaluasi rule (prioritas besar dieksekusi terakhir = menang)
    overrides = {}
    fired = []
    for rule in sorted(Rule.query.filter_by(house_id=house.id, enabled=True).all(),
                       key=lambda x: x.priority or 0):
        if not eval_rule(rule, ctx):
            continue
        fired.append(rule.name)
        rule.last_fired = house.clock
        rule.fire_count = (rule.fire_count or 0) + 1
        for act in (rule.actions or []):
            target = act.get("target", "device")
            act_kind = act.get("act", "on")
            try:
                ref = int(act.get("ref"))
            except (TypeError, ValueError):
                ref = None
            if target == "device" and ref in comp_by_id:
                if act_kind == "toggle":
                    overrides[ref] = not comp_by_id[ref].state
                else:
                    overrides[ref] = (act_kind == "on")
            elif target == "switch" and ref in comp_by_id:
                s = comp_by_id[ref]
                s.manual_state = (act_kind == "on")
                if s.subtype in ("tombol", "saklar", "infrared", "pintu", "asap") \
                        or s.mode == "manual":
                    sw_states[s.id] = s.manual_state
                    s.state = s.manual_state
            elif target == "circuit" and ref in circ_by_id:
                circ_by_id[ref].tripped = (act_kind == "off")
            elif target == "all_devices":
                for d in devices:
                    overrides[d.id] = (act_kind == "on")

    # ---- 6. status perangkat
    for d in devices:
        d.state = resolve_device(d, sw_states, links_by_device, overrides)

    # ---- 7. kelistrikan: daya, arus, proteksi
    volt = house.voltage or 220.0
    pf = house.power_factor or 0.9
    src_by_id = {s.id: s for s in sources}

    for attempt in range(5):
        for c in circuits:
            src = src_by_id.get(c.source_id)
            energized = bool(src and src.enabled) and not c.tripped \
                and not house.main_tripped
            load = 0.0
            for d in devices:
                if d.circuit_id != c.id:
                    continue
                if not energized:
                    d.state = False
                    continue
                load += d.power_w if d.state else (d.standby_w or 0.0)
            for sw in switches:
                if sw.circuit_id == c.id and energized:
                    load += float((sw.params or {}).get("idle_w", 0.3))
            c.load_w = load
            c.current_a = load / (volt * pf) if volt else 0.0

        overloaded = [c for c in circuits
                      if not c.tripped and c.current_a > (c.mcb_a or 6.0)]
        if overloaded:
            for c in overloaded:
                c.tripped = True
                log(house, "MCB %s trip: %.2f A > %.1f A" %
                    (c.name, c.current_a, c.mcb_a), "alarm", "proteksi")
            continue

        total_w = sum(c.load_w for c in circuits if not c.tripped)
        total_a = total_w / (volt * pf) if volt else 0.0
        if not house.main_tripped and total_a > (house.main_mcb_a or 10.0):
            house.main_tripped = True
            log(house, "MCB utama trip: %.2f A > %.1f A" %
                (total_a, house.main_mcb_a), "alarm", "proteksi")
            continue
        break

    # kapasitas sumber
    for s in sources:
        beban = sum(c.load_w for c in circuits
                    if c.source_id == s.id and not c.tripped)
        s.state = s.enabled and beban > 0
        if s.enabled and s.capacity_w and beban > s.capacity_w:
            log(house, "Sumber %s kelebihan beban: %.0f W > %.0f W" %
                (s.name, beban, s.capacity_w), "warn", "sumber")

    total_w = sum(c.load_w for c in circuits if not c.tripped)
    total_a = total_w / (volt * pf) if volt else 0.0

    # ---- 8. energi
    house.energy_kwh = (house.energy_kwh or 0.0) + total_w * dt_h / 1000.0
    db.session.add(EnergySample(house_id=house.id, sim_time=house.clock,
                                minute=house.sim_minute, total_w=total_w,
                                current_a=total_a, kwh=house.energy_kwh))

    # tombol momentary kembali ke posisi semula
    for sw in switches:
        if (sw.params or {}).get("momentary") and sw.manual_state:
            sw.manual_state = False

    # ---- 9. pangkas riwayat
    n_log = EventLog.query.filter_by(house_id=house.id).count()
    if n_log > MAX_LOG:
        old = (EventLog.query.filter_by(house_id=house.id)
               .order_by(EventLog.id).limit(n_log - MAX_LOG).all())
        for o in old:
            db.session.delete(o)
    n_s = EnergySample.query.filter_by(house_id=house.id).count()
    if n_s > MAX_SAMPLES:
        old = (EnergySample.query.filter_by(house_id=house.id)
               .order_by(EnergySample.id).limit(n_s - MAX_SAMPLES).all())
        for o in old:
            db.session.delete(o)

    db.session.commit()
    return {
        "clock": house.clock, "day": house.sim_day,
        "time_of_day": band_of(house.hour), "weather": house.weather,
        "outdoor_temp": round(o_temp, 1), "outdoor_lux": round(o_lux, 0),
        "total_w": round(total_w, 1), "total_a": round(total_a, 2),
        "energy_kwh": round(house.energy_kwh, 4),
        "cost": round(house.energy_kwh * (house.tariff or 0), 0),
        "rules_fired": fired, "main_tripped": house.main_tripped,
    }


def snapshot(house):
    """Kumpulkan seluruh keadaan rumah untuk dikirim ke antarmuka."""
    rooms = Room.query.filter_by(house_id=house.id).order_by(Room.id).all()
    comps = Component.query.filter_by(house_id=house.id).order_by(Component.id).all()
    circuits = Circuit.query.filter_by(house_id=house.id).order_by(Circuit.id).all()
    rules = Rule.query.filter_by(house_id=house.id).order_by(Rule.priority, Rule.id).all()
    occupants = Occupant.query.filter_by(house_id=house.id).order_by(Occupant.id).all()
    ids = [c.id for c in comps]
    links = (SwitchLink.query.filter(SwitchLink.device_id.in_(ids)).all()
             if ids else [])
    logs = (EventLog.query.filter_by(house_id=house.id)
            .order_by(EventLog.id.desc()).limit(60).all())
    samples = (EnergySample.query.filter_by(house_id=house.id)
               .order_by(EnergySample.id.desc()).limit(120).all())

    hour = house.hour
    volt = house.voltage or 220.0
    pf = house.power_factor or 0.9
    total_w = sum(c.load_w for c in circuits if not c.tripped)

    return {
        "house": house.to_dict(),
        "env": {
            "hour": round(hour, 2), "clock": house.clock, "day": house.sim_day,
            "time_of_day": band_of(hour), "weather": house.weather,
            "outdoor_temp": round(outdoor_temp(hour, house.weather,
                                               house.outdoor_temp_offset or 0), 1),
            "outdoor_lux": round(outdoor_lux(hour, house.weather), 0),
            "outdoor_humidity": outdoor_humidity(house.weather),
        },
        "rooms": [r.to_dict() for r in rooms],
        "components": [c.to_dict() for c in comps],
        "circuits": [c.to_dict() for c in circuits],
        "links": [l.to_dict() for l in links],
        "rules": [r.to_dict() for r in rules],
        "occupants": [o.to_dict() for o in occupants],
        "logs": [l.to_dict() for l in logs],
        "samples": [s.to_dict() for s in reversed(samples)],
        "totals": {
            "power_w": round(total_w, 1),
            "current_a": round(total_w / (volt * pf), 2) if volt else 0,
            "energy_kwh": round(house.energy_kwh or 0, 4),
            "cost": round((house.energy_kwh or 0) * (house.tariff or 0), 0),
            "devices_on": sum(1 for c in comps if c.kind == "device" and c.state),
            "devices_total": sum(1 for c in comps if c.kind == "device"),
        },
    }
