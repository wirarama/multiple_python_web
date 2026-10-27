"""Ekspor dan impor konfigurasi rumah dalam format JSON.

Dokumen JSON memuat seluruh isi rumah: properti rumah, kamar, sumber listrik,
grup instalasi, komponen, pengkabelan saklar, aturan otomasi, dan penghuni.
Seluruh pengenal (id) di dalam berkas bersifat lokal terhadap berkas itu dan
dipetakan ulang saat diimpor.
"""
from datetime import datetime

from models import (Circuit, Component, EnergySample, EventLog, House,
                    Occupant, Room, Rule, SwitchLink, db)

FORMAT = "smarthome-sim"
VERSION = 1

HOUSE_FIELDS = ["name", "voltage", "power_factor", "tariff", "main_mcb_a",
                "plan_w", "plan_h", "sim_minute", "sim_day", "step_minutes",
                "weather", "auto_weather", "outdoor_temp_offset",
                "energy_kwh", "main_tripped"]
ROOM_FIELDS = ["name", "kind", "x", "y", "w", "h", "color", "window_factor",
               "temp_c", "humidity", "lux"]
CIRCUIT_FIELDS = ["name", "mcb_a", "color", "tripped"]
COMP_FIELDS = ["kind", "subtype", "name", "x", "y", "power_w", "standby_w",
               "capacity_w", "mode", "combine", "manual_state", "state",
               "enabled", "params"]
RULE_FIELDS = ["name", "enabled", "priority", "logic", "conditions", "actions"]
OCC_FIELDS = ["name", "role", "activity", "auto_move", "away", "schedule", "color"]

# subjek syarat yang mengacu pada kamar / komponen
ROOM_SUBJECTS = {"room_temp", "room_lux", "room_humidity", "occupancy"}
COMP_SUBJECTS = {"sensor", "device"}


def _pick(obj, fields):
    return {f: getattr(obj, f) for f in fields}


# --------------------------------------------------------------------- ekspor
def export_house(house, meta=None):
    rooms = Room.query.filter_by(house_id=house.id).order_by(Room.id).all()
    circuits = Circuit.query.filter_by(house_id=house.id).order_by(Circuit.id).all()
    comps = Component.query.filter_by(house_id=house.id).order_by(Component.id).all()
    rules = Rule.query.filter_by(house_id=house.id).order_by(Rule.id).all()
    occupants = Occupant.query.filter_by(house_id=house.id).order_by(Occupant.id).all()
    ids = [c.id for c in comps]
    links = (SwitchLink.query.filter(SwitchLink.device_id.in_(ids)).all()
             if ids else [])

    doc = {
        "format": FORMAT,
        "version": VERSION,
        "exported_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "meta": meta or {"title": house.name, "description": ""},
        "house": _pick(house, HOUSE_FIELDS),
        "rooms": [dict(_pick(r, ROOM_FIELDS), id=r.id) for r in rooms],
        "components": [dict(_pick(c, COMP_FIELDS), id=c.id, room_id=c.room_id,
                            circuit_id=c.circuit_id) for c in comps],
        "circuits": [dict(_pick(c, CIRCUIT_FIELDS), id=c.id, source_id=c.source_id)
                     for c in circuits],
        "links": [{"switch_id": l.switch_id, "device_id": l.device_id,
                   "inverted": l.inverted} for l in links],
        "rules": [dict(_pick(r, RULE_FIELDS), id=r.id) for r in rules],
        "occupants": [dict(_pick(o, OCC_FIELDS), id=o.id, room_id=o.room_id)
                      for o in occupants],
    }
    return doc


# ---------------------------------------------------------------------- impor
def _as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def validate(doc):
    """Kembalikan daftar galat; kosong berarti dokumen dapat diimpor."""
    err = []
    if not isinstance(doc, dict):
        return ["Dokumen bukan objek JSON."]
    if doc.get("format") != FORMAT:
        err.append("Berkas ini bukan konfigurasi Simulator Smart Home "
                   "(kunci 'format' tidak cocok).")
    if _as_int(doc.get("version")) is None or int(doc["version"]) > VERSION:
        err.append("Versi format berkas tidak didukung.")
    for key in ("house", "rooms", "components", "circuits"):
        if key not in doc:
            err.append("Bagian '%s' tidak ditemukan pada berkas." % key)
    return err


def clear_house(house):
    """Kosongkan seluruh isi rumah, tetapi pertahankan baris rumah itu sendiri."""
    comp_ids = [c.id for c in Component.query.filter_by(house_id=house.id)]
    if comp_ids:
        SwitchLink.query.filter(
            SwitchLink.device_id.in_(comp_ids)).delete(synchronize_session=False)
        SwitchLink.query.filter(
            SwitchLink.switch_id.in_(comp_ids)).delete(synchronize_session=False)
    for model in (EnergySample, EventLog, Rule, Occupant, Component, Circuit, Room):
        model.query.filter_by(house_id=house.id).delete(synchronize_session=False)
    db.session.flush()


def import_house(house, doc):
    """Ganti seluruh isi rumah dengan isi dokumen. Kembalikan daftar catatan."""
    notes = []
    clear_house(house)

    # ---- properti rumah
    for f in HOUSE_FIELDS:
        if f in (doc.get("house") or {}):
            setattr(house, f, doc["house"][f])

    # ---- kamar
    room_map = {}
    for d in doc.get("rooms", []):
        r = Room(house_id=house.id)
        for f in ROOM_FIELDS:
            if f in d:
                setattr(r, f, d[f])
        db.session.add(r)
        db.session.flush()
        room_map[_as_int(d.get("id"))] = r.id

    # ---- komponen (circuit_id ditetapkan setelah grup dibuat)
    comp_map = {}
    for d in doc.get("components", []):
        c = Component(house_id=house.id, kind=d.get("kind", "device"),
                      subtype=d.get("subtype", "lainnya"),
                      name=d.get("name", "Komponen"))
        for f in COMP_FIELDS:
            if f in d:
                setattr(c, f, d[f])
        c.room_id = room_map.get(_as_int(d.get("room_id")))
        if c.params is None:
            c.params = {}
        db.session.add(c)
        db.session.flush()
        comp_map[_as_int(d.get("id"))] = c.id

    # ---- grup instalasi
    circ_map = {}
    for d in doc.get("circuits", []):
        c = Circuit(house_id=house.id, name=d.get("name", "Grup"))
        for f in CIRCUIT_FIELDS:
            if f in d:
                setattr(c, f, d[f])
        src = comp_map.get(_as_int(d.get("source_id")))
        if d.get("source_id") is not None and src is None:
            notes.append("Sumber listrik untuk grup '%s' tidak ditemukan." % c.name)
        c.source_id = src
        db.session.add(c)
        db.session.flush()
        circ_map[_as_int(d.get("id"))] = c.id

    for d in doc.get("components", []):
        new_id = comp_map.get(_as_int(d.get("id")))
        if new_id is None:
            continue
        comp = db.session.get(Component, new_id)
        comp.circuit_id = circ_map.get(_as_int(d.get("circuit_id")))

    # ---- pengkabelan
    for d in doc.get("links", []):
        s = comp_map.get(_as_int(d.get("switch_id")))
        t = comp_map.get(_as_int(d.get("device_id")))
        if s is None or t is None:
            notes.append("Satu hubungan saklar dilewati karena komponennya hilang.")
            continue
        db.session.add(SwitchLink(switch_id=s, device_id=t,
                                  inverted=bool(d.get("inverted", False))))

    # ---- aturan (acuan id dipetakan ulang)
    for d in doc.get("rules", []):
        conds = []
        for c in (d.get("conditions") or []):
            c = dict(c)
            ref = _as_int(c.get("ref"))
            if c.get("subject") in ROOM_SUBJECTS:
                c["ref"] = room_map.get(ref)
            elif c.get("subject") in COMP_SUBJECTS:
                c["ref"] = comp_map.get(ref)
            else:
                c["ref"] = None
            conds.append(c)
        acts = []
        for a in (d.get("actions") or []):
            a = dict(a)
            ref = _as_int(a.get("ref"))
            if a.get("target") in ("device", "switch"):
                a["ref"] = comp_map.get(ref)
            elif a.get("target") == "circuit":
                a["ref"] = circ_map.get(ref)
            else:
                a["ref"] = None
            acts.append(a)
        rule = Rule(house_id=house.id, name=d.get("name", "Rule"),
                    conditions=conds, actions=acts)
        for f in ("enabled", "priority", "logic"):
            if f in d:
                setattr(rule, f, d[f])
        db.session.add(rule)

    # ---- penghuni
    for d in doc.get("occupants", []):
        o = Occupant(house_id=house.id, name=d.get("name", "Penghuni"))
        for f in OCC_FIELDS:
            if f in d and f != "schedule":
                setattr(o, f, d[f])
        sched = {}
        for hour, target in (d.get("schedule") or {}).items():
            if target == "away":
                sched[str(hour)] = "away"
            else:
                mapped = room_map.get(_as_int(target))
                if mapped is not None:
                    sched[str(hour)] = mapped
        o.schedule = sched
        o.room_id = room_map.get(_as_int(d.get("room_id")))
        db.session.add(o)

    db.session.commit()
    return notes
