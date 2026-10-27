"""Simulator Smart Home & Instalasi Listrik berbasis Flask.

Jalankan:  python app.py   lalu buka http://127.0.0.1:5000
"""
import json
import os

from flask import Flask, Response, jsonify, render_template, request

import simulation as sim
from models import (Circuit, Component, DEFAULT_POWER, DEFAULT_STANDBY,
                    DEVICE_TYPES, EnergySample, EventLog, House, Occupant,
                    Room, Rule, SOURCE_TYPES, SWITCH_TYPES, SwitchLink,
                    WEATHERS, db)
from seed import layout_components, seed_demo
from transfer import export_house, import_house, validate

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SCENARIO_DIR = os.path.join(BASE_DIR, "scenarios")


def create_app():
    app = Flask(__name__)
    os.makedirs(os.path.join(BASE_DIR, "instance"), exist_ok=True)
    # Lokasi basis data dapat diubah lewat variabel lingkungan SMARTHOME_DB
    db_path = os.environ.get(
        "SMARTHOME_DB", os.path.join(BASE_DIR, "instance", "smarthome.db"))
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.abspath(db_path)
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["JSON_SORT_KEYS"] = False
    db.init_app(app)

    with app.app_context():
        db.create_all()
        if not House.query.first():
            seed_demo()

    register_routes(app)
    return app


# ------------------------------------------------------------------ utils
def house_or_404():
    h = House.query.first()
    if h is None:
        h = seed_demo()
    return h


def payload():
    return request.get_json(silent=True) or {}


def assign(obj, data, fields):
    for f in fields:
        if f in data:
            setattr(obj, f, data[f])
    return obj


def num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ----------------------------------------------------------------- routes
def register_routes(app):

    # ------------------------------------------------------------ halaman
    @app.route("/")
    def page_simulator():
        return render_template("index.html", page="simulator")

    @app.route("/komponen")
    def page_components():
        return render_template("components.html", page="komponen")

    @app.route("/rule")
    def page_rules():
        return render_template("rules.html", page="rule")

    @app.route("/laporan")
    def page_report():
        return render_template("report.html", page="laporan")

    @app.route("/kasus")
    def page_cases():
        return render_template("cases.html", page="kasus")

    @app.route("/panduan")
    def page_help():
        return render_template("help.html", page="panduan")

    # ------------------------------------------------------------- katalog
    @app.get("/api/catalog")
    def api_catalog():
        return jsonify({
            "source_types": SOURCE_TYPES,
            "switch_types": SWITCH_TYPES,
            "device_types": DEVICE_TYPES,
            "weathers": list(WEATHERS.keys()),
            "default_power": DEFAULT_POWER,
            "default_standby": DEFAULT_STANDBY,
        })

    # --------------------------------------------------------------- state
    @app.get("/api/state")
    def api_state():
        return jsonify(sim.snapshot(house_or_404()))

    @app.put("/api/house")
    def api_house_update():
        h = house_or_404()
        d = payload()
        assign(h, d, ["name", "voltage", "power_factor", "tariff",
                      "main_mcb_a", "plan_w", "plan_h", "step_minutes",
                      "weather", "auto_weather", "outdoor_temp_offset",
                      "running", "sim_minute", "sim_day"])
        db.session.commit()
        return jsonify(h.to_dict())

    # ----------------------------------------------------------- simulasi
    @app.post("/api/sim/tick")
    def api_tick():
        h = house_or_404()
        steps = int(payload().get("steps", 1))
        out = None
        for _ in range(max(1, min(steps, 288))):
            out = sim.tick(h)
        return jsonify({"result": out, "state": sim.snapshot(h)})

    @app.post("/api/sim/recompute")
    def api_recompute():
        h = house_or_404()
        sim.tick(h, advance=False)
        return jsonify(sim.snapshot(h))

    @app.post("/api/sim/env")
    def api_env():
        h = house_or_404()
        d = payload()
        if "clock" in d:
            try:
                hh, mm = str(d["clock"]).split(":")
                h.sim_minute = int(hh) * 60 + int(mm)
            except ValueError:
                pass
        if "hour" in d:
            h.sim_minute = int(num(d["hour"]) * 60)
        if "time_of_day" in d:
            h.sim_minute = {"pagi": 7 * 60, "siang": 12 * 60,
                            "sore": 16 * 60, "malam": 20 * 60}.get(
                                d["time_of_day"], h.sim_minute)
        assign(h, d, ["weather", "auto_weather", "outdoor_temp_offset",
                      "step_minutes"])
        db.session.commit()
        sim.tick(h, advance=False)
        return jsonify(sim.snapshot(h))

    @app.post("/api/sim/reset-trip")
    def api_reset_trip():
        h = house_or_404()
        h.main_tripped = False
        for c in Circuit.query.filter_by(house_id=h.id):
            c.tripped = False
        sim.log(h, "Proteksi direset (MCB dinaikkan kembali)", "info", "proteksi")
        db.session.commit()
        sim.tick(h, advance=False)
        return jsonify(sim.snapshot(h))

    @app.post("/api/sim/reset-energy")
    def api_reset_energy():
        h = house_or_404()
        h.energy_kwh = 0.0
        EnergySample.query.filter_by(house_id=h.id).delete()
        db.session.commit()
        return jsonify(sim.snapshot(h))

    @app.post("/api/sim/reset-demo")
    def api_reset_demo():
        for model in (SwitchLink, EnergySample, EventLog, Rule, Occupant,
                      Component, Circuit, Room, House):
            model.query.delete()
        db.session.commit()
        seed_demo()
        return jsonify(sim.snapshot(house_or_404()))

    @app.post("/api/sim/clear")
    def api_clear():
        """Kosongkan rumah (hapus kamar & komponen) tanpa data contoh."""
        h = house_or_404()
        for model in (SwitchLink, EnergySample, EventLog, Rule, Occupant,
                      Component, Circuit, Room):
            if model is SwitchLink:
                SwitchLink.query.delete()
            else:
                model.query.filter_by(house_id=h.id).delete() \
                    if hasattr(model, "house_id") else model.query.delete()
        h.energy_kwh = 0.0
        h.main_tripped = False
        db.session.commit()
        return jsonify(sim.snapshot(h))

    # ------------------------------------------------------ ekspor & impor
    @app.get("/api/export")
    def api_export():
        h = house_or_404()
        doc = export_house(h, meta={
            "title": h.name,
            "description": request.args.get("description", ""),
        })
        body = json.dumps(doc, indent=1, ensure_ascii=False)
        name = "".join(ch if ch.isalnum() or ch in "-_" else "-"
                       for ch in (h.name or "rumah")).strip("-").lower()
        return Response(body, mimetype="application/json", headers={
            "Content-Disposition": 'attachment; filename="%s.json"' % (name or "rumah"),
        })

    @app.post("/api/import")
    def api_import():
        h = house_or_404()
        doc = payload()
        errors = validate(doc)
        if errors:
            return jsonify(error=" ".join(errors)), 400
        notes = import_house(h, doc)
        sim.log(h, "Konfigurasi diimpor: %s" % (doc.get("meta", {}).get("title")
                                                or "tanpa judul"), "info", "impor")
        db.session.commit()
        sim.tick(h, advance=False)
        return jsonify(notes=notes, state=sim.snapshot(h))

    # ----------------------------------------------------------- studi kasus
    def _scenario_files():
        if not os.path.isdir(SCENARIO_DIR):
            return []
        return sorted(f for f in os.listdir(SCENARIO_DIR)
                      if f.endswith(".json") and f != "index.json")

    def _read_scenario(slug):
        path = os.path.join(SCENARIO_DIR, slug + ".json")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    @app.get("/api/scenarios")
    def api_scenarios():
        out = []
        for fname in _scenario_files():
            slug = fname[:-5]
            try:
                with open(os.path.join(SCENARIO_DIR, fname), encoding="utf-8") as fh:
                    doc = json.load(fh)
            except (OSError, ValueError):
                continue
            meta = doc.get("meta", {})
            case = meta.get("case", {})
            out.append({
                "slug": slug,
                "title": meta.get("title", slug),
                "description": meta.get("description", ""),
                "case": case,
                "counts": {
                    "rooms": len(doc.get("rooms", [])),
                    "components": len(doc.get("components", [])),
                    "circuits": len(doc.get("circuits", [])),
                    "rules": len(doc.get("rules", [])),
                    "occupants": len(doc.get("occupants", [])),
                },
            })
        out.sort(key=lambda x: x["case"].get("kode", x["slug"]))
        return jsonify(out)

    @app.post("/api/scenarios/<slug>")
    def api_scenario_load(slug):
        doc = _read_scenario(slug)
        if doc is None:
            return jsonify(error="kasus tidak ditemukan"), 404
        h = house_or_404()
        errors = validate(doc)
        if errors:
            return jsonify(error=" ".join(errors)), 400
        notes = import_house(h, doc)
        sim.log(h, "Kasus dimuat: %s" % doc.get("meta", {}).get("title", slug),
                "info", "kasus")
        db.session.commit()
        sim.tick(h, advance=False)
        return jsonify(notes=notes, state=sim.snapshot(h))

    @app.get("/api/scenarios/<slug>/file")
    def api_scenario_file(slug):
        doc = _read_scenario(slug)
        if doc is None:
            return jsonify(error="kasus tidak ditemukan"), 404
        return Response(json.dumps(doc, indent=1, ensure_ascii=False),
                        mimetype="application/json",
                        headers={"Content-Disposition":
                                 'attachment; filename="%s.json"' % slug})

    @app.post("/api/layout")
    def api_layout():
        h = house_or_404()
        layout_components(h)
        db.session.commit()
        return jsonify(sim.snapshot(h))

    # -------------------------------------------------------------- kamar
    @app.post("/api/rooms")
    def api_room_create():
        h = house_or_404()
        d = payload()
        r = Room(house_id=h.id, name=d.get("name", "Kamar Baru"))
        assign(r, d, ["kind", "x", "y", "w", "h", "color", "window_factor",
                      "temp_c", "humidity"])
        db.session.add(r)
        db.session.commit()
        return jsonify(r.to_dict()), 201

    @app.put("/api/rooms/<int:rid>")
    def api_room_update(rid):
        r = db.session.get(Room, rid)
        if not r:
            return jsonify(error="not found"), 404
        assign(r, payload(), ["name", "kind", "x", "y", "w", "h", "color",
                              "window_factor", "temp_c", "humidity"])
        db.session.commit()
        return jsonify(r.to_dict())

    @app.delete("/api/rooms/<int:rid>")
    def api_room_delete(rid):
        r = db.session.get(Room, rid)
        if not r:
            return jsonify(error="not found"), 404
        for c in Component.query.filter_by(room_id=rid):
            c.room_id = None
        for o in Occupant.query.filter_by(room_id=rid):
            o.room_id, o.away = None, True
        db.session.delete(r)
        db.session.commit()
        return jsonify(ok=True)

    # ------------------------------------------------------------ sirkuit
    @app.post("/api/circuits")
    def api_circuit_create():
        h = house_or_404()
        d = payload()
        c = Circuit(house_id=h.id, name=d.get("name", "Grup Baru"))
        assign(c, d, ["source_id", "mcb_a", "color", "tripped"])
        db.session.add(c)
        db.session.commit()
        return jsonify(c.to_dict()), 201

    @app.put("/api/circuits/<int:cid>")
    def api_circuit_update(cid):
        c = db.session.get(Circuit, cid)
        if not c:
            return jsonify(error="not found"), 404
        assign(c, payload(), ["name", "source_id", "mcb_a", "color", "tripped"])
        db.session.commit()
        return jsonify(c.to_dict())

    @app.delete("/api/circuits/<int:cid>")
    def api_circuit_delete(cid):
        c = db.session.get(Circuit, cid)
        if not c:
            return jsonify(error="not found"), 404
        for comp in Component.query.filter_by(circuit_id=cid):
            comp.circuit_id = None
        db.session.delete(c)
        db.session.commit()
        return jsonify(ok=True)

    # ----------------------------------------------------------- komponen
    @app.post("/api/components")
    def api_component_create():
        h = house_or_404()
        d = payload()
        kind = d.get("kind", "device")
        subtype = d.get("subtype") or (
            "pln" if kind == "source" else "tombol" if kind == "switch"
            else "lampu")
        c = Component(house_id=h.id, kind=kind, subtype=subtype,
                      name=d.get("name") or subtype.replace("_", " ").title())
        c.power_w = num(d.get("power_w", DEFAULT_POWER.get(subtype, 0)))
        c.standby_w = num(d.get("standby_w", DEFAULT_STANDBY.get(subtype, 0)))
        assign(c, d, ["room_id", "circuit_id", "x", "y", "capacity_w", "mode",
                      "combine", "manual_state", "enabled", "params"])
        if c.params is None:
            c.params = {}
        db.session.add(c)
        db.session.commit()
        return jsonify(c.to_dict()), 201

    @app.put("/api/components/<int:cid>")
    def api_component_update(cid):
        c = db.session.get(Component, cid)
        if not c:
            return jsonify(error="not found"), 404
        d = payload()
        assign(c, d, ["name", "room_id", "circuit_id", "subtype", "x", "y",
                      "power_w", "standby_w", "capacity_w", "mode", "combine",
                      "manual_state", "enabled", "params"])
        db.session.commit()
        return jsonify(c.to_dict())

    @app.delete("/api/components/<int:cid>")
    def api_component_delete(cid):
        c = db.session.get(Component, cid)
        if not c:
            return jsonify(error="not found"), 404
        SwitchLink.query.filter(
            (SwitchLink.switch_id == cid) | (SwitchLink.device_id == cid)
        ).delete(synchronize_session=False)
        Circuit.query.filter_by(source_id=cid).update({"source_id": None})
        db.session.delete(c)
        db.session.commit()
        return jsonify(ok=True)

    @app.post("/api/components/<int:cid>/toggle")
    def api_component_toggle(cid):
        c = db.session.get(Component, cid)
        if not c:
            return jsonify(error="not found"), 404
        d = payload()
        c.manual_state = bool(d["value"]) if "value" in d else not c.manual_state
        if c.kind == "device" and c.mode == "auto" and not SwitchLink.query \
                .filter_by(device_id=cid).count():
            pass                      # perangkat tanpa saklar: pakai manual
        db.session.commit()
        h = house_or_404()
        sim.tick(h, advance=False)
        return jsonify(sim.snapshot(h))

    # ---------------------------------------------------------- pengkabelan
    @app.post("/api/links")
    def api_link_create():
        d = payload()
        try:
            sid, did = int(d["switch_id"]), int(d["device_id"])
        except (KeyError, TypeError, ValueError):
            return jsonify(error="switch_id & device_id wajib"), 400
        if SwitchLink.query.filter_by(switch_id=sid, device_id=did).first():
            return jsonify(error="sudah terhubung"), 409
        lk = SwitchLink(switch_id=sid, device_id=did,
                        inverted=bool(d.get("inverted", False)))
        db.session.add(lk)
        db.session.commit()
        return jsonify(lk.to_dict()), 201

    @app.put("/api/links/<int:lid>")
    def api_link_update(lid):
        lk = db.session.get(SwitchLink, lid)
        if not lk:
            return jsonify(error="not found"), 404
        assign(lk, payload(), ["inverted"])
        db.session.commit()
        return jsonify(lk.to_dict())

    @app.delete("/api/links/<int:lid>")
    def api_link_delete(lid):
        lk = db.session.get(SwitchLink, lid)
        if not lk:
            return jsonify(error="not found"), 404
        db.session.delete(lk)
        db.session.commit()
        return jsonify(ok=True)

    # --------------------------------------------------------------- rule
    @app.post("/api/rules")
    def api_rule_create():
        h = house_or_404()
        d = payload()
        r = Rule(house_id=h.id, name=d.get("name", "Rule Baru"),
                 conditions=d.get("conditions", []),
                 actions=d.get("actions", []))
        assign(r, d, ["enabled", "priority", "logic"])
        db.session.add(r)
        db.session.commit()
        return jsonify(r.to_dict()), 201

    @app.put("/api/rules/<int:rid>")
    def api_rule_update(rid):
        r = db.session.get(Rule, rid)
        if not r:
            return jsonify(error="not found"), 404
        assign(r, payload(), ["name", "enabled", "priority", "logic",
                              "conditions", "actions"])
        db.session.commit()
        return jsonify(r.to_dict())

    @app.delete("/api/rules/<int:rid>")
    def api_rule_delete(rid):
        r = db.session.get(Rule, rid)
        if not r:
            return jsonify(error="not found"), 404
        db.session.delete(r)
        db.session.commit()
        return jsonify(ok=True)

    @app.post("/api/rules/<int:rid>/test")
    def api_rule_test(rid):
        r = db.session.get(Rule, rid)
        if not r:
            return jsonify(error="not found"), 404
        h = house_or_404()
        snap = sim.snapshot(h)
        ctx = {
            "hour": snap["env"]["hour"],
            "time_of_day": snap["env"]["time_of_day"],
            "weather": snap["env"]["weather"],
            "outdoor_temp": snap["env"]["outdoor_temp"],
            "outdoor_lux": snap["env"]["outdoor_lux"],
            "occupancy": {rm["id"]: sum(1 for o in snap["occupants"]
                                        if o["room_id"] == rm["id"] and not o["away"])
                          for rm in snap["rooms"]},
            "occupant_total": sum(1 for o in snap["occupants"] if not o["away"]),
            "room_temp": {rm["id"]: rm["temp_c"] for rm in snap["rooms"]},
            "room_lux": {rm["id"]: rm["lux"] for rm in snap["rooms"]},
            "room_humidity": {rm["id"]: rm["humidity"] for rm in snap["rooms"]},
            "switch_states": {c["id"]: c["state"] for c in snap["components"]},
            "device_states": {c["id"]: c["state"] for c in snap["components"]},
            "total_w": snap["totals"]["power_w"],
            "energy_kwh": snap["totals"]["energy_kwh"],
        }
        detail = [{"cond": c, "value": sim.subject_value(c, ctx),
                   "pass": sim.compare(sim.subject_value(c, ctx),
                                       c.get("op", "=="), c.get("value"))}
                  for c in (r.conditions or [])]
        return jsonify(match=sim.eval_rule(r, ctx), detail=detail)

    # ----------------------------------------------------------- penghuni
    @app.post("/api/occupants")
    def api_occ_create():
        h = house_or_404()
        d = payload()
        o = Occupant(house_id=h.id, name=d.get("name", "Penghuni"))
        assign(o, d, ["role", "room_id", "activity", "auto_move", "away",
                      "schedule", "color"])
        db.session.add(o)
        db.session.commit()
        return jsonify(o.to_dict()), 201

    @app.put("/api/occupants/<int:oid>")
    def api_occ_update(oid):
        o = db.session.get(Occupant, oid)
        if not o:
            return jsonify(error="not found"), 404
        assign(o, payload(), ["name", "role", "room_id", "activity",
                              "auto_move", "away", "schedule", "color"])
        db.session.commit()
        h = house_or_404()
        sim.tick(h, advance=False)
        return jsonify(sim.snapshot(h))

    @app.delete("/api/occupants/<int:oid>")
    def api_occ_delete(oid):
        o = db.session.get(Occupant, oid)
        if not o:
            return jsonify(error="not found"), 404
        db.session.delete(o)
        db.session.commit()
        return jsonify(ok=True)

    # ---------------------------------------------------------------- log
    @app.delete("/api/logs")
    def api_logs_clear():
        EventLog.query.delete()
        db.session.commit()
        return jsonify(ok=True)


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
