"""Model data Simulator Smart Home & Instalasi Listrik.

Seluruh entitas disimpan pada SQLite melalui SQLAlchemy.
Data model for the Smart Home & Electrical Installation Simulator.
"""
from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# ---------------------------------------------------------------- konstanta
TIME_BANDS = ((5, 10, "pagi"), (10, 15, "siang"), (15, 18, "sore"))

WEATHERS = {
    # kode      : (faktor cahaya, offset suhu C, kelembaban %)
    "cerah":     (1.00, 1.5, 55.0),
    "berawan":   (0.55, 0.0, 70.0),
    "hujan":     (0.28, -2.5, 90.0),
    "badai":     (0.14, -3.5, 95.0),
    "kabut":     (0.40, -1.5, 88.0),
}

SOURCE_TYPES = ["pln", "genset", "solar", "baterai"]
SWITCH_TYPES = ["tombol", "saklar", "pir", "ldr", "infrared", "suhu",
                "kelembaban", "hujan", "timer", "asap", "pintu"]
DEVICE_TYPES = ["lampu", "tv", "ac", "komputer", "mesin_cuci", "kipas",
                "kulkas", "pompa_air", "microwave", "rice_cooker",
                "water_heater", "router", "cctv", "charger", "setrika",
                "dispenser", "exhaust", "lainnya"]

# daya default (W) supaya pengguna tidak perlu mengisi manual
DEFAULT_POWER = {
    "lampu": 12, "tv": 90, "ac": 750, "komputer": 200, "mesin_cuci": 400,
    "kipas": 55, "kulkas": 120, "pompa_air": 375, "microwave": 800,
    "rice_cooker": 400, "water_heater": 1200, "router": 12, "cctv": 8,
    "charger": 20, "setrika": 350, "dispenser": 350, "exhaust": 35,
    "lainnya": 50,
}
DEFAULT_STANDBY = {"tv": 1.5, "komputer": 3.0, "ac": 2.0, "microwave": 2.0,
                   "router": 5.0, "cctv": 3.0, "kulkas": 0.0}


def band_of(hour):
    """Kembalikan nama periode waktu (pagi/siang/sore/malam)."""
    for a, b, name in TIME_BANDS:
        if a <= hour < b:
            return name
    return "malam"


# ------------------------------------------------------------------- rumah
class House(db.Model):
    __tablename__ = "house"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), default="Rumah Simulasi")

    voltage = db.Column(db.Float, default=220.0)        # tegangan (V)
    power_factor = db.Column(db.Float, default=0.90)    # cos phi
    tariff = db.Column(db.Float, default=1444.70)       # Rp / kWh
    main_mcb_a = db.Column(db.Float, default=10.0)      # MCB utama (A)

    plan_w = db.Column(db.Integer, default=1040)
    plan_h = db.Column(db.Integer, default=660)

    sim_minute = db.Column(db.Integer, default=18 * 60)
    sim_day = db.Column(db.Integer, default=1)
    step_minutes = db.Column(db.Integer, default=10)

    weather = db.Column(db.String(20), default="cerah")
    auto_weather = db.Column(db.Boolean, default=False)
    outdoor_temp_offset = db.Column(db.Float, default=0.0)
    running = db.Column(db.Boolean, default=False)

    energy_kwh = db.Column(db.Float, default=0.0)
    main_tripped = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def hour(self):
        return (self.sim_minute % 1440) / 60.0

    @property
    def time_of_day(self):
        return band_of(self.hour)

    @property
    def clock(self):
        m = int(self.sim_minute) % 1440
        return "%02d:%02d" % (m // 60, m % 60)

    def to_dict(self):
        return dict(
            id=self.id, name=self.name, voltage=self.voltage,
            power_factor=self.power_factor, tariff=self.tariff,
            main_mcb_a=self.main_mcb_a, plan_w=self.plan_w, plan_h=self.plan_h,
            sim_minute=self.sim_minute, sim_day=self.sim_day,
            step_minutes=self.step_minutes, weather=self.weather,
            auto_weather=self.auto_weather,
            outdoor_temp_offset=self.outdoor_temp_offset,
            running=self.running, energy_kwh=round(self.energy_kwh, 5),
            cost=round(self.energy_kwh * self.tariff, 2),
            main_tripped=self.main_tripped,
            hour=round(self.hour, 2), clock=self.clock,
            time_of_day=self.time_of_day,
        )


# ------------------------------------------------------------------- kamar
class Room(db.Model):
    __tablename__ = "room"
    id = db.Column(db.Integer, primary_key=True)
    house_id = db.Column(db.Integer, db.ForeignKey("house.id"), nullable=False)
    name = db.Column(db.String(80), nullable=False)
    kind = db.Column(db.String(40), default="umum")
    x = db.Column(db.Float, default=40)
    y = db.Column(db.Float, default=40)
    w = db.Column(db.Float, default=220)
    h = db.Column(db.Float, default=170)
    color = db.Column(db.String(20), default="#e8eef7")
    window_factor = db.Column(db.Float, default=0.22)   # bagian cahaya luar masuk
    # keadaan dinamis
    temp_c = db.Column(db.Float, default=28.0)
    humidity = db.Column(db.Float, default=65.0)
    lux = db.Column(db.Float, default=0.0)

    def to_dict(self):
        return dict(id=self.id, house_id=self.house_id, name=self.name,
                    kind=self.kind, x=self.x, y=self.y, w=self.w, h=self.h,
                    color=self.color, window_factor=self.window_factor,
                    temp_c=round(self.temp_c, 2),
                    humidity=round(self.humidity, 1),
                    lux=round(self.lux, 1))


# ----------------------------------------------------------------- sirkuit
class Circuit(db.Model):
    """Grup instalasi / jalur MCB yang disuplai satu sumber listrik."""
    __tablename__ = "circuit"
    id = db.Column(db.Integer, primary_key=True)
    house_id = db.Column(db.Integer, db.ForeignKey("house.id"), nullable=False)
    name = db.Column(db.String(80), nullable=False)
    source_id = db.Column(db.Integer, db.ForeignKey("component.id"))
    mcb_a = db.Column(db.Float, default=6.0)
    color = db.Column(db.String(20), default="#f59e0b")
    tripped = db.Column(db.Boolean, default=False)
    # hasil perhitungan terakhir
    load_w = db.Column(db.Float, default=0.0)
    current_a = db.Column(db.Float, default=0.0)

    def to_dict(self):
        return dict(id=self.id, house_id=self.house_id, name=self.name,
                    source_id=self.source_id, mcb_a=self.mcb_a,
                    color=self.color, tripped=self.tripped,
                    load_w=round(self.load_w, 1),
                    current_a=round(self.current_a, 2))


# --------------------------------------------------------------- komponen
class Component(db.Model):
    """Sumber listrik, saklar/sensor, atau perangkat elektronik."""
    __tablename__ = "component"
    id = db.Column(db.Integer, primary_key=True)
    house_id = db.Column(db.Integer, db.ForeignKey("house.id"), nullable=False)
    room_id = db.Column(db.Integer, db.ForeignKey("room.id"))
    circuit_id = db.Column(db.Integer, db.ForeignKey("circuit.id"))

    kind = db.Column(db.String(20), nullable=False)      # source|switch|device
    subtype = db.Column(db.String(40), nullable=False)
    name = db.Column(db.String(80), nullable=False)
    x = db.Column(db.Float, default=0)
    y = db.Column(db.Float, default=0)

    power_w = db.Column(db.Float, default=0.0)
    standby_w = db.Column(db.Float, default=0.0)
    capacity_w = db.Column(db.Float, default=2200.0)     # khusus sumber

    mode = db.Column(db.String(12), default="auto")      # auto|manual|off
    combine = db.Column(db.String(12), default="parallel")  # parallel|series
    manual_state = db.Column(db.Boolean, default=False)
    state = db.Column(db.Boolean, default=False)         # hasil simulasi
    enabled = db.Column(db.Boolean, default=True)
    params = db.Column(db.JSON, default=dict)

    room = db.relationship("Room", backref="components")

    def to_dict(self):
        return dict(
            id=self.id, house_id=self.house_id, room_id=self.room_id,
            circuit_id=self.circuit_id, kind=self.kind, subtype=self.subtype,
            name=self.name, x=self.x, y=self.y, power_w=self.power_w,
            standby_w=self.standby_w, capacity_w=self.capacity_w,
            mode=self.mode, combine=self.combine,
            manual_state=self.manual_state, state=self.state,
            enabled=self.enabled, params=self.params or {},
        )


class SwitchLink(db.Model):
    """Hubungan saklar/sensor -> perangkat."""
    __tablename__ = "switch_link"
    id = db.Column(db.Integer, primary_key=True)
    switch_id = db.Column(db.Integer, db.ForeignKey("component.id"),
                          nullable=False)
    device_id = db.Column(db.Integer, db.ForeignKey("component.id"),
                          nullable=False)
    inverted = db.Column(db.Boolean, default=False)      # NC / logika terbalik

    def to_dict(self):
        return dict(id=self.id, switch_id=self.switch_id,
                    device_id=self.device_id, inverted=self.inverted)


# -------------------------------------------------------------------- rule
class Rule(db.Model):
    __tablename__ = "rule"
    id = db.Column(db.Integer, primary_key=True)
    house_id = db.Column(db.Integer, db.ForeignKey("house.id"), nullable=False)
    name = db.Column(db.String(120), nullable=False)
    enabled = db.Column(db.Boolean, default=True)
    priority = db.Column(db.Integer, default=10)
    logic = db.Column(db.String(4), default="AND")       # AND|OR
    conditions = db.Column(db.JSON, default=list)
    actions = db.Column(db.JSON, default=list)
    last_fired = db.Column(db.String(20), default="")
    fire_count = db.Column(db.Integer, default=0)

    def to_dict(self):
        return dict(id=self.id, house_id=self.house_id, name=self.name,
                    enabled=self.enabled, priority=self.priority,
                    logic=self.logic, conditions=self.conditions or [],
                    actions=self.actions or [], last_fired=self.last_fired,
                    fire_count=self.fire_count)


# ---------------------------------------------------------------- penghuni
class Occupant(db.Model):
    __tablename__ = "occupant"
    id = db.Column(db.Integer, primary_key=True)
    house_id = db.Column(db.Integer, db.ForeignKey("house.id"), nullable=False)
    name = db.Column(db.String(80), nullable=False)
    role = db.Column(db.String(40), default="penghuni")
    room_id = db.Column(db.Integer, db.ForeignKey("room.id"))
    activity = db.Column(db.String(40), default="santai")
    auto_move = db.Column(db.Boolean, default=True)
    away = db.Column(db.Boolean, default=False)
    schedule = db.Column(db.JSON, default=dict)          # {"7": room_id, ...}
    color = db.Column(db.String(20), default="#2563eb")

    def to_dict(self):
        return dict(id=self.id, house_id=self.house_id, name=self.name,
                    role=self.role, room_id=self.room_id,
                    activity=self.activity, auto_move=self.auto_move,
                    away=self.away, schedule=self.schedule or {},
                    color=self.color)


# ------------------------------------------------------------------- log
class EventLog(db.Model):
    __tablename__ = "event_log"
    id = db.Column(db.Integer, primary_key=True)
    house_id = db.Column(db.Integer, db.ForeignKey("house.id"), nullable=False)
    sim_time = db.Column(db.String(20), default="")
    level = db.Column(db.String(12), default="info")     # info|warn|alarm
    category = db.Column(db.String(24), default="sim")
    message = db.Column(db.String(300), default="")

    def to_dict(self):
        return dict(id=self.id, sim_time=self.sim_time, level=self.level,
                    category=self.category, message=self.message)


class EnergySample(db.Model):
    __tablename__ = "energy_sample"
    id = db.Column(db.Integer, primary_key=True)
    house_id = db.Column(db.Integer, db.ForeignKey("house.id"), nullable=False)
    sim_time = db.Column(db.String(20), default="")
    minute = db.Column(db.Integer, default=0)
    total_w = db.Column(db.Float, default=0.0)
    current_a = db.Column(db.Float, default=0.0)
    kwh = db.Column(db.Float, default=0.0)

    def to_dict(self):
        return dict(id=self.id, sim_time=self.sim_time, minute=self.minute,
                    total_w=round(self.total_w, 1),
                    current_a=round(self.current_a, 2),
                    kwh=round(self.kwh, 5))
