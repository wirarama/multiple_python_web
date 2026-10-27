/* Penggambar denah rumah berbasis SVG + interaksi (pilih, geser, ubah ukuran). */
const SVGNS = "http://www.w3.org/2000/svg";

const Plan = {
  svg: null, state: null, mode: "view",
  opts: { wire: false, ctrl: true, label: true },
  sel: { type: null, id: null },
  pending: null,                 // komponen yang akan ditempel
  onSelect: null, onChange: null, onPlaced: null, onToggle: null,
  _drag: null,

  init(svgEl){
    this.svg = svgEl;
    svgEl.addEventListener("pointerdown", e => this._down(e));
    window.addEventListener("pointermove", e => this._move(e));
    window.addEventListener("pointerup", e => this._up(e));
    svgEl.addEventListener("click", e => this._click(e));
    svgEl.addEventListener("dblclick", e => this._dbl(e));
  },

  pt(evt){
    const m = this.svg.getScreenCTM().inverse();
    const p = this.svg.createSVGPoint(); p.x = evt.clientX; p.y = evt.clientY;
    const q = p.matrixTransform(m);
    return { x: q.x, y: q.y };
  },

  select(type, id){
    this.sel = { type, id };
    this.render(this.state);
    if(this.onSelect) this.onSelect(type, id);
  },

  roomAt(x, y){
    const rs = (this.state && this.state.rooms) || [];
    for(let i = rs.length - 1; i >= 0; i--){
      const r = rs[i];
      if(x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h) return r;
    }
    return null;
  },

  // ------------------------------------------------------------- interaksi
  _target(e){
    let n = e.target;
    while(n && n !== this.svg){
      if(n.dataset && n.dataset.kind) return n;
      n = n.parentNode;
    }
    return null;
  },
  _down(e){
    const n = this._target(e);
    if(!n) return;
    const kind = n.dataset.kind, id = +n.dataset.id;
    const p = this.pt(e);
    if(kind === "handle"){
      const r = this.state.rooms.find(x => x.id === id);
      this._drag = { kind:"resize", id, ox:p.x, oy:p.y, w0:r.w, h0:r.h, moved:false };
      e.preventDefault(); return;
    }
    if(kind === "room" && this.mode === "edit"){
      const r = this.state.rooms.find(x => x.id === id);
      this._drag = { kind:"room", id, ox:p.x, oy:p.y, x0:r.x, y0:r.y, moved:false };
      e.preventDefault(); return;
    }
    if(kind === "comp" && this.mode === "edit"){
      const c = this.state.components.find(x => x.id === id);
      this._drag = { kind:"comp", id, ox:p.x, oy:p.y, x0:c.x, y0:c.y, moved:false };
      e.preventDefault(); return;
    }
    if(kind === "person"){
      this._drag = { kind:"person", id, ox:p.x, oy:p.y, moved:false };
      e.preventDefault();
    }
  },
  _move(e){
    if(!this._drag) return;
    const p = this.pt(e), d = this._drag;
    const dx = p.x - d.ox, dy = p.y - d.oy;
    if(Math.abs(dx) > 2 || Math.abs(dy) > 2) d.moved = true;
    if(d.kind === "room"){
      const r = this.state.rooms.find(x => x.id === d.id);
      r.x = Math.max(0, Math.round((d.x0 + dx) / 5) * 5);
      r.y = Math.max(0, Math.round((d.y0 + dy) / 5) * 5);
    } else if(d.kind === "resize"){
      const r = this.state.rooms.find(x => x.id === d.id);
      r.w = Math.max(80, Math.round((d.w0 + dx) / 5) * 5);
      r.h = Math.max(70, Math.round((d.h0 + dy) / 5) * 5);
    } else if(d.kind === "comp"){
      const c = this.state.components.find(x => x.id === d.id);
      c.x = Math.round(d.x0 + dx); c.y = Math.round(d.y0 + dy);
    } else if(d.kind === "person"){
      const o = this.state.occupants.find(x => x.id === d.id);
      o._px = p.x; o._py = p.y;
    }
    this.render(this.state);
  },
  _up(){
    const d = this._drag; this._drag = null;
    if(!d || !d.moved || !this.onChange) return;
    if(d.kind === "room" || d.kind === "resize"){
      const r = this.state.rooms.find(x => x.id === d.id);
      this.onChange("room", d.id, { x:r.x, y:r.y, w:r.w, h:r.h });
    } else if(d.kind === "comp"){
      const c = this.state.components.find(x => x.id === d.id);
      const room = this.roomAt(c.x, c.y);
      this.onChange("component", d.id, { x:c.x, y:c.y, room_id: room ? room.id : null });
    } else if(d.kind === "person"){
      const o = this.state.occupants.find(x => x.id === d.id);
      const room = this.roomAt(o._px, o._py);
      this.onChange("occupant", d.id, room ? { room_id: room.id, away:false, auto_move:false }
                                           : { room_id:null, away:true, auto_move:false });
    }
  },
  _click(e){
    if(this._dragJustEnded) return;
    const n = this._target(e);
    if(this.mode === "place" && this.pending){
      const p = this.pt(e);
      if(this.onPlaced) this.onPlaced(p.x, p.y, this.roomAt(p.x, p.y));
      return;
    }
    if(!n){ this.select(null, null); return; }
    const kind = n.dataset.kind === "handle" ? "room" : n.dataset.kind;
    this.select(kind, +n.dataset.id);
  },
  _dbl(e){
    const n = this._target(e);
    if(!n || n.dataset.kind !== "comp") return;
    if(this.onToggle) this.onToggle(+n.dataset.id);
  },

  // ------------------------------------------------------------- rendering
  render(state){
    if(!state) return;
    this.state = state;
    const h = state.house, S = this.svg;
    S.setAttribute("viewBox", "0 0 " + h.plan_w + " " + h.plan_h);
    S.setAttribute("style", "max-height:78vh");
    while(S.firstChild) S.removeChild(S.firstChild);

    const circById = {}; state.circuits.forEach(c => circById[c.id] = c);
    const compById = {}; state.components.forEach(c => compById[c.id] = c);
    const roomById = {}; state.rooms.forEach(r => roomById[r.id] = r);

    const gRooms = mk("g"), gWire = mk("g"), gCtrl = mk("g"),
          gGlow = mk("g"), gComp = mk("g"), gPeople = mk("g");
    [gRooms, gGlow, gWire, gCtrl, gComp, gPeople].forEach(g => S.appendChild(g));

    // ---- kamar
    state.rooms.forEach(r => {
      const g = mk("g"); g.dataset.kind = "room"; g.dataset.id = r.id;
      if(this.mode === "edit") g.style.cursor = "move";
      const rect = mk("rect", { x:r.x, y:r.y, width:r.w, height:r.h, rx:8, fill:r.color,
        class: "room-rect" + (this.sel.type === "room" && this.sel.id === r.id ? " sel" : "") });
      g.appendChild(rect);
      g.appendChild(txt(r.x + 10, r.y + 19, r.name, "room-label"));
      g.appendChild(txt(r.x + 10, r.y + 34,
        Math.round(r.temp_c) + "°C · " + Math.round(r.lux) + " lx · " + Math.round(r.humidity) + "%",
        "room-meta"));
      gRooms.appendChild(g);
      if(this.mode === "edit"){
        const hd = mk("rect", { x:r.x + r.w - 12, y:r.y + r.h - 12, width:12, height:12, rx:3,
          fill:"#2563eb", opacity:.85, style:"cursor:nwse-resize" });
        hd.dataset.kind = "handle"; hd.dataset.id = r.id;
        gRooms.appendChild(hd);
      }
    });

    // ---- kotak panel listrik (komponen yang tidak berada di dalam kamar)
    const loose = state.components.filter(c => !c.room_id);
    if(loose.length){
      const xs = loose.map(c => c.x), ys = loose.map(c => c.y);
      const x0 = Math.min.apply(null, xs) - 42, x1 = Math.max.apply(null, xs) + 42;
      const y0 = Math.min.apply(null, ys) - 44, y1 = Math.max.apply(null, ys) + 42;
      gRooms.appendChild(mk("rect", { x:x0, y:y0, width:(x1-x0), height:(y1-y0), rx:8,
        fill:"#f8fafc", stroke:"#94a3b8", "stroke-width":1.6, "stroke-dasharray":"6 4" }));
      gRooms.appendChild(txt(x0 + 10, y0 + 19,
        LANG === "en" ? "Electrical Panel" : "Panel Listrik", "room-label"));
    }

    // ---- kabel daya: sumber -> perangkat (lewat warna grup)
    if(this.opts.wire){
      state.components.filter(c => c.kind !== "source" && c.circuit_id).forEach(c => {
        const ci = circById[c.circuit_id]; if(!ci) return;
        const src = compById[ci.source_id]; if(!src) return;
        const mx = (src.x + c.x) / 2;
        const hot = (this.sel.type === "circuit" && this.sel.id === ci.id) ||
                    (this.sel.type === "comp" && this.sel.id === c.id);
        const p = mk("path", {
          d: "M" + src.x + " " + src.y + " C " + mx + " " + src.y + ", " + mx + " " + c.y + ", " + c.x + " " + c.y,
          class: "wire", stroke: ci.tripped ? "#dc2626" : ci.color,
          "stroke-width": hot ? 2.6 : 1.3, opacity: hot ? .95 : .22,
          "stroke-dasharray": ci.tripped ? "5 4" : "none" });
        gWire.appendChild(p);
      });
    }
    // ---- jalur kendali: saklar -> perangkat
    if(this.opts.ctrl){
      state.links.forEach(l => {
        const s = compById[l.switch_id], d = compById[l.device_id];
        if(!s || !d) return;
        gCtrl.appendChild(mk("path", {
          d: "M" + s.x + " " + s.y + " Q " + ((s.x + d.x) / 2) + " " + (Math.min(s.y, d.y) - 26) +
             " " + d.x + " " + d.y, class: "ctrl-wire" }));
      });
    }

    // ---- komponen
    state.components.forEach(c => {
      if(c.kind === "device" && c.state && (c.subtype === "lampu")){
        gGlow.appendChild(mk("circle", { cx:c.x, cy:c.y, r:46, fill:"#fbbf24", opacity:.22 }));
      }
      const on = !!c.state;
      const g = mk("g", { class: "comp" + (on ? " on" : "") +
        (this.sel.type === "comp" && this.sel.id === c.id ? " sel" : "") });
      g.dataset.kind = "comp"; g.dataset.id = c.id;
      if(this.mode === "edit") g.style.cursor = "move";
      const r = c.kind === "source" ? 17 : 14;
      g.appendChild(mk("circle", { cx:c.x, cy:c.y, r:r, class:"ring" }));
      if(!c.enabled) g.appendChild(mk("line", { x1:c.x-r, y1:c.y+r, x2:c.x+r, y2:c.y-r,
        stroke:"#dc2626", "stroke-width":2 }));
      const sc = (r * 2 - 8) / 24;
      const ig = mk("g", { class:"ico",
        transform: "translate(" + (c.x - 12 * sc) + "," + (c.y - 12 * sc) + ") scale(" + sc + ")" });
      ig.innerHTML = icon(c.subtype);
      g.appendChild(ig);
      if(c.kind === "switch"){
        g.appendChild(mk("circle", { cx:c.x + r - 3, cy:c.y - r + 3, r:4,
          fill: on ? "#16a34a" : "#cbd5e1", stroke:"#fff", "stroke-width":1.2 }));
      }
      if(this.opts.label){
        const label = c.name.length > 14 ? c.name.slice(0, 13) + "…" : c.name;
        const tw = label.length * 4.6 + 8;
        g.appendChild(mk("rect", { x:c.x - tw/2, y:c.y + r + 2, width:tw, height:12, rx:5,
          fill:"#ffffff", opacity:.8 }));
        const tn = txt(c.x, c.y + r + 11, label, "cap"); tn.setAttribute("text-anchor", "middle");
        g.appendChild(tn);
      }
      const title = mk("title"); title.textContent =
        c.name + " — " + tt(c.subtype) + (c.kind === "device" ? " · " + c.power_w + " W" : "") +
        " · " + (on ? "ON" : "OFF");
      g.appendChild(title);
      gComp.appendChild(g);
    });

    // ---- penghuni
    const byRoom = {};
    state.occupants.forEach(o => {
      if(o.away || !o.room_id) return;
      (byRoom[o.room_id] = byRoom[o.room_id] || []).push(o);
    });
    Object.keys(byRoom).forEach(rid => {
      const room = roomById[rid]; if(!room) return;
      byRoom[rid].forEach((o, i) => {
        const x = (o._px != null && this._drag && this._drag.id === o.id) ? o._px
                : room.x + room.w / 2 + (i - (byRoom[rid].length - 1) / 2) * 34;
        const y = (o._py != null && this._drag && this._drag.id === o.id) ? o._py
                : room.y + room.h - 30;
        const g = mk("g", { class:"person" }); g.dataset.kind = "person"; g.dataset.id = o.id;
        g.appendChild(mk("circle", { cx:x, cy:y, r:13, fill:o.color, opacity:.92,
          stroke: (this.sel.type === "person" && this.sel.id === o.id) ? "#0f172a" : "#fff",
          "stroke-width":2 }));
        const ig = mk("g", { transform:"translate(" + (x - 8) + "," + (y - 8) + ") scale(" + (16/24) + ")",
          fill:"none", stroke:"#fff", "stroke-width":2.2, "stroke-linecap":"round" });
        ig.innerHTML = icon("person");
        g.appendChild(ig);
        const tn = txt(x, y + 26, o.name, "cap"); tn.setAttribute("text-anchor", "middle");
        g.appendChild(tn);
        const ti = mk("title"); ti.textContent = o.name + " — " + o.activity;
        g.appendChild(ti);
        gPeople.appendChild(g);
      });
    });
  }
};

function mk(tag, attrs){
  const e = document.createElementNS(SVGNS, tag);
  if(attrs) for(const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}
function txt(x, y, s, cls){
  const e = mk("text", { x:x, y:y, class:cls }); e.textContent = s; return e;
}
