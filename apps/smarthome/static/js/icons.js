/* Ikon SVG (viewBox 24x24) untuk tiap jenis komponen. */
const ICONS = {
  pln:'<path d="M13 2 4 14h6l-1 8 9-12h-6l1-8z"/>',
  genset:'<rect x="3" y="8" width="18" height="11" rx="2"/><path d="M7 8V5h10v3M6 12h5M15 12h3M6 15.5h3"/>',
  solar:'<path d="M3 4h18l2 9H1zM8 4 7 13M16 4l1 9M2 8.5h20M12 13v7M9 20h6"/>',
  baterai:'<rect x="3" y="7" width="15" height="10" rx="2"/><path d="M18 10h3v4h-3M7 12h5M9.5 9.5v5"/>',
  tombol:'<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="3.5"/>',
  saklar:'<rect x="4" y="7" width="16" height="10" rx="5"/><circle cx="9" cy="12" r="3"/>',
  pir:'<circle cx="8" cy="6" r="2.4"/><path d="M8 9c-2 0-3.4 1.6-3.4 3.5V20M14 6a7 7 0 0 1 0 12M17 4a10.5 10.5 0 0 1 0 16"/>',
  ldr:'<circle cx="12" cy="12" r="4"/><path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M19.1 4.9l-1.8 1.8M6.7 17.3l-1.8 1.8"/>',
  infrared:'<rect x="7" y="2" width="10" height="20" rx="2.5"/><circle cx="12" cy="5.8" r="1.2"/><path d="M9.6 10.5h4.8M9.6 13.5h4.8M9.6 16.5h4.8"/>',
  suhu:'<path d="M14 14.8V5a2 2 0 1 0-4 0v9.8a4 4 0 1 0 4 0z"/><path d="M12 9.5v5"/>',
  kelembaban:'<path d="M12 3s6 6.4 6 10.4A6 6 0 0 1 6 13.4C6 9.4 12 3 12 3z"/>',
  hujan:'<path d="M7 15a4 4 0 0 1 .6-8 5.5 5.5 0 0 1 10.4 1.6A3.5 3.5 0 0 1 17.4 15z"/><path d="M8 18l-1 3M12 18l-1 3M16 18l-1 3"/>',
  timer:'<circle cx="12" cy="13.5" r="7.5"/><path d="M12 9.5v4l2.5 1.8M9.5 2h5"/>',
  asap:'<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="2"/><path d="M12 4v2M12 18v2M4 12h2M18 12h2"/>',
  pintu:'<path d="M5.5 21V4a1 1 0 0 1 1-1h9a1 1 0 0 1 1 1v17M3 21h18"/><circle cx="13.5" cy="12.5" r="1"/>',
  lampu:'<path d="M9.5 18.5h5M10.5 21h3M12 3a6 6 0 0 0-3.5 10.9c.6.5.9 1.2.9 1.9v.7h5.2v-.7c0-.7.3-1.4.9-1.9A6 6 0 0 0 12 3z"/>',
  tv:'<rect x="2.5" y="4" width="19" height="13" rx="2"/><path d="M8 21h8M12 17v4"/>',
  ac:'<rect x="2.5" y="4" width="19" height="7" rx="2"/><path d="M6 8h12M7 14c1.5 0 1.5 2 3 2M13.5 14c1.5 0 1.5 2 3 2M7 17.5c1.5 0 1.5 2 3 2M13.5 17.5c1.5 0 1.5 2 3 2"/>',
  komputer:'<rect x="2.5" y="4" width="13" height="9.5" rx="1.5"/><path d="M6.5 17.5h5.5M9 13.5v4"/><rect x="17.5" y="4" width="4" height="13.5" rx="1"/>',
  mesin_cuci:'<rect x="4" y="2.5" width="16" height="19" rx="2"/><circle cx="12" cy="14" r="4.5"/><path d="M7 6.5h2M15.5 6.5h1.5"/>',
  kipas:'<circle cx="12" cy="12" r="2"/><path d="M12 10c0-3 1-5 3.4-5s2.6 3-.4 4.5S12 10 12 10zM14 12c3 0 5 1 5 3.4s-3 2.6-4.5-.4S14 12 14 12zM10 12c-3 0-5-1-5-3.4S8 6 9.5 9 10 12 10 12zM12 14c0 3-1 5-3.4 5S6 16 9 14.5 12 14 12 14z"/>',
  kulkas:'<rect x="5" y="2.5" width="14" height="19" rx="2"/><path d="M5 9.5h14M8 5.5v2.5M8 12.5v3"/>',
  pompa_air:'<circle cx="9" cy="12" r="5"/><path d="M14 12h6v6H4v-6M12 7V4h5"/>',
  microwave:'<rect x="2.5" y="5" width="19" height="13" rx="2"/><rect x="5" y="8" width="10" height="7" rx="1"/><path d="M18 9v1.5M18 12.5v1.5"/>',
  rice_cooker:'<path d="M4 10.5h16V16a4 4 0 0 1-4 4H8a4 4 0 0 1-4-4z"/><path d="M3 10.5h18M12 4v3.5M9.5 6l.6 1.3M14.5 6l-.6 1.3"/>',
  water_heater:'<rect x="6.5" y="2.5" width="11" height="19" rx="4.5"/><path d="M12 7v4.5M9.5 14.5h5"/>',
  router:'<rect x="3" y="13.5" width="18" height="6" rx="2"/><path d="M7 16.5h.01M10 16.5h.01M12 13.5V9.5M9 7.5a4 4 0 0 1 6 0M6.5 5a8 8 0 0 1 11 0"/>',
  cctv:'<path d="M3 7.5l13.5-4 1.8 6L4.8 13.5z"/><path d="M6.5 12.5 8 16.5M17 9.5l3.5 1M12 16.5v4.5M9 21h6"/>',
  charger:'<path d="M9 3v5M15 3v5M7 8h10v3a5 5 0 0 1-10 0zM12 16v5"/>',
  setrika:'<path d="M3 17.5h18a7 7 0 0 0-7-7H8a5 5 0 0 0-5 5z"/><path d="M6.5 13.5h6"/>',
  dispenser:'<rect x="5.5" y="8" width="13" height="13.5" rx="2"/><path d="M9 3h6l-1 5h-4zM12 12v3M9.5 18.5h5"/>',
  exhaust:'<rect x="3" y="3" width="18" height="18" rx="2.5"/><circle cx="12" cy="12" r="4"/><path d="M12 8l3.2 2M12 16l-3.2-2"/>',
  lainnya:'<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5v5l3 2"/>',
  person:'<circle cx="12" cy="6.5" r="3.2"/><path d="M5.5 21c0-3.6 2.9-6.5 6.5-6.5s6.5 2.9 6.5 6.5"/>'
};
function icon(sub){ return ICONS[sub] || ICONS.lainnya; }
function iconSvg(sub, size, cls){
  return '<svg viewBox="0 0 24 24" width="'+(size||18)+'" height="'+(size||18)+'" class="'+(cls||'')+'" '+
    'fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'+
    icon(sub)+'</svg>';
}
