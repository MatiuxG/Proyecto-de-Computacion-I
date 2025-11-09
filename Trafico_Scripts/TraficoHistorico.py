# -*- coding: utf-8 -*-
"""
Tráfico (Madrid) — Datasheet normalizado (contrato común)
Estrategia:
- PM (mediciones): si hay coords, se convierten y se guardan en Lookups/pm_lookup.csv por idelem.
                   si NO hay coords, se busca ese idelem en la caché y se rellena lon/lat/distrito.
- Incidencias: si hay coords, se convierten y se mapea distrito; si no, quedará NA.
- Fallback pedido: si district_name sigue en NA, se rellena con 'location' para RapidMiner.
- Distritos: Lookups/distritos.geojson o conversión desde Lookups/distritos.kml (shapely).
- Salidas: Resultados/datasheet_trafico.csv + Resultados/debug_trafico_geo.csv
"""

import csv, re, math, json, requests, xml.etree.ElementTree as ET
from datetime import datetime, date as date_cls, timedelta
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# ===== Rutas
OUT_DIR = Path("./Trafico_Scripts/Resultados"); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "datasheet_trafico.csv"
DEBUG_FILE = OUT_DIR / "debug_trafico_geo.csv"

LOOKUPS = Path("./Lookups"); LOOKUPS.mkdir(parents=True, exist_ok=True)
PM_LK_FILE = LOOKUPS / "pm_lookup.csv"  # idelem;lon;lat;district_code;district_name

# ===== Contrato
CONTRACT = [
    "dataset","event_type","date","time","datetime",
    "district_code","district_name","lat","lon","location",
    "severity","value","units","source_id"
]

# ===== Ventana
TODAY = datetime.today().date()
START_DATE = TODAY - timedelta(days=61)  # ~2 meses

HEADERS = {
    "Accept": "application/xml, text/xml, */*",
    "User-Agent": "MadridTrafficETL/2.1"
}

INCID_XML = "https://informo.madrid.es/informo/tmadrid/incid_aytomadrid.xml"
PM_XML    = "https://informo.madrid.es/informo/tmadrid/pm.xml"

# ===== UTM30 → WGS84 (EPSG:25830/23030 aprox a WGS84)
_A = 6378137.0
_F = 1/298.257223563
_E2 = _F*(2-_F)
_E1 = (1 - math.sqrt(1-_E2)) / (1 + math.sqrt(1-_E2))
_K0 = 0.9996
_LON0_ZONE30 = math.radians(-3.0)

def utm30_to_wgs84(easting: float, northing: float, northern: bool = True) -> Tuple[float,float]:
    x = easting - 500000.0
    y = northing if northern else northing + 10000000.0
    M = y / _K0
    mu = M / (_A * (1 - _E2/4 - 3*_E2**2/64 - 5*_E2**3/256))
    e1 = _E1
    J1 = (3*e1/2 - 27*e1**3/32); J2 = (21*e1**2/16 - 55*e1**4/32)
    J3 = (151*e1**3/96); J4 = (1097*e1**4/512)
    fp = mu + J1*math.sin(2*mu) + J2*math.sin(4*mu) + J3*math.sin(6*mu) + J4*math.sin(8*mu)
    e2p = _E2/(1-_E2); C1 = e2p*math.cos(fp)**2; T1 = math.tan(fp)**2
    N1 = _A/math.sqrt(1-_E2*math.sin(fp)**2); R1 = N1*(1-_E2)/(1-_E2*math.sin(fp)**2)
    D = x/(N1*_K0)
    lat = fp - (N1*math.tan(fp)/R1)*((D**2)/2 - (5+3*T1+10*C1-4*C1**2-9*e2p)*(D**4)/24
                                     + (61+90*T1+298*C1+45*T1**2-252*e2p-3*C1**2)*(D**6)/720)
    lon = _LON0_ZONE30 + (D - (1+2*T1+C1)*(D**3)/6
                          + (5-2*C1+28*T1-3*C1**2+8*e2p+24*T1**2)*(D**5)/120)/math.cos(fp)
    return (math.degrees(lon), math.degrees(lat))

# ===== Distritos
def _parse_kml_polygons(kml_path: str):
    from shapely.geometry import Polygon, MultiPolygon
    try:
        tree = ET.parse(kml_path); root = tree.getroot()
    except Exception as e:
        print(f"[KML] error abriendo: {e}"); return []
    ns = {'kml':'http://www.opengis.net/kml/2.2'}
    feats = []
    for pm in root.findall('.//kml:Placemark', ns):
        name = (pm.find('kml:name', ns).text or "").strip() if pm.find('kml:name', ns) is not None else ""
        polys = []
        for poly in pm.findall('.//kml:Polygon', ns):
            coords_el = poly.find('.//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', ns)
            if coords_el is None or not coords_el.text: continue
            pts=[]
            for tok in coords_el.text.strip().replace("\n"," ").split():
                sp=tok.split(","); 
                if len(sp)>=2:
                    try: pts.append((float(sp[0]), float(sp[1])))
                    except: pass
            if len(pts)>=3:
                try: polys.append(Polygon(pts))
                except: pass
        if polys:
            geom = polys[0] if len(polys)==1 else MultiPolygon(polys)
            code=""; m=re.match(r"\s*(\d{1,2})\s+(.+)", name) if name else None
            dname=name
            if m: code=m.group(1).zfill(2); dname=m.group(2).strip()
            feats.append({'code':code,'name':dname,'geom':geom})
    if feats and not any(f['code'] for f in feats):
        feats.sort(key=lambda x: x['name']); 
        for i,f in enumerate(feats,1): f['code']=str(i).zfill(2)
    return feats

def ensure_distritos_geojson(lookups_dir: Path)->Optional[Path]:
    gj = lookups_dir/"distritos.geojson"
    if gj.exists(): return gj
    kml = lookups_dir/"distritos.kml"
    if not kml.exists():
        print("[Distritos] No hay distritos.geojson ni distritos.kml"); return None
    print("[KML] Convirtiendo distritos.kml → distritos.geojson…")
    feats=_parse_kml_polygons(str(kml))
    if not feats: 
        print("[KML] Sin polígonos"); return None
    try:
        from shapely.geometry import mapping
    except Exception:
        mapping=lambda g: g.__geo_interface__
    data={"type":"FeatureCollection","features":[]}
    for f in feats:
        data["features"].append({
            "type":"Feature",
            "properties":{"district_code":f['code'],"district_name":f['name']},
            "geometry":mapping(f['geom'])
        })
    with open(gj,"w",encoding="utf-8") as fo: json.dump(data,fo,ensure_ascii=False)
    print(f"[KML] Guardado {gj}"); return gj

from shapely.geometry import shape, Point

def load_districts(lookups_dir: Path):
    gj=ensure_distritos_geojson(lookups_dir)
    if not gj: return []
    try:
        with open(gj,"r",encoding="utf-8") as f: data=json.load(f)
        out=[]
        for ft in data.get("features",[]):
            pr=ft.get("properties",{}) or {}
            code=str(pr.get("district_code","")).zfill(2)
            name=str(pr.get("district_name","")).strip()
            geom=shape(ft.get("geometry"))
            out.append((code,name,geom))
        print(f"[Distritos] Cargados {len(out)} polígonos.")
        return out
    except Exception as e:
        print(f"[Distritos] Error: {e}"); return []

def point_to_district(lon:Optional[float], lat:Optional[float], districts)->Tuple[str,str]:
    if lon is None or lat is None or not districts: return ("NA","NA")
    p=Point(lon,lat)
    for code,name,poly in districts:
        try:
            if poly.contains(p) or poly.touches(p): return (code,name)
        except: pass
    return ("NA","NA")

# ===== PM Lookup (persistente por idelem)
def load_pm_lookup()->Dict[str,Dict[str,str]]:
    d={}
    if PM_LK_FILE.exists():
        with open(PM_LK_FILE,"r",encoding="utf-8") as f:
            rd=csv.DictReader(f,delimiter=";")
            for r in rd:
                idelem=r.get("idelem","").strip()
                if idelem:
                    d[idelem]={
                        "lon":r.get("lon",""),
                        "lat":r.get("lat",""),
                        "district_code":r.get("district_code",""),
                        "district_name":r.get("district_name",""),
                    }
    return d

def save_pm_lookup(d:Dict[str,Dict[str,str]]):
    with open(PM_LK_FILE,"w",newline="",encoding="utf-8") as f:
        wr=csv.writer(f,delimiter=";")
        wr.writerow(["idelem","lon","lat","district_code","district_name"])
        for k,v in sorted(d.items()):
            wr.writerow([k, v.get("lon",""), v.get("lat",""), v.get("district_code",""), v.get("district_name","")])

# ===== Util CSV salida
def write_contract(rows: List[Dict], out_path: Path):
    with open(out_path,"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=CONTRACT,delimiter=";",quoting=csv.QUOTE_NONE,escapechar="\\")
        w.writeheader()
        for r in rows:
            row={c:"" for c in CONTRACT}; row.update({k:("" if v is None else str(v)) for k,v in r.items()})
            row={k:("NA" if str(v).strip()=="" else v) for k,v in row.items()}
            w.writerow(row)

# ===== XML helpers
def fetch_xml(url:str)->Optional[ET.Element]:
    try:
        r=requests.get(url,headers=HEADERS,timeout=60); r.raise_for_status()
        return ET.fromstring(r.content)
    except Exception as e:
        print(f"[Red] {url} → {e}"); return None

def _first(el:ET.Element, keys:List[str])->str:
    low=[k.lower() for k in keys]
    for ch in el.iter():
        t=(ch.tag.split('}',1)[-1]).lower()
        if any(k in t for k in low):
            txt=(ch.text or "").strip()
            if txt: return txt
    return ""

def _alls(el:ET.Element, keys:List[str])->List[str]:
    out=[]; low=[k.lower() for k in keys]
    for ch in el.iter():
        t=(ch.tag.split('}',1)[-1]).lower()
        if any(k in t for k in low):
            txt=(ch.text or "").strip()
            if txt: out.append(txt)
    return out

def parse_dt(s:str)->Tuple[str,str,str]:
    if not s: return ("","","")
    s=s.replace("T"," ").replace("Z","").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M","%d/%m/%Y %H:%M","%d/%m/%Y","%Y-%m-%d"):
        try:
            dt=datetime.strptime(s,fmt)
            return (dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S"), dt.strftime("%Y-%m-%d %H:%M:%S"))
        except: pass
    return ("","","")

def in_window(date_str:str)->bool:
    if not date_str or date_str=="NA": return True
    try:
        d=datetime.strptime(date_str,"%Y-%m-%d").date()
        return START_DATE <= d <= TODAY
    except: return True

# ===== Fallback: district_name <- location si está en NA
def coalesce_district_with_location(rec: Dict) -> Dict:
    """Si district_name es NA/vacío, lo sustituye por 'location' para que RapidMiner tenga texto."""
    dname = (rec.get("district_name") or "").strip()
    if dname == "" or dname.upper() == "NA":
        loc = (rec.get("location") or "").strip()
        if loc:
            rec["district_name"] = loc
    return rec

# ===== Parsers
def parse_pm(root:ET.Element, districts, pm_cache:Dict[str,Dict[str,str]])->Tuple[List[Dict], Dict]:
    rows=[]; updated=False
    for node in root.iter():
        tag=node.tag.split('}',1)[-1].lower()
        if tag not in ("pm","pmitem","medicion","row","reporte","item"): continue

        idelem = _first(node, ["idelem","nombre","name","id","pm"]).strip()
        intensity = _first(node, ["intensidad","int","valor","value"])

        fh = _first(node, ["fechahora","fecha_hora","fecha","date"])
        d,t,dt = parse_dt(fh)

        xs = _alls(node, ["x","coorx","coordx","utm_x","st_x"])
        ys = _alls(node, ["y","coory","coordy","utm_y","st_y"])

        lon=lat=None
        # 1) coords en la fila
        try:
            if xs and ys:
                e=float(xs[0].replace(",",".")); n=float(ys[0].replace(",",".")); lon,lat = utm30_to_wgs84(e,n,True)
        except: lon=lat=None

        dcode=dname=""
        if lon is not None and lat is not None:
            dcode,dname = point_to_district(lon,lat,districts)
            # guarda/actualiza caché
            if idelem:
                prev = pm_cache.get(idelem,{})
                if (prev.get("lon")!=f"{lon:.6f}") or (prev.get("lat")!=f"{lat:.6f}") or (prev.get("district_code")!=dcode):
                    pm_cache[idelem] = {
                        "lon":f"{lon:.6f}","lat":f"{lat:.6f}",
                        "district_code":dcode,"district_name":dname
                    }
                    updated=True
        elif idelem and idelem in pm_cache:
            # 2) sin coords: usa caché
            rec = pm_cache[idelem]
            try:
                lon=float(rec.get("lon","")); lat=float(rec.get("lat",""))
                dcode=rec.get("district_code",""); dname=rec.get("district_name","")
            except:
                lon=lat=None

        row = {
            "dataset":"trafico",
            "event_type":"medicion",
            "date": d or "NA",
            "time": t or "NA",
            "datetime": dt or "NA",
            "district_code": dcode or "NA",
            "district_name": dname or "NA",
            "lat": (f"{lat:.6f}" if isinstance(lat,float) else "NA"),
            "lon": (f"{lon:.6f}" if isinstance(lon,float) else "NA"),
            "location": idelem or "NA",
            "severity": "NA",
            "value": intensity or "NA",
            "units": "veh/h",
            "source_id": PM_XML
        }
        # <<<<<<<<<<<<<<<< Fallback pedido
        row = coalesce_district_with_location(row)
        rows.append(row)
    # filtra por ventana
    rows = [r for r in rows if in_window(r["date"])]
    return rows, pm_cache if updated else pm_cache

def parse_incid(root:ET.Element, districts)->List[Dict]:
    rows=[]
    for node in root.iter():
        tag=node.tag.split('}',1)[-1].lower()
        if tag not in ("incidencia","incid","evento","row","item"): continue
        desc = _first(node, ["descripcion","texto","detalle","observ","title"])
        prioridad = _first(node, ["prioridad","nivel","severity","estado"])
        fh = _first(node, ["fechahora","fecha_hora","fecha","date"])
        d,t,dt = parse_dt(fh)
        xs = _alls(node, ["x","coorx","coordx","utm_x","st_x"])
        ys = _alls(node, ["y","coory","coordy","utm_y","st_y"])
        lon=lat=None
        try:
            if xs and ys:
                e=float(xs[0].replace(",",".")); n=float(ys[0].replace(",",".")); lon,lat=utm30_to_wgs84(e,n,True)
        except: lon=lat=None
        dcode,dname = point_to_district(lon,lat,districts)

        row = {
            "dataset":"trafico","event_type":"incidencia",
            "date": d or "NA","time": t or "NA","datetime": dt or "NA",
            "district_code": dcode or "NA","district_name": dname or "NA",
            "lat": (f"{lat:.6f}" if isinstance(lat,float) else "NA"),
            "lon": (f"{lon:.6f}" if isinstance(lon,float) else "NA"),
            "location": (desc or "NA"),
            "severity": (prioridad or "NA"),
            "value": "NA","units": "NA","source_id": INCID_XML
        }
        # <<<<<<<<<<<<<<<< Fallback pedido
        row = coalesce_district_with_location(row)
        rows.append(row)
    return [r for r in rows if in_window(r["date"])]

# ===== Debug report
def write_debug(rows:List[Dict]):
    with open(DEBUG_FILE,"w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f,delimiter=";"); 
        w.writerow(["event_type","location","has_coords","district_code","district_name","reason"])
        for r in rows[:2000]:  # tope
            has_coords = "YES" if r["lat"]!="NA" and r["lon"]!="NA" else "NO"
            reason = "no_coords" if has_coords=="NO" else ("no_match" if r["district_code"]=="NA" and r["district_name"]!="NA" and r["district_name"]==r["location"] else "")
            w.writerow([r["event_type"], r["location"], has_coords, r["district_code"], r["district_name"], reason])

# ===== Main
def main():
    print("=== DATASHEET TRÁFICO (Madrid) — ventana fija HOY-2m ===")
    print(f"[Ventana] {START_DATE} -> {TODAY}")

    from shapely import speedups
    try:
        if speedups.available: speedups.enable()
    except Exception:
        pass

    districts = load_districts(LOOKUPS)
    pm_cache = load_pm_lookup()

    all_rows: List[Dict] = []

    # Incidencias
    print(f"[Descarga] {INCID_XML}")
    r = fetch_xml(INCID_XML)
    if r is not None:
        rows = parse_incid(r, districts); print(f"  [Incidencias] {len(rows)} filas"); all_rows += rows
    else:
        print("  [Incidencias] sin datos")

    # PM
    print(f"[Descarga] {PM_XML}")
    r = fetch_xml(PM_XML)
    if r is not None:
        rows, pm_cache = parse_pm(r, districts, pm_cache); print(f"  [PM] {len(rows)} filas"); all_rows += rows
    else:
        print("  [PM] sin datos")

    # Guardados
    if pm_cache: save_pm_lookup(pm_cache)
    write_contract(all_rows, OUT_FILE)
    write_debug(all_rows)
    print(f"\n[OK] Datasheet: {OUT_FILE.resolve()} (rows={len(all_rows)})")
    print(f"[OK] Debug: {DEBUG_FILE.resolve()}")
    has_dist = sum(1 for r in all_rows if r["district_name"] not in ("","NA"))
    print(f"[Resumen] con_district_name(no NA)={has_dist} ; total={len(all_rows)}")

if __name__ == "__main__":
    main()
