"""
Fractal Model v2 - alertas com ficha de contexto (Telegram).
Leitura no 4H, execucao no 5m/15m. Dados publicos da Binance (+ Deribit).
Uso:  python fractal_model.py          (loop, 1 varredura por minuto)
      python fractal_model.py --once   (uma varredura e sai; usado no GitHub)
Variaveis: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS (opcional, separadas por virgula)
Interpretacao propria das regras do modelo. Nao e recomendacao de investimento.
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

# ============================ CONFIGURACAO ============================
SYMBOLS = [s.strip() for s in os.getenv(
    "SYMBOLS", "BTCUSDT,ETHUSDT,XAUUSDT,XAGUSDT").split(",") if s.strip()]
FUTURES_ONLY = {"XAUUSDT", "XAGUSDT"}   # ouro/prata: so no Futuros da Binance
HTF = "4h"                    # leitura (C1-C4)
EXEC_TFS = ["5m", "15m"]      # execucao (CISD, FVG)
POLL_SECONDS = 60
HTF_LOOKBACK = 6              # velas 4H recentes onde procurar modelos
PIVOT_N = 2                   # velas de cada lado p/ confirmar swing
FVG_MAX_AGE = 60              # velas de execucao em que o FVG ainda vale
SR_LOOKBACK = 120             # velas 4H usadas p/ suporte/resistencia
SR_PIVOT_N = 3
FLOW_BARS_15M = 12            # 12 x 15m = 3h de fluxo
DEPTH_PCT = 0.002             # janela do livro: +-0,2% (reduz se o livro for curto)
SEND_SUMMARY = True           # resumo a cada fechamento de vela 4H
TZ_OFFSET = -3                # horario de Brasilia
TZ_NAME = "BRT"
SESSIONS = {"Asia": (0, 8), "Londres": (8, 13), "NY": (13, 21)}  # UTC
STATE_FILE = "alertas_enviados.json"
LOG_FILE = "alertas_log.txt"
SPOT_BASES = ["https://api.binance.com", "https://data-api.binance.vision"]
FAPI = "https://fapi.binance.com"
DERIBIT = "https://www.deribit.com/api/v2"
# =====================================================================

LOCAL_TZ = timezone(timedelta(hours=TZ_OFFSET))
SILENT = False  # True na 1a execucao: registra o estado sem enviar alertas antigos


def fmt_t(ts):
    return f"{ts.tz_convert(LOCAL_TZ):%d/%m %H:%M} {TZ_NAME}"


def p2(x):
    return f"{x:.2f}"


def cisd_dir(side):
    return "BAIXA (venda)" if side == "baixa" else "ALTA (compra)"


# ------------------------------ REDE ---------------------------------
def http_json(url, params=None, timeout=15):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def first_json(urls, params=None):
    last = None
    for u in urls:
        try:
            return http_json(u, params)
        except Exception as e:
            last = e
    raise RuntimeError(f"{last}")


def safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:
        print(f"  (aviso) {fn.__name__} falhou: {str(e)[:90]}")
        return None


def fetch_klines(symbol, interval, limit=500):
    """Velas FECHADAS com volume e volume agressor comprador (tb)."""
    if symbol in FUTURES_ONLY:
        urls = [FAPI + "/fapi/v1/klines"]
    else:
        urls = [b + "/api/v3/klines" for b in SPOT_BASES]
    rows = first_json(urls, {"symbol": symbol, "interval": interval, "limit": limit})
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "vol",
                                     "ct", "qv", "n", "tb", "tq", "ig"])
    for c in ("open", "high", "low", "close", "vol", "tb"):
        df[c] = df[c].astype(float)
    df["time"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df[df["ct"] < int(time.time() * 1000)]
    return df[["time", "open", "high", "low", "close", "vol", "tb"]].reset_index(drop=True)


# ------------------------------ ALERTAS ------------------------------
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            return set(json.load(open(STATE_FILE)))
        except Exception:
            return set()
    return set()


def save_state(state):
    json.dump(sorted(state)[-3000:], open(STATE_FILE, "w"))


def tg_send(text):
    tok = (os.getenv("TELEGRAM_TOKEN") or "").strip()
    chat = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not (tok and chat):
        return None  # Telegram nao configurado
    try:
        r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          data={"chat_id": chat, "text": text}, timeout=15)
        if not r.ok:
            print("Telegram recusou:", r.text[:200])
        return r.ok
    except Exception as e:
        print("Telegram falhou:", e)
        return False


def send_alert(key, text, state, force=False):
    """1 envio por chave. 'text' pode ser funcao (so e montada se for enviar)."""
    if key in state:
        return
    state.add(key)
    if SILENT and not force:
        return
    msg = text() if callable(text) else text
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"[{stamp}] {msg}", flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {msg}\n")
    if tg_send(msg) is False:   # configurado mas falhou: tenta de novo na proxima
        state.discard(key)


# --------------------------- FRACTAL MODEL ---------------------------
def scan_fractal(df):
    """C1 define o range; C2 varre a maxima (minima) do C1 e fecha de volta
    dentro -> POI; C3 = deslocamento; C4 = continuacao."""
    models = []
    start = max(1, len(df) - HTF_LOOKBACK)
    for i in range(start, len(df)):
        c1, c2 = df.iloc[i - 1], df.iloc[i]
        for side in ("baixa", "alta"):
            if side == "baixa":
                ok = c2.high > c1.high and c2.close < c1.high and c2.close < c2.open
                poi = c1.high
            else:
                ok = c2.low < c1.low and c2.close > c1.low and c2.close > c2.open
                poi = c1.low
            if not ok:
                continue
            m = {"side": side, "poi": poi, "c2_time": c2.time,
                 "c2_low": c2.low, "c2_high": c2.high, "stage": "C2"}
            if i + 1 < len(df):
                c3 = df.iloc[i + 1]
                disp = (c3.close < c2.low) if side == "baixa" else (c3.close > c2.high)
                if disp:
                    m["stage"] = "C3"
                    if i + 2 < len(df):
                        c4 = df.iloc[i + 2]
                        cont = (c4.close < c3.low) if side == "baixa" else (c4.close > c3.high)
                        if cont:
                            m["stage"] = "C4"
                else:
                    m["stage"] = "C2 (sem deslocamento no C3)"
            models.append(m)
    return models


# ------------------------------- CISD --------------------------------
def find_cisd(df, side, from_time, poi):
    """Antecipado: fecha alem da abertura da ultima vela da sequencia.
    Confirmado: fecha alem da abertura da primeira vela da sequencia."""
    d = df[df.time >= from_time].reset_index(drop=True)
    res = {"early": None, "confirmed": None, "sweep_i": None, "extreme": None}
    if d.empty:
        return res, d
    bear = side == "baixa"
    swept = (d.high > poi) if bear else (d.low < poi)
    if not swept.any():
        return res, d
    s = int(swept.idxmax())
    res["sweep_i"] = s
    run_open = last_open = None
    prev_with = False
    extreme = d.high[s] if bear else d.low[s]
    for j in range(len(d)):
        o, c = d.open[j], d.close[j]
        with_trend = (c > o) if bear else (c < o)
        if with_trend:
            if not prev_with:
                run_open = o
            last_open = o
            prev_with = True
            if j >= s:
                extreme = max(extreme, d.high[j]) if bear else min(extreme, d.low[j])
            continue
        prev_with = False
        if j < s or run_open is None:
            continue
        broke_last = c < last_open if bear else c > last_open
        broke_first = c < run_open if bear else c > run_open
        if broke_last and res["early"] is None:
            res["early"] = j
            res["early_level"] = last_open
            res["early_close"] = c
        if broke_first and res["confirmed"] is None:
            res["confirmed"] = j
            res["conf_level"] = run_open
            res["conf_close"] = c
            res["extreme"] = extreme
            break
    return res, d


def find_protected_swing(d, side, conf_i, extreme):
    n = PIVOT_N
    for p in range(conf_i + 1, len(d) - n):
        if side == "baixa":
            win = d.high[p - n:p + n + 1]
            if d.high[p] == win.max() and d.high[p] < extreme:
                return p, d.high[p]
        else:
            win = d.low[p - n:p + n + 1]
            if d.low[p] == win.min() and d.low[p] > extreme:
                return p, d.low[p]
    return None, None


# ------------------------------- NIVEIS ------------------------------
def session_levels(df15):
    out = {}
    now = datetime.now(timezone.utc)
    for name, (h0, h1) in SESSIONS.items():
        d = df15[(df15.time.dt.hour >= h0) & (df15.time.dt.hour < h1)]
        if d.empty:
            continue
        d = d.assign(day=d.time.dt.date)
        today = now.date()
        done = [x for x in sorted(d.day.unique())
                if x < today or (x == today and now.hour >= h1)]
        if not done:
            continue
        g = d[d.day == done[-1]]
        out[f"Max {name}"] = g.high.max()
        out[f"Min {name}"] = g.low.min()
    return out


def key_levels(sym):
    levels = {}
    try:
        d1 = fetch_klines(sym, "1d", 5)
        levels["PDH"], levels["PDL"] = d1.high.iloc[-1], d1.low.iloc[-1]
        w1 = fetch_klines(sym, "1w", 5)
        levels["PWH"], levels["PWL"] = w1.high.iloc[-1], w1.low.iloc[-1]
    except Exception as e:
        print("Niveis D/W falharam:", str(e)[:90])
    return levels


def active_fvgs(df):
    fv = []
    start = max(2, len(df) - FVG_MAX_AGE)
    for i in range(start, len(df)):
        a, c = df.iloc[i - 2], df.iloc[i]
        if c.low > a.high:
            fv.append(("alta", a.high, c.low, df.time[i], i))
        elif c.high < a.low:
            fv.append(("baixa", c.high, a.low, df.time[i], i))
    ativos = []
    for kind, lo, hi, t, i in fv:
        later = df.iloc[i + 1:]
        if kind == "alta" and (later.low <= lo).any():
            continue
        if kind == "baixa" and (later.high >= hi).any():
            continue
        ativos.append((kind, lo, hi, t, i))
    return ativos


# ----------------------- CONTEXTO: ESTRUTURA / S-R --------------------
def pivots(df, n):
    h, l = df.high.values, df.low.values
    hi, lo = [], []
    for i in range(n, len(df) - n):
        if h[i] == h[i - n:i + n + 1].max():
            hi.append((i, float(h[i])))
        if l[i] == l[i - n:i + n + 1].min():
            lo.append((i, float(l[i])))
    return hi, lo


def trend_4h(df):
    d = df.tail(SR_LOOKBACK).reset_index(drop=True)
    hi, lo = pivots(d, PIVOT_N)
    if len(hi) < 2 or len(lo) < 2:
        return "lateral", "poucos dados"
    hh, hl = hi[-1][1] > hi[-2][1], lo[-1][1] > lo[-2][1]
    if hh and hl:
        return "alta", "topos e fundos ascendentes"
    if not hh and not hl:
        return "baixa", "topos e fundos descendentes"
    return "lateral", "estrutura mista"


def atr(df, n=14):
    tr = pd.concat([df.high - df.low,
                    (df.high - df.close.shift()).abs(),
                    (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    v = tr.rolling(n).mean().iloc[-1]
    return float(v) if v == v else None


def sr_zones(df, price):
    """Zonas de suporte/resistencia = agrupamento de pivos do 4H."""
    d = df.tail(SR_LOOKBACK).reset_index(drop=True)
    hi, lo = pivots(d, SR_PIVOT_N)
    pts = sorted([p for _, p in hi] + [p for _, p in lo])
    tol = 0.5 * (atr(d) or price * 0.01)
    clusters = []
    for p in pts:
        if clusters and p - clusters[-1][-1] <= tol and p - clusters[-1][0] <= 2 * tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    zones = [(min(c), max(c), len(c)) for c in clusters]
    above = [z for z in zones if z[0] > price]
    below = [z for z in zones if z[1] < price]
    res = min(above, key=lambda z: z[0] - price) if above else None
    sup = max(below, key=lambda z: z[1]) if below else None
    return sup, res


def zone_txt(z):
    if z is None:
        return "n/d"
    a, b, n = z
    rng = p2(a) if abs(b - a) < 1e-9 else f"{p2(a)}-{p2(b)}"
    return f"{rng} ({n} toque{'s' if n > 1 else ''})"


# --------------------- CONTEXTO: FLUXO / LIVRO / DERIVATIVOS ----------
def flow_pct(df, n):
    d = df.tail(n)
    vol = d.vol.sum()
    if vol <= 0:
        return None
    b = d.tb.sum() / vol * 100
    return b, 100 - b


def flow_tag(buy):
    if buy >= 55:
        return "pressao compradora"
    if buy <= 45:
        return "pressao vendedora"
    return "equilibrado"


def book_imbalance(sym):
    if sym in FUTURES_ONLY:
        urls = [FAPI + "/fapi/v1/depth"]
    else:
        urls = [b + "/api/v3/depth" for b in SPOT_BASES]
    j = first_json(urls, {"symbol": sym, "limit": 1000})
    bids = [(float(p), float(q)) for p, q in j["bids"]]
    asks = [(float(p), float(q)) for p, q in j["asks"]]
    mid = (bids[0][0] + asks[0][0]) / 2
    w = min(DEPTH_PCT * mid, mid - bids[-1][0], asks[-1][0] - mid)
    if w <= 0:
        return None
    bv = sum(q for p, q in bids if p >= mid - w)
    av = sum(q for p, q in asks if p <= mid + w)
    tot = bv + av
    if tot <= 0:
        return None
    return bv / tot * 100, av / tot * 100, w / mid * 100


def derivs(sym):
    out = {}

    def part(key, fn):
        try:
            out[key] = fn()
        except Exception:
            pass

    part("funding", lambda: float(http_json(
        FAPI + "/fapi/v1/premiumIndex", {"symbol": sym})["lastFundingRate"]) * 100)
    part("oi", lambda: float(http_json(
        FAPI + "/fapi/v1/openInterest", {"symbol": sym})["openInterest"]))

    def oi_chg():
        h = http_json(FAPI + "/futures/data/openInterestHist",
                      {"symbol": sym, "period": "1h", "limit": 5})
        return (float(h[-1]["sumOpenInterest"]) / float(h[0]["sumOpenInterest"]) - 1) * 100
    part("oi_chg", oi_chg)
    part("ls", lambda: float(http_json(
        FAPI + "/futures/data/globalLongShortAccountRatio",
        {"symbol": sym, "period": "1h", "limit": 1})[-1]["longShortRatio"]))

    cur = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}.get(sym)
    if "funding" not in out and cur:   # plano B (Binance Futures bloqueada na regiao)
        try:
            t = http_json(f"{DERIBIT}/public/ticker",
                          {"instrument_name": f"{cur}-PERPETUAL"})["result"]
            out["funding"] = float(t.get("funding_8h", t.get("current_funding"))) * 100
            out["src"] = "Deribit"
        except Exception:
            pass
    return out


def max_pain(cur):
    rows = http_json(f"{DERIBIT}/public/get_book_summary_by_currency",
                     {"currency": cur, "kind": "option"}, timeout=25)["result"]
    now = datetime.now(timezone.utc)
    by = {}
    for r in rows:
        parts = r["instrument_name"].split("-")
        if len(parts) != 4:
            continue
        _, exp, strike, typ = parts
        try:
            edt = datetime.strptime(exp, "%d%b%y").replace(hour=8, tzinfo=timezone.utc)
        except ValueError:
            continue
        if edt <= now:
            continue
        by.setdefault(edt, []).append((float(strike), typ, float(r.get("open_interest") or 0)))
    if not by:
        return None
    tot = {e: sum(x[2] for x in v) for e, v in by.items()}
    big = max(tot.values())
    if big <= 0:
        return None
    chosen = next(e for e in sorted(by) if tot[e] >= 0.1 * big)
    legs = by[chosen]
    best = None
    for S in sorted({k for k, _, _ in legs}):
        pay = sum((max(0, S - k) if t == "C" else max(0, k - S)) * oi for k, t, oi in legs)
        if best is None or pay < best[1]:
            best = (S, pay)
    return {"expiry": chosen, "strike": best[0]}


# ------------------------------ FICHA --------------------------------
def build_card(sym, htf, ex15, title):
    price = float(ex15.close.iloc[-1])
    tr, why = trend_4h(htf)
    lines = [title, f"Preco: {p2(price)} | Tendencia 4H: {tr.upper()} ({why})", ""]

    f15, f24 = flow_pct(ex15, FLOW_BARS_15M), flow_pct(htf, 6)
    if f15:
        lines.append(f"Fluxo agressor 3h: compra {f15[0]:.0f}% | venda {f15[1]:.0f}% ({flow_tag(f15[0])})")
    if f24:
        lines.append(f"Fluxo agressor 24h: compra {f24[0]:.0f}% | venda {f24[1]:.0f}% ({flow_tag(f24[0])})")
    bk = safe(book_imbalance, sym)
    if bk:
        lines.append(f"Livro de ofertas (+-{bk[2]:.2f}%): compra {bk[0]:.0f}% | venda {bk[1]:.0f}% ({flow_tag(bk[0])})")

    dv = safe(derivs, sym) or {}
    if "funding" in dv:
        who = "longs pagam" if dv["funding"] > 0 else "shorts pagam"
        src = f" [{dv['src']}]" if dv.get("src") else ""
        lines.append(f"Funding: {dv['funding']:+.4f}% ({who}){src}")
    if "oi" in dv:
        chg = f" ({dv['oi_chg']:+.1f}% em 4h)" if "oi_chg" in dv else ""
        lines.append(f"Open Interest: {dv['oi']:,.0f}{chg}")
    if "ls" in dv:
        lines.append(f"Contas long/short: {dv['ls']:.2f}")

    cur = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}.get(sym)
    if cur:
        mp = safe(max_pain, cur)
        if mp:
            dist = (price / mp["strike"] - 1) * 100
            lines.append(f"Max Pain opcoes ({mp['expiry']:%d/%m}): {mp['strike']:.0f} "
                         f"(preco {dist:+.1f}% vs nivel)")

    sup, res = sr_zones(htf, price)
    lines += ["", f"Zona de venda / resistencia: {zone_txt(res)}",
              f"Zona de compra / suporte: {zone_txt(sup)}"]
    if tr == "alta":
        lines.append("Vies: COMPRA em recuo ate a zona de suporte (a favor da tendencia)")
    elif tr == "baixa":
        lines.append("Vies: VENDA em repique ate a zona de resistencia (a favor da tendencia)")
    else:
        lines.append("Vies: sem tendencia clara, priorizar as extremidades do range")
    lines.append("(contexto estatistico, nao e recomendacao)")
    return "\n".join(lines)


# ----------------------------- VARREDURA -----------------------------
def scan_symbol(sym, state):
    htf = fetch_klines(sym, HTF, 200)
    models = scan_fractal(htf)
    levels = key_levels(sym)
    tfs = list(dict.fromkeys(EXEC_TFS + ["15m"]))
    ex = {tf: fetch_klines(sym, tf, 500) for tf in tfs}
    ex15 = ex["15m"]

    cache = {}

    def card(title):
        if "c" not in cache:
            cache["c"] = build_card(sym, htf, ex15, title)
        return cache["c"]

    # resumo a cada fechamento de vela 4H
    if SEND_SUMMARY:
        k = f"{sym}:resumo:{htf.time.iloc[-1]:%Y%m%d%H}"
        send_alert(k, lambda: build_card(
            sym, htf, ex15, f"RESUMO {HTF} - {sym} (vela fechada {fmt_t(htf.time.iloc[-1])})"),
            state, force=True)

    for m in models:
        tag = f"{sym}:{HTF}:{m['side']}:{m['c2_time']:%Y%m%d%H}"
        arrow = "🔻" if m["side"] == "baixa" else "🔺"
        vies = "VENDA" if m["side"] == "baixa" else "COMPRA"
        head = (f"{arrow} {sym} {HTF}: C2 varreu o range do C1 e fechou de volta no POI "
                f"({p2(m['poi'])}) - modelo de {m['side']} (vies de {vies})")
        send_alert(f"{tag}:C2", lambda h=head: h + "\n\n" + card("CONTEXTO"), state)
        if m["stage"] in ("C3", "C4"):
            send_alert(f"{tag}:C3", f"{arrow} {sym}: C3 de deslocamento confirmado ({m['side']})", state)
        if m["stage"] == "C4":
            send_alert(f"{tag}:C4", f"{arrow} {sym}: C4 de continuacao confirmado ({m['side']})", state)

    for tf in EXEC_TFS:
        e = ex[tf]
        last = e.iloc[-1]
        for m in models:
            tag = f"{sym}:{tf}:{m['side']}:{m['c2_time']:%Y%m%d%H}"
            res, d = find_cisd(e, m["side"], m["c2_time"], m["poi"])
            if res["early"] is not None:
                t = d.time[res["early"]]
                lado = "abaixo" if m["side"] == "baixa" else "acima"
                send_alert(f"{tag}:cisd_early",
                           f"⚡ {sym} {tf}: CISD ANTECIPADO de {cisd_dir(m['side'])} - fechou "
                           f"{p2(res['early_close'])} {lado} de {p2(res['early_level'])} "
                           f"(abertura da ultima vela da sequencia) as {fmt_t(t)}", state)
            if res["confirmed"] is None:
                continue
            t = d.time[res["confirmed"]]
            vies = "VENDA" if m["side"] == "baixa" else "COMPRA"
            lado = "abaixo" if m["side"] == "baixa" else "acima"
            head = (f"✅ {sym} {tf}: CISD CONFIRMADO de {cisd_dir(m['side'])} - fechou "
                    f"{p2(res['conf_close'])} {lado} de {p2(res['conf_level'])} "
                    f"(abertura da 1a vela da sequencia) as {fmt_t(t)}")
            send_alert(f"{tag}:cisd_conf", lambda h=head: h + "\n\n" + card("CONTEXTO"), state)
            p, lvl = find_protected_swing(d, m["side"], res["confirmed"], res["extreme"])
            if p is not None:
                send_alert(f"{tag}:swing",
                           f"🛡️ {sym} {tf}: swing protegido formado em {p2(lvl)} ({m['side']})", state)
            # reteste em FVG a favor do modelo, depois da varredura
            sweep_t = d.time[res["sweep_i"]]
            for kind, lo, hi, ft, _i in active_fvgs(e.iloc[:-1]):
                if kind == m["side"] and ft >= sweep_t and last.low <= hi and last.high >= lo:
                    send_alert(f"{sym}:{tf}:fvg:{ft:%Y%m%d%H%M}:{m['c2_time']:%Y%m%d%H}",
                               f"🧲 {sym} {tf}: reteste em FVG de {kind} ({p2(lo)}-{p2(hi)}) "
                               f"apos CISD - zona de entrada do modelo", state)

    # toques em niveis-chave (uma vez por nivel por dia)
    l15 = ex15.iloc[-1]
    alls = dict(levels)
    alls.update(session_levels(ex15))
    for name, lvl in alls.items():
        if l15.low <= lvl <= l15.high:
            send_alert(f"{sym}:touch:{name}:{l15.time:%Y%m%d}",
                       f"📍 {sym}: tocou {name} ({p2(lvl)})", state)


def main():
    global SILENT
    once = "--once" in sys.argv
    first_run = not os.path.exists(STATE_FILE)
    state = load_state()
    if first_run:
        SILENT = True
        print("Primeira execucao: registrando o estado atual sem enviar alertas antigos.")
    print(f"Fractal Model v2: {SYMBOLS} | leitura {HTF} | execucao {EXEC_TFS}")
    while True:
        for sym in SYMBOLS:
            try:
                scan_symbol(sym, state)
            except Exception as e:
                print(f"Erro em {sym}: {str(e)[:150]}")
        save_state(state)
        if first_run:
            SILENT = False
            first_run = False
            tg_send("🤖 Fractal Model v2 ligado. A partir de agora so chegam alertas novos.")
        if once:
            break
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
