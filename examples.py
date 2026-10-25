"""
Top-5 per method for two concrete users, with ids, titles and categories.
Same pipeline as run.py, but no validation loop -- runs in ~80 seconds.

Usage:  python3 examples.py /path/to/data
Writes: examples.txt
"""
import csv, sys, os, time, bisect
from datetime import datetime
from collections import defaultdict, Counter

t0 = time.time()
BASE = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = os.path.dirname(os.path.abspath(__file__))

VIEW = os.path.join(BASE, "dataset11-30.csv")
SCHED_TRAIN = os.path.join(BASE, "export_arh_11-20-final.csv")
SCHED_VAL = os.path.join(BASE, "export_arh_21-30-final.csv")
SCHED_TEST = os.path.join(BASE, "export_arh_31-42-final.csv")
SUB_TEMPLATE = os.path.join(BASE, "submission.csv")

MIN_OVERLAP_SEC = 300
MIN_OVERLAP_FRAC = 0.30
K = 5

csv.field_size_limit(10_000_000)
_DAY_CACHE = {}
_EPOCH = datetime(1970, 1, 1)


def _day_seconds(y, m, d):
    key = (y, m, d)
    v = _DAY_CACHE.get(key)
    if v is None:
        v = int((datetime(y, m, d) - _EPOCH).total_seconds())
        _DAY_CACHE[key] = v
    return v


def t_view(s):
    try:
        return (_day_seconds(int(s[0:4]), int(s[5:7]), int(s[8:10]))
                + int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19]))
    except Exception:
        return None


def t_sched(s):
    try:
        return (_day_seconds(int(s[6:10]), int(s[3:5]), int(s[0:2]))
                + int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19]))
    except Exception:
        return None


def log(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def to_int(x):
    try:
        return int(float(x))
    except Exception:
        return None


def read_schedule(path):
    by_channel = defaultdict(list)
    meta = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            sid = to_int(row.get("tv_show_id"))
            if not sid:
                continue
            ch = to_int(row.get("channel_id"))
            st = t_sched(row.get("start_time") or "")
            du = to_int(row.get("duration"))
            if ch is None or st is None or du is None or du <= 0:
                continue
            by_channel[ch].append((st, st + du, sid))
            if sid not in meta:
                meta[sid] = (
                    (row.get("tv_show_title") or "").strip(),
                    (row.get("tv_show_category") or "").strip(),
                    tuple(sorted({(row.get(f"tv_show_genre_{i}") or "").strip()
                                  for i in (1, 2, 3)} - {""})),
                )
    for ch in by_channel:
        by_channel[ch].sort()
    return by_channel, meta


sched_tr, meta_tr = read_schedule(SCHED_TRAIN)
sched_va, meta_va = read_schedule(SCHED_VAL)
sched_te, meta_te = read_schedule(SCHED_TEST)
log("schedules read")

META = {}
for m in (meta_tr, meta_va, meta_te):
    META.update(m)
TEST_POOL = set(meta_te)


def build_lookup(by_channel):
    return {ch: ([a for a, _, _ in it], [b for _, b, _ in it], [c for _, _, c in it])
            for ch, it in by_channel.items()}


LOOK_TR = build_lookup(sched_tr)


def match(look, ch, v_start, v_stop):
    entry = look.get(ch)
    if entry is None:
        return ()
    starts, ends, ids = entry
    j = bisect.bisect_right(starts, v_stop)
    hits = []
    i, scanned = j - 1, 0
    while i >= 0 and scanned < 40:
        s, e, sid = starts[i], ends[i], ids[i]
        if e <= v_start:
            scanned += 1
            i -= 1
            continue
        ov = min(e, v_stop) - max(s, v_start)
        if ov > 0:
            dur = e - s
            if ov >= MIN_OVERLAP_SEC or (dur > 0 and ov / dur >= MIN_OVERLAP_FRAC):
                hits.append(sid)
        scanned += 1
        i -= 1
    return hits


train_items = defaultdict(Counter)
n = 0
with open(VIEW, newline="", encoding="utf-8", errors="replace") as f:
    for row in csv.DictReader(f):
        n += 1
        if n % 2_000_000 == 0:
            log(f"view log: {n} rows")
        u = to_int(row.get("user_id"))
        ch = to_int(row.get("vsetv_id"))
        st = t_view(row.get("start_time") or "")
        sp = t_view(row.get("stop_time") or "")
        if u is None or ch is None or st is None or sp is None or sp <= st:
            continue
        for sid in match(LOOK_TR, ch, st, sp):
            train_items[u][sid] += 1
log(f"train interactions over {len(train_items)} users")

pop = Counter()
for u, c in train_items.items():
    for sid in c:
        pop[sid] += 1
pop_pool = [sid for sid, _ in pop.most_common() if sid in TEST_POOL]

co = defaultdict(Counter)
for u, c in train_items.items():
    hist = [s for s in c if s in pop]
    if len(hist) > 60:
        hist = [s for s, _ in c.most_common(60)]
    for i, a in enumerate(hist):
        for b in hist[i + 1:]:
            co[a][b] += 1
            co[b][a] += 1
log(f"co-view graph: {len(co)} programmes")

item_attr = {}
for sid, (title, cat, genres) in META.items():
    feats = set()
    if cat:
        feats.add("cat:" + cat)
    for g in genres:
        feats.add("gen:" + g)
    item_attr[sid] = feats
attr_items = defaultdict(set)
for sid, feats in item_attr.items():
    if sid in TEST_POOL:
        for ft in feats:
            attr_items[ft].add(sid)


def rec_popularity(u, k=K):
    return pop_pool[:k]


def rec_coview(u, k=K):
    hist = train_items.get(u)
    if not hist:
        return pop_pool[:k]
    score = Counter()
    for sid, w in hist.items():
        for nb, c in co.get(sid, {}).items():
            if nb in TEST_POOL and nb not in hist:
                score[nb] += c * w
    out = [s for s, _ in score.most_common(k)]
    if len(out) < k:
        out += [s for s in pop_pool if s not in out][:k - len(out)]
    return out[:k]


def rec_content(u, k=K):
    hist = train_items.get(u)
    if not hist:
        return pop_pool[:k]
    prof = Counter()
    for sid, w in hist.items():
        for ft in item_attr.get(sid, ()):
            prof[ft] += w
    if not prof:
        return pop_pool[:k]
    score = Counter()
    for ft, w in prof.items():
        for sid in attr_items.get(ft, ()):
            if sid not in hist:
                score[sid] += w
    ranked = sorted(score.items(), key=lambda kv: (-kv[1], -pop.get(kv[0], 0)))
    out = [s for s, _ in ranked[:k]]
    if len(out) < k:
        out += [s for s in pop_pool if s not in out][:k - len(out)]
    return out[:k]


def rec_hybrid(u, k=K):
    out = []
    for s in rec_coview(u, k):
        if s not in out:
            out.append(s)
    if len(out) < k:
        for s in rec_content(u, k):
            if s not in out:
                out.append(s)
            if len(out) == k:
                break
    if len(out) < k:
        for s in pop_pool:
            if s not in out:
                out.append(s)
            if len(out) == k:
                break
    return out[:k]


# ---------------------------------------------------------------- pick users
test_users = []
with open(SUB_TEMPLATE, newline="", encoding="utf-8", errors="replace") as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        if row:
            v = to_int(row[0])
            if v is not None:
                test_users.append(v)

with_hist = [(len(train_items[u]), u) for u in test_users if u in train_items]
with_hist.sort()
heavy = with_hist[-1][1]
median = with_hist[len(with_hist) // 2][1]
log(f"picked users: heavy={heavy} ({with_hist[-1][0]} programmes), "
    f"median={median} ({with_hist[len(with_hist)//2][0]} programmes)")

lines = []


def emit(s=""):
    print(s, flush=True)
    lines.append(s)


for u in (heavy, median):
    hist = train_items.get(u, Counter())
    emit("=" * 78)
    emit(f"USER {u} — {len(hist)} distinct programmes in weeks 11-20, "
         f"{sum(hist.values())} matched viewings")
    emit("-" * 78)
    emit("Top-5 most-watched in training history:")
    for sid, c in hist.most_common(5):
        t, cat, gen = META.get(sid, ("?", "?", ()))
        emit(f"   {sid:>7}  {c:>4}x  {t[:46]:46s}  {cat}")
    for name, fn in (("popularity", rec_popularity), ("co-view", rec_coview),
                     ("content", rec_content), ("hybrid", rec_hybrid)):
        emit("")
        emit(f"{name} Top-5:")
        for rank, sid in enumerate(fn(u), 1):
            t, cat, gen = META.get(sid, ("?", "?", ()))
            g = ", ".join(gen) if gen else "—"
            emit(f"  {rank}. {sid:>7}  {t[:44]:44s}  {cat:22s}  {g}")
    emit("")

with open(os.path.join(OUT, "examples.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
log("wrote examples.txt — done")
