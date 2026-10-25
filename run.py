"""
SWEET.TV TV-programme recommender — exam solution.

Pipeline
  1. read the three schedule files, keep only rows with tv_show_id != 0
  2. join the view log to the schedule by channel + interval overlap
  3. build user-programme interactions from the training weeks (11-20)
     and a validation set from weeks 21-30
  4. score three methods on the validation hold-out with MAP@5
  5. write submission.csv for the 1260 test users with the best method

Usage:  python3 run.py /path/to/data            # full run
        python3 run.py /path/to/data --fast     # sample the view log (debug)
"""
import csv, sys, os, time, math, bisect
from datetime import datetime
from collections import defaultdict, Counter

t0 = time.time()
BASE = sys.argv[1] if len(sys.argv) > 1 else "."
FAST = "--fast" in sys.argv
OUT = os.path.dirname(os.path.abspath(__file__))

VIEW = os.path.join(BASE, "dataset11-30.csv")
SCHED_TRAIN = os.path.join(BASE, "export_arh_11-20-final.csv")
SCHED_VAL   = os.path.join(BASE, "export_arh_21-30-final.csv")
SCHED_TEST  = os.path.join(BASE, "export_arh_31-42-final.csv")
SUB_TEMPLATE = os.path.join(BASE, "submission.csv")

MIN_OVERLAP_SEC = 300      # a view counts as "watched" at >= 5 minutes of overlap
MIN_OVERLAP_FRAC = 0.30    # ... or at >= 30% of the programme's airtime
K = 5

csv.field_size_limit(10_000_000)



# ---------------------------------------------------------------- time
# Two different textual formats in the same dataset:
#   view log  : "2020-03-09 00:00:02"   (Y-m-d H:M:S)
#   schedule  : "09.03.2020 06:00:00"   (d.m.Y H:M:S)
# Dates are cached, so only the time-of-day arithmetic runs per row.
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
    """2020-03-09 00:00:02 -> epoch seconds"""
    try:
        return (_day_seconds(int(s[0:4]), int(s[5:7]), int(s[8:10]))
                + int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19]))
    except Exception:
        return None


def t_sched(s):
    """09.03.2020 06:00:00 -> epoch seconds"""
    try:
        return (_day_seconds(int(s[6:10]), int(s[3:5]), int(s[0:2]))
                + int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19]))
    except Exception:
        return None


def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


def to_int(x):
    try:
        return int(float(x))
    except Exception:
        return None


# ---------------------------------------------------------------- schedules
def read_schedule(path, label):
    """channel -> sorted [(start, end, show_id)]; show_id -> metadata."""
    by_channel = defaultdict(list)
    meta = {}
    n_rows = n_lab = 0
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            n_rows += 1
            sid = to_int(row.get("tv_show_id"))
            if not sid:                      # 0 or unparsable = unlabelled airtime
                continue
            ch = to_int(row.get("channel_id"))
            st = t_sched(row.get("start_time") or "")
            du = to_int(row.get("duration"))
            if ch is None or st is None or du is None or du <= 0:
                continue
            n_lab += 1
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
    log(f"{label}: {n_rows} rows, {n_lab} labelled ({100*n_lab/max(n_rows,1):.1f}%), "
        f"{len(by_channel)} channels, {len(meta)} distinct programmes")
    return by_channel, meta, n_rows, n_lab


sched_tr, meta_tr, rows_tr, lab_tr = read_schedule(SCHED_TRAIN, "schedule 11-20")
sched_va, meta_va, rows_va, lab_va = read_schedule(SCHED_VAL,   "schedule 21-30")
sched_te, meta_te, rows_te, lab_te = read_schedule(SCHED_TEST,  "schedule 31-42")

META = {}
for m in (meta_tr, meta_va, meta_te):
    META.update(m)
TEST_POOL = set(meta_te)              # only programmes that actually air in weeks 31-42
log(f"candidate pool (airs in test weeks): {len(TEST_POOL)} programmes")


def build_lookup(by_channel):
    """channel -> (starts[], ends[], ids[]) for bisect."""
    out = {}
    for ch, items in by_channel.items():
        starts = [a for a, _, _ in items]
        ends = [b for _, b, _ in items]
        ids = [c for _, _, c in items]
        out[ch] = (starts, ends, ids)
    return out


LOOK_TR = build_lookup(sched_tr)
LOOK_VA = build_lookup(sched_va)


def match(look, ch, v_start, v_stop):
    """Programme ids whose airing interval overlaps the view interval enough."""
    entry = look.get(ch)
    if entry is None:
        return ()
    starts, ends, ids = entry
    # programmes starting before the view ends, scanning back while they can still overlap
    j = bisect.bisect_right(starts, v_stop)
    hits = []
    i = j - 1
    scanned = 0
    while i >= 0 and scanned < 40:
        s, e, sid = starts[i], ends[i], ids[i]
        if e <= v_start:
            # sorted by start: earlier entries may still be long, so allow a short look-back
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


# ---------------------------------------------------------------- view log
train_items = defaultdict(Counter)   # user -> Counter(show_id)
val_items = defaultdict(set)         # user -> {show_id} in weeks 21-30
n_views = n_hit_tr = n_hit_va = 0

with open(VIEW, newline="", encoding="utf-8", errors="replace") as f:
    r = csv.DictReader(f)
    for row in r:
        n_views += 1
        if FAST and n_views > 400_000:
            break
        if n_views % 1_000_000 == 0:
            log(f"view log: {n_views} rows")
        u = to_int(row.get("user_id"))
        ch = to_int(row.get("vsetv_id"))
        st = t_view(row.get("start_time") or "")
        sp = t_view(row.get("stop_time") or "")
        if u is None or ch is None or st is None or sp is None or sp <= st:
            continue
        for sid in match(LOOK_TR, ch, st, sp):
            train_items[u][sid] += 1
            n_hit_tr += 1
        for sid in match(LOOK_VA, ch, st, sp):
            val_items[u].add(sid)
            n_hit_va += 1

log(f"view rows {n_views}; train interactions {n_hit_tr} over {len(train_items)} users; "
    f"validation interactions {n_hit_va} over {len(val_items)} users")

# ---------------------------------------------------------------- methods
# A. popularity (reference baseline, not one of the two methods)
pop = Counter()
for u, c in train_items.items():
    for sid in c:
        pop[sid] += 1                      # distinct users per programme
pop_pool = [sid for sid, _ in pop.most_common() if sid in TEST_POOL]
log(f"popularity ranking over the test pool: {len(pop_pool)} programmes")

# B. item-item co-view (collaborative signal)
#    counts how often two programmes appear in the same user's history
co = defaultdict(Counter)
for u, c in train_items.items():
    hist = [s for s in c if s in pop]
    if len(hist) > 60:                      # cap very heavy users; keeps the pass O(n)
        hist = [s for s, _ in c.most_common(60)]
    for i, a in enumerate(hist):
        for b in hist[i+1:]:
            co[a][b] += 1
            co[b][a] += 1
log(f"co-view graph: {len(co)} programmes with neighbours")

# C. content-based (category + genres)
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
    # popularity as a mild tie-break inside the same attribute profile
    ranked = sorted(score.items(), key=lambda kv: (-kv[1], -pop.get(kv[0], 0)))
    out = [s for s, _ in ranked[:k]]
    if len(out) < k:
        out += [s for s in pop_pool if s not in out][:k - len(out)]
    return out[:k]


def rec_hybrid(u, k=K):
    """co-view first, content fills the remaining slots, popularity last."""
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


# ---------------------------------------------------------------- evaluation
def apk(actual, predicted, k=K):
    if not actual:
        return None
    predicted = predicted[:k]
    hits, score = 0, 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(actual), k)


eval_users = [u for u, s in val_items.items() if s]
log(f"validation cohort: {len(eval_users)} users with at least one week-21-30 interaction")

results = {}
for name, fn in (("popularity", rec_popularity),
                 ("co-view", rec_coview),
                 ("content", rec_content),
                 ("hybrid", rec_hybrid)):
    tt = time.time()
    vals = []
    for u in eval_users:
        a = apk(val_items[u], fn(u))
        if a is not None:
            vals.append(a)
    results[name] = sum(vals) / len(vals) if vals else 0.0
    log(f"MAP@{K} {name:11s} = {results[name]:.6f}   ({len(vals)} users, {time.time()-tt:.0f}s)")

best = max(results, key=results.get)
log(f"best on validation: {best} ({results[best]:.6f})")

# ---------------------------------------------------------------- submission
fn_best = {"popularity": rec_popularity, "co-view": rec_coview,
           "content": rec_content, "hybrid": rec_hybrid}[best]

test_users = []
with open(SUB_TEMPLATE, newline="", encoding="utf-8", errors="replace") as f:
    r = csv.reader(f)
    header = next(r)
    for row in r:
        if row:
            test_users.append(row[0])
log(f"submission template: {len(test_users)} users, header {header}")

n_cold = 0
with open(os.path.join(OUT, "submission.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(header)
    for uid in test_users:
        key = to_int(uid)
        if key not in train_items:
            n_cold += 1
        ids = fn_best(key)
        ids = (ids + pop_pool)[:K]
        w.writerow([uid, " ".join(str(i) for i in ids)])
log(f"wrote submission.csv  ({n_cold} of {len(test_users)} test users had no training history)")

with open(os.path.join(OUT, "validation_results.txt"), "w", encoding="utf-8") as f:
    f.write(f"schedule 11-20: {rows_tr} rows, {lab_tr} labelled\n")
    f.write(f"schedule 21-30: {rows_va} rows, {lab_va} labelled\n")
    f.write(f"schedule 31-42: {rows_te} rows, {lab_te} labelled\n")
    f.write(f"candidate pool (test weeks): {len(TEST_POOL)}\n")
    f.write(f"view rows: {n_views}\n")
    f.write(f"train interactions: {n_hit_tr} over {len(train_items)} users\n")
    f.write(f"validation interactions: {n_hit_va} over {len(val_items)} users\n")
    f.write(f"validation cohort: {len(eval_users)}\n")
    f.write(f"overlap rule: >= {MIN_OVERLAP_SEC}s or >= {MIN_OVERLAP_FRAC:.0%} of airtime\n\n")
    for k_, v_ in results.items():
        f.write(f"MAP@{K} {k_}: {v_:.6f}\n")
    f.write(f"\nselected: {best}\n")
    f.write(f"cold-start test users: {n_cold} of {len(test_users)}\n")
log("wrote validation_results.txt — done")
