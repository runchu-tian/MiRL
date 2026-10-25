"""Error taxonomy construction (paper §3.3): group the error steps from error profiling into error types.

Stages, each a subcommand that reads and writes files under --out:
  errors   collect the error steps from the io/ files of the rollouts             -> errors.jsonl
  propose  the annotator proposes error types from one batch of error steps      -> batches/b00.*
  refine   on each further batch, up to --batches in total, the annotator assigns the
           steps to the types and proposes new types only for patterns that recur -> batches/b01.* ...
  select   keep the types that recur across batches, with two examples each      -> types.json
  label    the annotator labels every error step with the fixed types, and the
           types that cover at least --min-share of the error steps are kept     -> labels.jsonl, misbehaviors.json

Batches are disjoint: propose and refine draw from one shuffled order of the error steps, so no step is shown twice.
The prompts are in prompts/ (paper Appendix A). The annotator is called through the Anthropic API (ANTHROPIC_API_KEY).

Usage:
  python taxonomy.py errors  --io runs/profiling_seed1/io runs/profiling_seed2/io --out <dir>
  python taxonomy.py propose --size N --out <dir>
  python taxonomy.py refine  --batches B --size N --out <dir>
  python taxonomy.py select  --min-batches K --out <dir>
  python taxonomy.py label   --size N --out <dir> [--min-share 0.01]
"""
from __future__ import annotations
import argparse, collections, glob, json, os, random, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"
MODEL = "claude-opus-5"   # Claude Opus 5, the annotator in the paper
CMD_CAP = 200             # rejected commands are cut to this many characters in the prompts
TRIES = 3                 # an annotator reply that is not valid JSON is asked for again, up to this many times


# ---------------------------------------------------------------- errors

MARKER = "=== WHAT THE AGENT DID ==="


def rejected_command(judge_input):
    """The proposed command, taken from the error localizer's prompt. A rejected step never runs, so this is its
    only copy."""
    i = (judge_input or "").find(MARKER)
    if i < 0:
        return ""
    block = judge_input[i + len(MARKER):]
    j = block.rfind("</think>")
    tail = block[j + len("</think>"):] if j >= 0 else block
    m = re.search(r"^\s*\$\s?(.*)", tail, re.S | re.M)
    return (m.group(1) if m else tail).strip()


def error_rows(path, prefix):
    """One row per policy step that the error localizer flagged in one rollout."""
    recs = json.load(open(path))
    stem = os.path.basename(path)[:-len(".json")]
    task, row = stem.rsplit("_r", 1)
    out = []
    for i, r in enumerate(recs):
        if r.get("kind") != "judge" or r.get("own_rewrite") or r.get("verdict") != "bad":
            continue
        f = r.get("fields") or {}
        # The step is fixed when the error localizer accepts one of the teacher's corrections before the next step.
        fixed = False
        for rr in recs[i + 1:]:
            k = rr.get("kind")
            if k == "judge" and rr.get("own_rewrite") and rr.get("verdict") == "good":
                fixed = True
            if k in ("execute", "propose", "k_cap_hit", "verifier", "propose_bad_request"):
                break
        out.append({"id": f"{prefix}{stem}#{r.get('step')}", "task": task, "row": row, "step": r.get("step"),
                    "command": rejected_command(r.get("input")), "command_seen": f.get("command_seen") or "",
                    "reason": f.get("reason") or "", "fixed": fixed})
    return out


def cmd_errors(a):
    """Error steps that the teacher fixed and that have an explanation. Ids carry the run index when there are
    several runs (one per seed), since the runs share file names."""
    n = 0
    with open(a.out / "errors.jsonl", "w") as fh:
        for k, d in enumerate(a.io):
            files = sorted(glob.glob(os.path.join(d, "*.json")))
            if not files:
                sys.exit(f"no io files under {d}")
            prefix = f"run{k + 1}:" if len(a.io) > 1 else ""
            for p in files:
                for r in error_rows(p, prefix):
                    if r["fixed"] and r["reason"] and r["command_seen"]:
                        fh.write(json.dumps(r) + "\n")
                        n += 1
    print(f"{n} error steps -> {a.out / 'errors.jsonl'}")


# ---------------------------------------------------------------- annotator

def load_errors(out):
    return [json.loads(l) for l in open(out / "errors.jsonl")]


def clip(t, n=CMD_CAP):
    t = (t or "").strip()
    return t if len(t) <= n else t[:n].rstrip() + " ...[%d chars total]" % len(t)


def render_examples(rows):
    return "\n\n".join("id: %s\nrejected command: %s\nwhat this command is doing: %s\nwhy this is a failure: %s"
                       % (r["id"], clip(r["command"]), r["command_seen"], r["reason"]) for r in rows)


def render_types(types, with_examples=False):
    out = []
    for t in types:
        s = "name: %s\ndefinition: %s" % (t["name"], t["definition"])
        if with_examples:
            for e in t.get("examples", []):
                s += "\nexample: %s -- %s" % (e["doing"], e["why"])
        out.append(s)
    return "\n\n".join(out)


def parse_json(text):
    """The reply is JSON, possibly inside a ``` fence."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return json.loads(m.group(1) if m else text[text.find("{"):text.rfind("}") + 1])


def annotate(prompt, effort):
    import anthropic
    client = anthropic.Anthropic()
    for attempt in range(1, TRIES + 1):
        # Streaming, since the replies are long. On a safety refusal the request falls back to the default model.
        with client.beta.messages.stream(model=MODEL, max_tokens=64000, output_config={"effort": effort},
                                         betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                                         messages=[{"role": "user", "content": prompt}]) as stream:
            msg = stream.get_final_message()
        text = "".join(b.text for b in msg.content if b.type == "text")
        try:
            return parse_json(text), text
        except ValueError:
            print(f"  reply is not valid JSON (stop reason {msg.stop_reason}), attempt {attempt}/{TRIES}", flush=True)
    sys.exit("the annotator gave no valid JSON reply")


def check_ids(rows, reply, keys):
    """Report input ids that the reply leaves out."""
    seen = set()
    for k in keys:
        for x in reply.get(k, []):
            if isinstance(x, dict):
                seen.add(x.get("id"))
                seen.update(x.get("example_ids", []))
            else:
                seen.add(x)
    miss = [r["id"] for r in rows if r["id"] not in seen]
    if miss:
        print(f"  {len(miss)} of {len(rows)} ids are missing from the reply", flush=True)


def next_batch(a, size):
    """The next `size` error steps in the shuffled order that no earlier batch used."""
    rows = load_errors(a.out)
    random.Random(a.seed).shuffle(rows)
    used = set()
    for f in sorted((a.out / "batches").glob("b*.ids")):
        used.update(f.read_text().split())
    pool = [r for r in rows if r["id"] not in used]
    if len(pool) < size:
        sys.exit(f"only {len(pool)} unused error steps left, asked for {size}")
    return pool[:size]


def save_batch(a, idx, kind, rows, prompt, text, reply):
    d = a.out / "batches"
    (d / f"b{idx:02d}.ids").write_text("\n".join(r["id"] for r in rows) + "\n")
    (d / f"b{idx:02d}.prompt.txt").write_text(prompt)
    (d / f"b{idx:02d}.reply.txt").write_text(text)
    (d / f"b{idx:02d}.json").write_text(json.dumps({"kind": kind, "reply": reply}, indent=1))


def cmd_propose(a):
    (a.out / "batches").mkdir(parents=True, exist_ok=True)
    if (a.out / "batches" / "b00.json").exists():
        sys.exit("batches/b00.json exists; propose runs once, then use refine")
    rows = next_batch(a, a.size)
    prompt = (PROMPTS / "annotator_propose.txt").read_text().replace("{examples}", render_examples(rows))
    reply, text = annotate(prompt, a.effort)
    check_ids(rows, reply, ["types", "unassigned"])
    save_batch(a, 0, "propose", rows, prompt, text, reply)
    print(f"b00: {len(reply.get('types', []))} types from {len(rows)} error steps")


def current_types(out):
    """All types so far, in the order they were proposed, with the ids assigned to each in each batch."""
    types, hits = {}, collections.defaultdict(dict)
    for f in sorted((out / "batches").glob("b*.json")):
        b = int(f.name[1:3])
        d = json.loads(f.read_text())
        r = d["reply"]
        new = r.get("types", []) if d["kind"] == "propose" else r.get("new_types", [])
        for t in new:
            if t["name"] not in types:
                types[t["name"]] = {"name": t["name"], "definition": t["definition"]}
            hits[t["name"]].setdefault(b, []).extend(t.get("example_ids", []))
        for x in r.get("assignments", []):
            for name in x.get("types", []):
                if name in types:
                    hits[name].setdefault(b, []).append(x["id"])
    return list(types.values()), hits


def cmd_refine(a):
    """Run refine batches until there are --batches batches in total, the first one included. A second call after an
    interruption continues up to this number."""
    while True:
        idx = len(list((a.out / "batches").glob("b*.json")))
        if idx == 0:
            sys.exit("run propose first")
        if idx >= a.batches:
            print(f"{idx} batches done")
            return
        types, _ = current_types(a.out)
        rows = next_batch(a, a.size)
        prompt = ((PROMPTS / "annotator_refine.txt").read_text()
                  .replace("{types}", render_types(types)).replace("{examples}", render_examples(rows)))
        reply, text = annotate(prompt, a.effort)
        check_ids(rows, reply, ["assignments", "unassigned"])
        save_batch(a, idx, "refine", rows, prompt, text, reply)
        print(f"b{idx:02d}: {len(reply.get('new_types', []))} new types, {len(types)} types before")


def cmd_select(a):
    """Keep the types that have error steps in at least --min-batches batches. Each kept type gets two examples,
    drawn at random from its error steps."""
    types, hits = current_types(a.out)
    nb = len(list((a.out / "batches").glob("b*.json")))
    rows = {r["id"]: r for r in load_errors(a.out)}
    rng = random.Random(a.seed)
    kept = []
    for t in types:
        counts = [len(set(hits[t["name"]].get(b, []))) for b in range(nb)]
        if sum(c > 0 for c in counts) < a.min_batches:
            continue
        ids = sorted({i for v in hits[t["name"]].values() for i in v if i in rows})
        ex = [rows[i] for i in rng.sample(ids, min(2, len(ids)))]
        kept.append(dict(t, counts=counts, examples=[{"doing": r["command_seen"], "why": r["reason"]} for r in ex]))
        print(f"  {counts}  {t['name']}")
    (a.out / "types.json").write_text(json.dumps(kept, indent=1))
    print(f"{len(kept)} of {len(types)} types are in at least {a.min_batches} of {nb} batches -> types.json")


def cmd_label(a):
    """Label every error step with the fixed types, in disjoint batches that cover all of them, then keep the types
    that cover at least --min-share of the error steps."""
    types = json.loads((a.out / "types.json").read_text())
    names = {t["name"] for t in types}
    rows = load_errors(a.out)
    tpl = (PROMPTS / "annotator_label.txt").read_text().replace("{types}", render_types(types, with_examples=True))
    d = a.out / "label"
    d.mkdir(parents=True, exist_ok=True)

    def label_batch(k, chunk):
        reply, text = annotate(tpl.replace("{examples}", render_examples(chunk)), a.effort)
        check_ids(chunk, reply, ["assignments", "unassigned"])
        (d / f"out_{k:03d}.txt").write_text(text)
        (d / f"out_{k:03d}.json").write_text(json.dumps(reply, indent=1))
        print(f"batch {k}: {len(chunk)} error steps", flush=True)

    for k, i in enumerate(range(0, len(rows), a.size)):
        if not (d / f"out_{k:03d}.json").exists():      # resume
            label_batch(k, rows[i:i + a.size])
    # A step that the reply leaves out has no answer (a step with no type is listed in "unassigned"), so it is sent
    # again, in new batches numbered after the existing ones.
    for _ in range(TRIES):
        seen = set()
        for f in d.glob("out_*.json"):
            r = json.loads(f.read_text())
            seen.update(x["id"] for x in r.get("assignments", []))
            seen.update(u["id"] if isinstance(u, dict) else u for u in r.get("unassigned", []))
        missing = [r for r in rows if r["id"] not in seen]
        if not missing:
            break
        print(f"{len(missing)} error steps have no answer; labeling them again", flush=True)
        k0 = max(int(f.stem[4:]) for f in d.glob("out_*.json")) + 1
        for k, i in enumerate(range(0, len(missing), a.size)):
            label_batch(k0 + k, missing[i:i + a.size])

    labels, per_type = {}, collections.Counter()
    for f in sorted(d.glob("out_*.json")):
        r = json.loads(f.read_text())
        for x in r.get("assignments", []):
            labels[x["id"]] = [n for n in x.get("types", []) if n in names]
        for u in r.get("unassigned", []):
            labels.setdefault(u["id"] if isinstance(u, dict) else u, [])
    task = {r["id"]: r["task"] for r in rows}
    with open(a.out / "labels.jsonl", "w") as fh:
        for r in rows:
            ts = labels.get(r["id"], [])
            per_type.update(ts)
            fh.write(json.dumps({"id": r["id"], "task": task[r["id"]], "types": ts}) + "\n")
    n = len(rows)
    kept = [dict(t, steps=per_type[t["name"]], share=round(per_type[t["name"]] / n, 4))
            for t in types if per_type[t["name"]] >= a.min_share * n]
    kept.sort(key=lambda t: -t["steps"])
    (a.out / "misbehaviors.json").write_text(json.dumps(kept, indent=1))
    print(f"{len(labels)} of {n} error steps labeled")
    for t in types:
        c = per_type[t["name"]]
        print(f"  {c:6d}  {100 * c / n:5.1f}%  {'kept' if c >= a.min_share * n else '    '}  {t['name']}")
    print(f"{len(kept)} types cover at least {100 * a.min_share:g}% of the error steps -> misbehaviors.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("errors", help="collect the error steps")
    p.add_argument("--io", nargs="+", required=True, help="io/ directories of the profiling runs, one per seed")
    for name, h in [("propose", "propose types on the first batch"), ("refine", "refine the types on more batches"),
                    ("select", "keep the types that recur across batches"), ("label", "label every error step")]:
        q = sub.add_parser(name, help=h)
        if name in ("propose", "refine", "label"):
            q.add_argument("--size", type=int, required=True, help="error steps per batch")
            q.add_argument("--effort", default="high", help="effort of the annotator (default: high)")
        if name == "refine":
            q.add_argument("--batches", type=int, required=True,
                           help="total number of batches, the first one included")
        if name == "select":
            q.add_argument("--min-batches", type=int, required=True,
                           help="keep a type only if it has error steps in at least this many batches")
        if name == "label":
            q.add_argument("--min-share", type=float, default=0.01,
                           help="keep the types that cover at least this share of the error steps (default: 0.01)")
        q.add_argument("--seed", type=int, default=0, help="seed of the batch order and of the examples (default: 0)")
    for q in sub.choices.values():
        q.add_argument("--out", type=Path, required=True, help="output directory")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    {"errors": cmd_errors, "propose": cmd_propose, "refine": cmd_refine, "select": cmd_select,
     "label": cmd_label}[a.cmd](a)


if __name__ == "__main__":
    main()
