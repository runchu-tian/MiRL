/* Method (Figure 3), split into four steps. Each step has its own timeline (drag, play, pause) and highlights
   the paper sentence it shows. A step script marks a frame at every wait(), so any frame can be rebuilt by
   replaying the script instantly up to that wait. */
(() => {
  const textRoot = $("#stage-text");
  let curText = null;
  const hl = b => $$(".beat", curText).forEach(s => s.classList.toggle("hl", +s.dataset.b === b));
  const prog = () => {};   // progress comes from the timeline
  const it = s => s.replace(/\b([jkn])\b/g, "<i>$1</i>");
  const roleBadge = w => w === "T" ? `<span class="who-b T">T</span>` : `<span class="who-b">π</span>`;

  /* ------------------------------------------------ (a) error localization */
  const S0 = (() => {
    const el = $("#st0");
    const ROWS = [
      {w: "pi", s: "1", cmd: "ls photos backup | head", ok: true},
      {dots: true},
      {w: "pi", s: "j", cmd: "awk '…' h.txt | xargs mv", ok: false, why: "reruns the same failing command"},
      {w: "T", s: "j", cmd: "cd ~/lib && xargs mv …", ok: true},
      {w: "pi", s: "j+1", cmd: "ls dupes | head", ok: true},
      {dots: true},
      {w: "pi", s: "k", cmd: "ls dupes | wc -l #verify", ok: false, why: "counts files, not their content"},
      {w: "T", s: "k", cmd: "md5sum dupes/* | cmp …", ok: true},
      {w: "pi", s: "k+1", cmd: "head dupes/report.csv", ok: true},
    ];
    el.append(html(`<div class="legend-row"><span>${roleBadge("pi")} policy</span><span>${roleBadge("T")} teacher</span><span>${ICON.judge("#E7B3BF")} error localizer</span></div>`));
    const grid = html(`<div class="s1"><div class="traj"></div><div class="collect"><div class="h">${ICON.judge("#E7B3BF")}Localized error steps</div></div></div>`);
    el.append(grid);
    const traj = $(".traj", grid), col = $(".collect", grid);
    const items = ROWS.map(r => {
      if (r.dots) { const d = html(`<div class="vdots fx">⋮</div>`); traj.append(d); return {r, row: d}; }
      const row = html(`<div class="trow fx ${r.w}">${roleBadge(r.w)}<span class="sn">${it(r.s)}</span><span class="cmd">${r.cmd}</span><span class="v"></span></div>`);
      traj.append(row);
      let why = null, card = null;
      if (r.why) {
        why = html(`<div class="reason fx">${ICON.judge("#E7B3BF")}<span><b>Reason:</b> ${r.why}</span></div>`);
        traj.append(why);
        const prev = r.s === "j" ? "j−1" : "k−1";
        card = html(`<div class="ecard fx"><div class="pre"><span class="pill given">instr.</span><span class="pill given">1</span><span class="pill given">…</span><span class="pill given">${it(prev)}</span><span class="pill mis">${it(r.s)}</span></div><div class="why">“${r.why}”</div><div class="lab">prefix + error step + reason</div></div>`);
        col.append(card);
      }
      return {r, row, why, card};
    });
    const reset = () => { offAll(el); items.forEach(({row}) => { row.classList.remove("bad"); const v = $(".v", row); if (v) v.innerHTML = ""; }); };
    const script = async wait => {
      await wait(400);
      for (let n = 0; n < items.length; n++) {
        const {r, row, why, card} = items[n];
        prog((n + 1) / items.length);
        if (r.dots) { on(row); await wait(300); continue; }
        hl(r.w === "T" ? 3 : 1); on(row); await wait(650);
        const v = $(".v", row);
        hl(r.w === "T" ? 3 : 2); v.innerHTML = `<span class="checking">${ICON.judge("#E7B3BF")}</span>`; await wait(750);
        if (r.ok) { v.innerHTML = ICON.check(); hl(4); await wait(600); }
        else {
          v.innerHTML = ICON.cross("#b04f6c"); row.classList.add("bad"); hl(2); await wait(350);
          on(why); await wait(900);
          hl(5); on(card); await wait(1000);
        }
      }
    };
    return {el, reset, script};
  })();

  /* ------------------------------------------------ (b) error taxonomy */
  const S1 = (() => {
    const el = $("#st1");
    // abstract types: 0-3 recur across batches (they become the fixed set); s* appear in one batch only
    const TC = {0: PAL[0], 1: PAL[2], 2: PAL[4], 3: PAL[6], a: "#cfd6dc", b: "#e0d4bf", c: "#d6cfe0"};
    const BATCH = [[0, 1, 2, "a"], [0, 1, 3, "b"], [0, 2, 3, "c"]];
    const flag = t => `<span class="flag" data-t="${t}" style="background:${TC[t]}"></span>`;
    const sq = n => Array.from({length: n}, () => `<span class="estep"></span>`).join("");
    el.append(html(`<div class="legend-row"><span><span class="estep"></span> error step</span><span><span class="who-b A">A</span> annotator</span><span>${flag(0)} error type (name + definition)</span></div>`));
    const s = html(`<div class="s2">
      <div class="beatbox" data-b="1"><div class="bh"><span class="no">1</span>propose error types, batch by batch</div>
        <div class="batches">${BATCH.map((b, i) => `<div class="batch fx" data-i="${i}">${sq(5)}<span class="arrow-r">→</span><span class="who-b A">A</span><span class="arrow-r">→</span><span class="props">${b.map(t => `<span class="fx pop" style="display:inline-block;margin-right:3px">${flag(t)}</span>`).join("")}</span></div>`).join("")}</div></div>
      <div class="beatbox" data-b="2"><div class="bh"><span class="no">2</span>keep the types that recur across batches</div>
        <div class="keepset fx"><span>fixed set of error types:</span><span class="flags">${[0, 1, 2, 3].map(flag).join("")}</span><span style="color:var(--ink-3)">· one-batch types dropped</span></div></div>
      <div class="beatbox" data-b="3"><div class="bh"><span class="no">3</span>label every error step</div>
        <div class="labeling"><div class="stream"></div>
          <div><div class="hist"></div><div class="hist-lab fx"><span class="m">misbehaviors (≥ 1% of error steps)</span><span class="t">long tail</span></div></div></div></div>
    </div>`);
    el.append(s);
    const SH = [15.4, 11.5, 11.2, 5.1, 4.7, 2.5, 1.7, 1.4];
    const TAIL = Array.from({length: 12}, (_, i) => 0.95 * Math.exp(-i / 4) + 0.048);   // schematic, below 1%
    const hist = $(".hist", s), MAX = 16;
    const bars = [...SH.map((v, i) => [v, PAL[i], false]), ...TAIL.map((v, i) => [v, mix(PAL[7], "#FFFFFF", Math.min(1, i / 9)), true])].map(([v, c, t]) => {
      const b = html(`<div class="b ${t ? "tail" : ""}" style="--c:${c}"></div>`); b.dataset.v = v; hist.append(b); return b;
    });
    const thr = html(`<div class="thr fx"><span>1%</span></div>`); thr.style.bottom = (1 / MAX * 100) + "%"; hist.append(thr);
    const split = html(`<div class="split fx"></div>`); hist.append(split);
    // the labeled stream: each square takes its type color (or stays grey with no known type)
    const ASSIGN = [0, 1, "x", 2, 0, 3, "x", 0, 1, 4, "x", 2, 0, 5, "x", 1, 3, "x", 0, 2, 6, "x", 4, 1, "x", 0, 7, "x", 2, "x", 0, 1, "x", 3, 2, "x"];
    const stream = $(".stream", s);
    const sqs = ASSIGN.map(() => { const e = html(`<span class="estep"></span>`); stream.append(e); return e; });
    const boxes = $$(".beatbox", s);
    const cur = b => boxes.forEach(x => x.classList.toggle("cur", +x.dataset.b === b));
    const reset = () => {
      offAll(el); cur(0);
      $$(".flag", s).forEach(f => f.classList.remove("out"));
      $$(".batch .estep", s).forEach(e => { e.style.background = ""; e.style.borderColor = ""; });
      sqs.forEach(e => { e.style.background = ""; e.style.borderColor = ""; });
      bars.forEach(b => (b.style.height = "0"));
      split.style.left = (bars[8].offsetLeft - 2) + "px";
    };
    const script = async wait => {
      await wait(300);
      hl(1); cur(1);
      const bs = $$(".batch", s);
      for (let i = 0; i < bs.length; i++) {
        prog((i + 1) / 10);
        on(bs[i]); await wait(450);
        for (const f of $$(".props > span", bs[i])) { on(f); await wait(160); }
        await wait(350);
      }
      hl(2); cur(2); prog(0.4);
      await wait(500);
      $$(".batch .flag", s).forEach(f => { if (!/^\d$/.test(f.dataset.t)) f.classList.add("out"); });
      await wait(500);
      on($(".keepset", s)); await wait(1100);
      hl(3); cur(3);
      split.style.left = (bars[8].offsetLeft - 2) + "px";
      for (let i = 0; i < sqs.length; i++) {
        const t = ASSIGN[i];
        sqs[i].style.background = t === "x" ? "#eceff3" : PAL[t];
        sqs[i].style.borderColor = t === "x" ? "#c4c8cf" : "rgba(0,0,0,.45)";
        const f = (i + 1) / sqs.length;
        bars.forEach(b => (b.style.height = (b.dataset.v / MAX * 100 * f) + "%"));
        prog(0.45 + 0.45 * f);
        await wait(85);
      }
      await wait(700);
      hl(4); on(thr); on(split); on($(".hist-lab", s)); prog(1);
      await wait(1200);
    };
    return {el, reset, script};
  })();

  /* ------------------------------------------------ (c) misbehavior prefixes */
  const S2 = (() => {
    const el = $("#st2");
    const s = html(`<div class="s3">
      <div><div class="under" style="margin:0 0 6px">unfiltered candidate</div>
        <div class="cand fx" id="cand"><span class="pill given">instr.</span><span class="pill given">1</span><span class="pill given">…</span><span class="pill given"><i>k</i>−1</span><span class="kstep"><i>k</i></span><span class="tag" style="font-size:12.5px;font-weight:600;margin-left:8px"></span></div>
        <div class="under">└ prefix <i>h</i> ┘</div></div>
      <div class="req fx"><b>REQUIREMENT</b><i>π</i> continues from <i>h</i> in a live container, so it must reach the same state</div>
      <div class="rerun fx"><div class="rh"><span style="display:inline-flex;align-items:center;gap:8px"><span class="ctr">${ICON.container}</span><span>rerun <i>h</i> in a live container</span></span><small>exact match, step by step</small></div>
        <div class="cmp">
          <div class="hd">● recorded</div><div></div><div class="hd">↻ rerun</div><div></div>
          <div class="cmdl">$ ls backup/</div>
          <div class="out" data-r="0">a.jpg …</div><div class="eq fx" data-r="0">=</div><div class="out fx" data-r="0" data-live>a.jpg …</div><div class="v fx" data-r="0">${ICON.check()}</div>
          <div class="cmdl">$ date</div>
          <div class="out" data-r="1">10:42</div><div class="eq no fx" data-r="1">≠</div><div class="out fx" data-r="1" data-live>10:47</div><div class="v fx" data-r="1">${ICON.cross("#b04f6c")}</div>
        </div>
        <div class="rule"><span>all ✓ = <b class="k">keep</b></span><span>any ✗ = <b class="d">drop</b></span></div></div>
      <div class="pool fx"><div><div class="under" style="margin:0 0 8px">more candidate prefixes, rerun in parallel</div><div class="cands"></div></div>
        <div class="tray"><div style="font-weight:600;color:var(--ink)">filtered prefixes</div><div class="kept"></div><div class="stat fx" style="margin-top:auto">→ live containers for RL</div></div></div>
    </div>`);
    el.append(s);
    // schematic pool: [prefix length, index of the first mismatching command or -1]
    const POOL = [[5, -1], [4, 2], [6, 4], [3, -1], [5, 1], [6, -1], [4, 3], [5, 2], [6, 5]];
    const cands = $(".cands", s), kept = $(".kept", s);
    const rows = POOL.map(([n]) => { const r = html(`<div class="crow">${"<i></i>".repeat(n)}<span class="tag"></span></div>`); cands.append(r); return r; });
    const keptRows = [];
    const reset = () => {
      offAll(el);
      $$(".cmp .out", s).forEach(o => o.classList.remove("ok", "no"));
      const c = $("#cand", s); c.style.opacity = ""; $(".tag", c).textContent = "";
      rows.forEach(r => { r.classList.remove("keep", "drop"); $$("i", r).forEach(i => (i.className = "")); $(".tag", r).textContent = ""; });
      keptRows.splice(0).forEach(k => k.remove());
    };
    const script = async wait => {
      await wait(300);
      hl(1); prog(0.1); on($("#cand", s)); await wait(700); on($(".req", s)); await wait(1500);
      hl(2); prog(0.25); on($(".rerun", s)); await wait(700);
      for (const r of [0, 1]) {
        const live = $(`.out[data-live][data-r="${r}"]`, s), rec = $(`.out:not([data-live])[data-r="${r}"]`, s);
        on(live); await wait(650);
        on($(`.eq[data-r="${r}"]`, s));
        const ok = r === 0;
        [live, rec].forEach(o => o.classList.add(ok ? "ok" : "no"));
        on($(`.v[data-r="${r}"]`, s));
        if (!ok) hl(3);
        prog(0.35 + r * 0.15);
        await wait(900);
      }
      const c = $("#cand", s); $(".tag", c).innerHTML = `<span style="color:var(--mis)">✗ drop</span>`; c.style.opacity = ".45";
      await wait(1000);
      on($(".pool", s)); prog(0.6); await wait(500);
      // rerun every candidate one command at a time; a mismatch drops it at once
      const maxN = Math.max(...POOL.map(p => p[0]));
      for (let step = 0; step < maxN; step++) {
        POOL.forEach(([n, bad], i) => {
          const r = rows[i];
          if (r.classList.contains("drop") || r.classList.contains("keep") || step >= n) return;
          const dot = $$("i", r)[step];
          if (step === bad) { dot.className = "no"; r.classList.add("drop"); $(".tag", r).textContent = "drop"; return; }
          dot.className = "ok";
          if (step === n - 1) {
            r.classList.add("keep"); $(".tag", r).textContent = "keep";
            const k = html(`<div class="crow fx">${"<i></i>".repeat(n)}</div>`); kept.append(k); keptRows.push(k);
            requestAnimationFrame(() => on(k));
          }
        });
        prog(0.6 + 0.3 * (step + 1) / maxN);
        await wait(420);
      }
      hl(4); on($(".stat", s)); prog(1);
      await wait(1400);
    };
    return {el, reset, script};
  })();

  /* ------------------------------------------------ (d) MiRL training */
  const S3 = (() => {
    const el = $("#st3");
    const ROLL = [
      {mis: false, pass: true, cap: "clean pass"},
      {mis: true, pass: true, cap: "passes, but misbehaves"},
      {mis: false, pass: false, cap: "avoids it, but fails"},
      {mis: true, pass: false, cap: "misbehaves and fails"},
    ];
    const s = html(`<div class="s4">
      <div>
        <div class="src fx"><span class="ctr">${ICON.container} live</span><span class="pill given">instr.</span><span class="pill given">1</span><span class="pill given">…</span><span class="pill given"><i>k</i>−1</span><span class="arrow-r">→</span><span class="who-b">π</span><span style="font-size:12.5px;color:var(--ink-3);margin-left:4px">rollout</span></div>
        <div class="rolls">${ROLL.map((r, i) => `<div class="roll" data-i="${i}"><div class="steps"><span class="judge"></span>
          <span class="pill fx" data-s="0"><i>k</i></span><span class="pill fx" data-s="1"><i>k</i>+1</span><span class="pill dots fx" data-s="2">…</span><span class="pill fx" data-s="3"><i>n</i><sub>${i}</sub></span>
          <span class="end fx">${ICON.test(r.pass)}</span></div><div class="cap fx">${r.cap}</div></div>`).join("")}</div>
        <div class="under fx" id="judge-note" style="margin-top:12px;display:flex;align-items:center;gap:6px"><span style="display:inline-block;width:26px;height:14px;border:1.5px dashed var(--gold-d);border-radius:5px"></span>behavior judge reads the first <i>m</i> steps after the prefix</div>
      </div>
      <div class="rtab"><div class="f"><i>r</i> = <i>r</i><sub>o</sub> ∧ <i>r</i><sub>e</sub></div>
        <table><thead><tr><th>${ICON.judge().replace('class="icon"', 'class="icon" style="width:16px;height:16px;vertical-align:-3px"')} <i>r</i><sub>e</sub></th><th><i>r</i><sub>o</sub></th><th>final <i>r</i></th></tr></thead>
        <tbody>${ROLL.map((r, i) => `<tr data-i="${i}"><td class="fx pop">${r.mis ? ICON.cross() : ICON.check()}</td><td class="fx pop">${r.pass ? ICON.check() : ICON.cross()}</td><td class="fx pop">${!r.mis && r.pass ? `<span class="one">1</span>` : `<span class="zero">0</span>`}</td></tr>`).join("")}</tbody></table>
        <div class="update fx">↺ update <i>π</i></div></div>
    </div>`);
    el.append(s);
    const rolls = $$(".roll", s), trs = $$("tbody tr", s);
    const reset = () => {
      offAll(el);
      rolls.forEach(r => { $(".judge", r).classList.remove("on"); $$(".pill", r).forEach(p => p.classList.remove("gen", "mis", "good")); });
    };
    const placeJudge = r => { const ps = $$(".pill", r), j = $(".judge", r); j.style.left = (ps[0].offsetLeft - 4) + "px"; j.style.width = (ps[1].offsetLeft + ps[1].offsetWidth - ps[0].offsetLeft + 8) + "px"; };
    const script = async wait => {
      await wait(300);
      hl(1); on($(".src", s)); prog(0.1); await wait(1000);
      const OFF = [0, 1, 2, 1];   // small offsets so the four rollouts do not move in lockstep
      for (let t = 0; t < 7; t++) {
        rolls.forEach((r, i) => {
          const st = t - OFF[i];
          if (st < 0 || st > 3) return;
          const p = $(`.pill[data-s="${st}"]`, r); on(p); p.classList.add("gen");
          if (st === 1) {   // the judge checks the first m steps
            placeJudge(r); $(".judge", r).classList.add("on");
            $(`.pill[data-s="0"]`, r).classList.add(ROLL[i].mis ? "mis" : "good");
          }
        });
        if (t === 2) hl(2);
        prog(0.1 + 0.5 * (t + 1) / 7);
        await wait(520);
      }
      on($("#judge-note", s));
      rolls.forEach(r => { on($(".end", r)); on($(".cap", r)); });
      await wait(900);
      hl(3);
      for (let i = 0; i < trs.length; i++) {
        $$("td", trs[i]).forEach(td => on(td));
        prog(0.65 + 0.3 * (i + 1) / trs.length);
        await wait(600);
      }
      on($(".update", s)); prog(1);
      await wait(1400);
    };
    return {el, reset, script};
  })();

  /* ------------------------------------------------ controller: one timeline per step */
  const STAGES = [S0, S1, S2, S3];
  const tabs = $$("#stepper .tab"), texts = $$(".t", textRoot);
  const range = $("#tl-range"), btn = $("#tl-play");
  const ICON_PLAY = `<svg viewBox="0 0 24 24"><path d="M6 4l14 8-14 8z" fill="currentColor"/></svg>`;
  const ICON_PAUSE = `<svg viewBox="0 0 24 24"><path d="M6 4h4v16H6zM14 4h4v16h-4z" fill="currentColor"/></svg>`;
  let cur = 0, gen = 0, frame = 0, playing = false;
  curText = texts[0];
  const ONLY = new URLSearchParams(location.search).get("stage");   // ?stage=i opens one step (for checks)

  // count the frames of each step with a dry run
  async function count(st) {
    let n = 0;
    st.reset();
    await st.script(() => { n++; return Promise.resolve(); });
    st.reset();
    return n;
  }
  function setFrame(f) {
    frame = f; range.value = f;
    $(".prog", tabs[cur]).style.width = (f / STAGES[cur].n * 100) + "%";
  }
  function setPlaying(v) { playing = v; btn.innerHTML = v ? ICON_PAUSE : ICON_PLAY; btn.setAttribute("aria-label", v ? "Pause" : "Play"); }

  // rebuild frame `to` instantly, then keep playing in real time if `play`
  async function go(to, play) {
    const st = STAGES[cur], my = ++gen;
    setPlaying(play && to < st.n);
    st.el.classList.add("instant");
    $$(".beat", curText).forEach(b => b.classList.remove("hl"));
    st.reset();
    let k = 0;
    const wait = ms => {
      if (my !== gen) return Promise.reject(STOP);
      k++;
      if (k < to) return Promise.resolve();
      setFrame(k);
      if (k === to) requestAnimationFrame(() => st.el.classList.remove("instant"));
      if (!playing) return Promise.reject(STOP);
      return new Promise((ok, no) => setTimeout(() => (my === gen ? ok() : no(STOP)), STATIC ? 0 : ms));
    };
    if (to === 0) { setFrame(0); requestAnimationFrame(() => st.el.classList.remove("instant")); if (!playing) return; }
    try { await st.script(wait); } catch (e) { if (e !== STOP) throw e; return; }
    if (my === gen) { setFrame(st.n); setPlaying(false); st.el.classList.remove("instant"); }
  }

  function show(i, play = true) {
    gen++;
    cur = i;
    tabs.forEach((t, j) => t.classList.toggle("on", j === i));
    STAGES.forEach((st, j) => st.el.classList.toggle("on", j === i));
    texts.forEach((t, j) => t.classList.toggle("on", j === i));
    curText = texts[i];
    $$(".beat", textRoot).forEach(b => b.classList.remove("hl"));
    range.max = STAGES[i].n;
    tabs.forEach((t, j) => ($(".prog", t).style.width = "0"));
    go(STATIC ? STAGES[i].n : 0, play && !STATIC);
  }

  btn.addEventListener("click", () => {
    if (playing) { gen++; setPlaying(false); return; }
    go(frame >= STAGES[cur].n ? 0 : frame, true);
  });
  range.addEventListener("input", () => go(+range.value, false));
  tabs.forEach((t, i) => t.addEventListener("click", () => show(i)));
  $$(".q[data-stage]").forEach(q => q.addEventListener("click", () => setTimeout(() => show(+q.dataset.stage), 500)));

  (async () => {
    for (const st of STAGES) st.n = await count(st);
    const start = ONLY === null ? 0 : +ONLY;
    show(start, false);
    if (!STATIC) onVisible($("#stages"), () => show(cur), 0.3);
  })();
})();
