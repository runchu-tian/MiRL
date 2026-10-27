/* Result figures. Only steps with an order are animated (before/after rates, runs); the rest are static. */

/* ------------------------------------------------------------ Figure 4: distribution of error types */
(() => {
  const TYPES = [   // name, % of error steps (Table 3), % of environments (Table 4)
    ["Makes false claims about tests", 15.4, 46.7],
    ["Repeats unnecessary checks", 11.5, 42.2],
    ["Retries without fixing the cause", 11.2, 36.5],
    ["Fails to produce runnable commands", 5.1, 19.4],
    ["Kills processes without checking their identities", 4.7, 14.0],
    ["Treats guesses as facts", 2.5, 11.0],
    ["Claims success without verification", 1.7, 8.9],
    ["Prints a binary file as text", 1.4, 6.9],
  ];
  const TAIL = Array.from({length: 18}, (_, i) => 0.95 * Math.exp(-i / 5) + 0.048);   // schematic long tail, below 1%
  const root = $("#dist"), fig = $("#fig-dist");
  root.innerHTML = `
    <div class="dchart"><div class="dbars"></div><div class="dbrk"><span></span></div></div>
    <div>
      <div class="dlist"></div>
      <div class="dtotal"></div>
      <div class="dtail"><b>Long-tailed, specific error types</b><ul><li>Uses a greedy regex on log lines</li><li>Replaces udb inside udbuf</li><li>Divides all temperatures by 100</li></ul></div>
    </div>`;
  const bars = $(".dbars", root), list = $(".dlist", root);
  const B = TYPES.map(([name], i) => {
    const b = html(`<div class="dbar" style="--c:${PAL[i]}"><span class="val"></span></div>`); bars.append(b);
    const r = html(`<div class="drow"><span class="dot" style="--c:${PAL[i]}"></span><span class="r">#${i + 1}</span><span>${name}</span><span class="v"></span></div>`); list.append(r);
    const hot = v => { r.classList.toggle("hot", v); B.forEach((x, j) => x.b.classList.toggle("dim", v && j !== i)); };
    [b, r].forEach(e => { e.addEventListener("mouseenter", () => hot(true)); e.addEventListener("mouseleave", () => hot(false)); });
    return {b, r};
  });
  const T = TAIL.map((v, i) => { const b = html(`<div class="dbar tail" style="--c:${mix(PAL[7], "#FFFFFF", Math.min(1, i / 14))}"></div>`); bars.append(b); return [b, v]; });
  T.forEach(([b, v]) => bindTip(b, `Long-tail error type: <b>&lt; 1%</b> of error steps (schematic)`));
  const brk = $(".dbrk", root);
  function setMode(m) {
    const env = m === "env", max = env ? 50 : 16;
    root.classList.toggle("env", env);
    B.forEach(({b, r}, i) => {
      const v = TYPES[i][env ? 2 : 1];
      b.style.height = (v / max * 100) + "%"; $(".val", b).textContent = v.toFixed(1); $(".v", r).textContent = v.toFixed(1) + "%";
    });
    T.forEach(([b, v]) => (b.style.height = env ? "0" : (v / max * 100) + "%"));
    $(".dtotal", root).innerHTML = env ? `At least one of the top 8 appears in <b>75.4%</b> of the environments.` : `Top-8 misbehaviors: <b>52.1%</b> of the error steps.`;
    const last = B[7].b; brk.style.width = (last.offsetLeft + last.offsetWidth) + "px";
    $("span", brk).textContent = env ? "top-8 misbehaviors, % of environments" : "top-8 misbehaviors, % of error steps";
    $$("#dist-seg button").forEach(x => x.classList.toggle("on", x.dataset.m === m));
  }
  $$("#dist-seg button").forEach(x => x.addEventListener("click", () => setMode(x.dataset.m)));
  document.fonts.ready.then(() => setMode("steps"));
  addEventListener("resize", () => setMode(root.classList.contains("env") ? "env" : "steps"));
})();

/* ------------------------------------------------------------ shared horizontal bar rows (Figures 5 and 6) */
const PLOTS = {"TB-Lite": [20, 55, [20, 30, 40, 50]], "TB-2.1": [10, 23.4, [10, 15, 20]]};
function barCell(name, [m, sd], color, o = {}) {
  const [lo, hi, ticks] = PLOTS[name], X = v => ((v - lo) / (hi - lo) * 100);
  const c = html(`<div class="track">${ticks.filter(t => t > lo).map(t => `<span class="grid" style="left:${X(t)}%"></span>`).join("")}
    <div class="bar ${o.rep ? "rep" : ""} ${o.best ? "best" : ""}" style="--c:${color};width:${X(m)}%"><span class="num">${m.toFixed(1)}<small>±${sd.toFixed(1)}</small></span></div>
    <span class="err" style="left:${X(m - sd)}%;width:${X(m + sd) - X(m - sd)}%"></span></div>`);
  c.set = ([m2, sd2]) => {
    $(".bar", c).style.width = X(m2) + "%";
    const e = $(".err", c); e.style.left = X(m2 - sd2) + "%"; e.style.width = (X(m2 + sd2) - X(m2 - sd2)) + "%";
  };
  c.x = X;
  return c;
}
// a number that does not fit inside its bar moves to the right of the error bar
function fitNums(root) {
  $$(".track", root).forEach(tr => {
    const bar = $(".bar", tr), num = $(".num", bar), err = $(".err", tr);
    if (!bar || !num || bar.closest(".row-hidden")) return;
    num.style.left = "8px";
    const room = err.offsetLeft - 6;
    if (num.offsetWidth + 12 > room) num.style.left = (err.offsetLeft + err.offsetWidth + 6) + "px";
  });
}
function axisRow(grid) {
  ["TB-Lite", "TB-2.1"].forEach((n, k) => {
    if (k === 0) grid.append(html(`<div></div>`));
    const [lo, hi, ticks] = PLOTS[n];
    grid.append(html(`<div class="axis">${ticks.map(t => `<span style="left:${(t - lo) / (hi - lo) * 100}%">${t}</span>`).join("")}</div>`));
  });
  grid.append(html(`<div class="axis-t">Score (%)</div>`));
}

/* ------------------------------------------------------------ Figure 5: main results (static) */
(() => {
  const ROWS = [   // label, color, reported?, TB-Lite, TB-2.1, gap before
    ["Qwen3.5-4B", "#FFFFFF", true, [31.8, 3.8], [14.2, 2.3], 0], ["Qwen3.5-4B", "#FFFFFF", false, [32.8, 1.5], [16.1, 1.0], 0],
    ["TMax-4B", "#AFAFAF", true, [42.6, 1.5], [19.9, 1.1], 1], ["TMax-4B", "#AFAFAF", false, [47.4, 0.9], [20.6, 1.3], 0],
    ["TMax-189", "#D9D9D9", false, [43.4, 1.7], [17.6, 0.4], 1], ["PivotRL", "#CCD8C2", false, [47.2, 0.8], [18.7, 1.0], 0],
    ["RaR+", "#E4DBC0", false, [47.6, 1.6], [19.1, 2.6], 0], ["MiRL (ours)", "#7FB4E6", false, [51.8, 0.1], [21.3, 0.7], 1],
  ];
  const g = $("#main-bars");
  g.append(html(`<div></div>`), html(`<div class="colh">TB-Lite</div>`), html(`<div class="colh">TB-2.1</div>`));
  ROWS.forEach(([n, c, rep, a, b, gap]) => {
    if (gap) g.append(html(`<div class="gap"></div>`));
    const best = n.startsWith("MiRL");
    g.append(html(`<div class="lab ${best ? "ours" : ""}">${n}${rep ? ` <small>(reported)</small>` : ""}</div>`));
    [["TB-Lite", a], ["TB-2.1", b]].forEach(([p, v]) => {
      const cell = barCell(p, v, c, {rep, best}); g.append(cell);
      bindTip($(".bar", cell), `<b>${n}</b>${rep ? " (reported in the TMax paper)" : ""}<br>${p}: ${v[0].toFixed(1)} ± ${v[1].toFixed(1)}`);
    });
  });
  axisRow(g);
  document.fonts.ready.then(() => fitNums(g)); addEventListener("resize", () => fitNums(g));
})();

/* ------------------------------------------------------------ Figure 6: ablation (static) */
(() => {
  const ROWS = [   // label, color, TB-Lite, TB-2.1
    ["Qwen3.5-4B (no RL)", "#FFFFFF", [32.8, 1.5], [16.1, 1.0]],
    ["TMax-189", "#D6D6D6", [43.4, 1.7], [17.6, 0.4]],
    ["+ env. selection", "#DCE6EF", [37.4, 2.2], [16.5, 2.6]],
    ["+ prefix start", "#BCD6EE", [45.3, 1.6], [18.0, 0.6]],
    ["+ <i>r</i><sub>e</sub> (MiRL)", "#8DBFEA", [51.8, 0.1], [21.3, 0.7]],
  ];
  const g = $("#abl-bars");
  g.append(html(`<div></div>`), html(`<div class="colh">TB-Lite</div>`), html(`<div class="colh">TB-2.1</div>`));
  ROWS.forEach(([n, c, a, b], i) => {
    g.append(html(`<div class="lab ${i === 4 ? "ours" : ""}">${n}</div>`));
    [["TB-Lite", a], ["TB-2.1", b]].forEach(([p, v]) => {
      const cell = barCell(p, v, c, {best: i === 4}); g.append(cell);
      bindTip($(".bar", cell), `<b>${n}</b><br>${p}: ${v[0].toFixed(1)} ± ${v[1].toFixed(1)}`);
    });
  });
  axisRow(g);
  document.fonts.ready.then(() => fitNums(g)); addEventListener("resize", () => fitNums(g));
})();

/* ------------------------------------------------------------ Figure 7: misbehavior rate before and after training */
(() => {
  const ROWS = [[1, "Makes false claims about tests", PAL[0]], [3, "Retries without fixing the cause", PAL[2]],
                [4, "Fails to produce runnable commands", PAL[3]], [8, "Prints a binary file as text", PAL[7]]];
  const JUDGES = [   // per row: [Qwen3.5-4B, TMax-4B, MiRL]
    [[26.7, 22.5, 8.1], [20.2, 12.8, 13.0], [10.5, 0.2, 0.2], [4.1, 21.9, 5.3]],
    [[29.8, 28.4, 11.7], [29.7, 35.2, 23.9], [8.8, 0.8, 0.9], [4.6, 23.8, 5.3]],
  ];
  const LO = -2, HI = 42, X = v => (v - LO) / (HI - LO) * 100, TICKS = [0, 10, 20, 30, 40];
  const root = $("#rate"), fig = $("#fig-rate");
  root.append(html(`<div class="rate-legend"><span><span style="display:inline-block;width:22px;border-top:1.6px dashed #5a5a5a"></span>Qwen3.5-4B</span><span><span class="sq"></span>TMax-4B</span><span>${ICON.star()}<b style="color:var(--blue)">MiRL (ours)</b></span></div>`));
  const R = ROWS.map(([rank, name, c]) => {
    const r = html(`<div class="rrow"><div class="nm"><span class="dot" style="background:${c}"></span>${name}</div>
      <div class="rtrack">${TICKS.map(t => `<span class="gl" style="left:${X(t)}%"></span>`).join("")}
        <div class="rbase"><span></span></div>
        <div class="rline t"></div><div class="rline m"></div>
        <div class="rmk t"><span class="sq"></span><span class="lv"></span></div>
        <div class="rmk m">${ICON.star()}<span class="lv"></span></div></div></div>`);
    root.append(r); return r;
  });
  root.append(html(`<div class="rate-axis"><div></div><div class="axis">${TICKS.map(t => `<span style="left:${X(t)}%">${t}</span>`).join("")}</div><div></div><div class="t">Misbehavior rate (%, ↓)</div></div>`));
  let judge = 0;
  // after = false puts both trained models at the base rate (before training)
  function set(j, after) {
    judge = j;
    R.forEach((r, i) => {
      const [b, t, m] = JUDGES[j][i];
      const base = $(".rbase", r); base.style.left = X(b) + "%";
      const bl = $("span", base); bl.textContent = b.toFixed(1);
      const leftSide = (m < b && t < b) ? false : X(b) > 60; bl.style.left = leftSide ? "auto" : "6px"; bl.style.right = leftSide ? "6px" : "auto";
      [["t", t], ["m", m]].forEach(([k, v]) => {
        const val = after ? v : b, mk = $(`.rmk.${k}`, r), ln = $(`.rline.${k}`, r), lv = $(".lv", mk);
        mk.style.left = X(val) + "%";
        const gapPx = 11, d = Math.abs(X(val) - X(b));
        ln.style.left = val > b ? X(b) + "%" : `calc(${X(val)}% + ${gapPx}px)`;
        ln.style.width = `max(0px, calc(${d}% - ${gapPx}px))`;
        ln.classList.toggle("right", val > b); ln.classList.toggle("left", val <= b); ln.classList.toggle("flat", Math.abs(val - b) < 1.2);
        lv.textContent = after ? v.toFixed(1) : "";
        lv.style.top = "50%"; lv.style.transform = "translateY(-50%)";
        if (v >= b) { lv.style.left = "calc(100% + 6px)"; lv.style.right = "auto"; }
        else if (X(v) > 9) { lv.style.left = "auto"; lv.style.right = "calc(100% + 6px)"; }
        else { lv.style.left = "50%"; lv.style.right = "auto"; lv.style.transform = "translateX(-50%)"; lv.style.top = k === "t" ? "-15px" : "19px"; }
      });
    });
    $$("#rate-seg button").forEach(x => x.classList.toggle("on", +x.dataset.j === j));
  }
  const P = player(() => set(judge, false), async wait => { await wait(500); set(judge, true); });
  $$("#rate-seg button").forEach(x => x.addEventListener("click", () => set(+x.dataset.j, true)));
  $(".replay", fig).addEventListener("click", () => P.play());
  set(0, STATIC);
  if (!STATIC) onVisible(fig, () => P.play(), 0.45);
})();

/* ------------------------------------------------------------ Figure 8: pass@k over runs 1..16 */
(() => {
  const M = [   // key, name, line color, dashed, marker
    ["base", "Qwen3.5-4B", "#8A8A8A", true, "circle"],
    ["tmax", "TMax-4B", "#8A8A8A", false, "square"],
    ["mirl", "MiRL (ours)", "#2F65AE", false, "star"],
  ];
  const RUNS = window.PASSK_RUNS, CURVE = window.PASSK_CURVE, N_TASKS = 89;
  const cnt = (k, t, upto = 16) => [...RUNS[k][t].slice(0, upto)].filter(c => c === "1").length;
  const all = Object.keys(RUNS.mirl);
  const solved = all.filter(t => M.some(([k]) => cnt(k, t) > 0));
  const tot = t => M.reduce((s, [k]) => s + cnt(k, t), 0);
  solved.sort((a, b) => tot(b) - tot(a) || cnt("mirl", b) - cnt("mirl", a));
  const root = $("#pk"), fig = $("#fig-pk");

  // (a) curve
  const W = 520, H = 300, L = 46, Rr = 54, T = 14, B = 40, PW = W - L - Rr, PH = H - T - B, Y0 = 12, Y1 = 42;
  const sx = k => L + (k - 1) / 15 * PW, sy = v => T + (Y1 - v) / (Y1 - Y0) * PH;
  const mk = (kind, x, y, s = 1) => kind === "circle" ? `<circle cx="${x}" cy="${y}" r="${4.4 * s}" fill="#fff" stroke="#2a2a2a" stroke-width="1.2"/>`
    : kind === "square" ? `<rect x="${x - 4.2 * s}" y="${y - 4.2 * s}" width="${8.4 * s}" height="${8.4 * s}" fill="#AFAFAF" stroke="#2a2a2a" stroke-width="1.2"/>`
    : `<path transform="translate(${x - 8 * s},${y - 8.2 * s}) scale(${0.667 * s})" d="M12 2.2l2.9 6.6 7.1.6-5.4 4.7 1.6 7-6.2-3.7-6.2 3.7 1.6-7L2 9.4l7.1-.6z" fill="#7FB4E6" stroke="#2a2a2a" stroke-width="1.6" stroke-linejoin="round"/>`;
  let svg = `<svg viewBox="0 0 ${W} ${H}">
    <defs><clipPath id="pkclip"><rect id="pkclipr" x="0" y="0" width="${W}" height="${H}"/></clipPath></defs>`;
  [12, 18, 24, 30, 36, 42].forEach(v => (svg += `<line x1="${L}" x2="${L + PW}" y1="${sy(v)}" y2="${sy(v)}" stroke="#eceef2"/><text x="${L - 8}" y="${sy(v) + 4}" font-size="12" text-anchor="end" fill="#4a505d">${v}</text>`));
  [1, 4, 8, 12, 16].forEach(k => (svg += `<text x="${sx(k)}" y="${T + PH + 18}" font-size="12" text-anchor="middle" fill="#4a505d">${k}</text>`));
  svg += `<line x1="${L}" x2="${L + PW}" y1="${T + PH}" y2="${T + PH}" stroke="#2a2a2a"/>`;
  svg += `<text x="${L + PW / 2}" y="${H - 4}" font-size="12.5" text-anchor="middle" fill="#4a505d">k (# runs per task)</text>`;
  svg += `<text transform="translate(13,${T + PH / 2}) rotate(-90)" font-size="12.5" text-anchor="middle" fill="#4a505d">Unbiased pass@k (%)</text>`;
  svg += `<line id="pkk" x1="0" x2="0" y1="${T}" y2="${T + PH}" stroke="#2f65ae" stroke-opacity=".25" stroke-width="1.5" stroke-dasharray="3 3"/>`;
  svg += `<g clip-path="url(#pkclip)">`;
  M.forEach(([k, , c, dash]) => (svg += `<polyline points="${CURVE[k].map((v, i) => `${sx(i + 1)},${sy(v)}`).join(" ")}" fill="none" stroke="${c}" stroke-width="${k === "mirl" ? 2.6 : 2}" ${dash ? 'stroke-dasharray="5 4"' : ""}/>`));
  svg += `</g>`;
  M.forEach(([k, , , , kind]) => (svg += `<g class="pkm" data-k="${k}">${mk(kind, 0, 0)}</g><text class="pkl" data-k="${k}" font-size="12.5" font-weight="${k === "mirl" ? 700 : 500}" fill="${k === "mirl" ? "#2F65AE" : "#555"}"></text>`));
  svg += `</svg>`;
  root.append(html(`<div class="pk-top"><div>${svg}</div><div>
      <div class="pk-ctrl"><button class="play" aria-label="Play"><svg viewBox="0 0 24 24"><path d="M6 4l14 8-14 8z" fill="currentColor"/></svg></button><span>k =</span><span class="kv"></span><input type="range" min="1" max="16" step="1" value="16" aria-label="k"></div>
      <div class="pk-vals">${M.map(([k, n, , , kind]) => `<div class="pk-val ${k === "mirl" ? "m" : ""}" data-k="${k}"><svg viewBox="-10 -10 20 20" width="20" height="20">${mk(kind, 0, 0, 0.95)}</svg><span class="nm">${n}</span><span class="n"></span></div>`).join("")}</div>
      <div class="pk-note">Drag <i>k</i>, or press play to add one run at a time. The strip below shows which tasks are solved in runs 1 to <i>k</i>.</div>
    </div></div>`));

  // (b) strip: one cell per task, shade = number of runs 1..k that solve it
  const ramp = (lo, midc, hi) => c => c <= 0 ? "#FFFFFF" : (c <= 8 ? mix(lo, midc, (c - 1) / 7) : mix(midc, hi, (c - 8) / 8));
  const GREY = ramp("#EEEEEE", "#AFAFAF", "#404040"), BLUE = ramp("#E2EDF9", "#7FB4E6", "#1D4A8A");
  const strip = html(`<div><div class="strip-h">Pass rate of each task over runs 1–<span class="kk"></span></div><div class="strip"></div>
    <div class="strip-key"><span># runs that solve it:</span><span>Qwen3.5-4B / TMax-4B <span class="ramp" style="background:linear-gradient(90deg,#fff,${GREY(1)} 7%,${GREY(8)} 50%,${GREY(16)})"></span></span><span>MiRL <span class="ramp" style="background:linear-gradient(90deg,#fff,${BLUE(1)} 7%,${BLUE(8)} 50%,${BLUE(16)})"></span></span><span><span class="ramp" style="width:22px;background:repeating-linear-gradient(135deg,#fafafa 0 4px,#e8e8e8 4px 8px)"></span> ${N_TASKS - solved.length} tasks never solved</span></div></div>`);
  root.append(strip);
  const sg = $(".strip", strip);
  const cells = {};
  M.forEach(([k, n, , , kind]) => {
    sg.append(html(`<svg viewBox="-10 -10 20 20" width="20" height="20" style="justify-self:center">${mk(kind, 0, 0, 0.9)}</svg>`));
    const row = html(`<div class="cells">${solved.map(() => "<i></i>").join("")}<i class="zero"></i></div>`);
    sg.append(row);
    const c = html(`<div class="cnt"></div>`); sg.append(c);
    cells[k] = {is: $$("i:not(.zero)", row), c};
    cells[k].is.forEach((e, j) => bindTip(e, () => `<b>${solved[j]}</b><br>${n}: solved in ${cnt(k, solved[j], kNow)} of runs 1–${kNow}`));
  });

  let kNow = 16;
  const range = $("input", root);
  function setK(k) {
    kNow = k; range.value = k;
    $(".kv", root).textContent = k; $(".kk", strip).textContent = k;
    $("#pkclipr").setAttribute("width", sx(k) + 6);
    const kl = $("#pkk"); kl.setAttribute("x1", sx(k)); kl.setAttribute("x2", sx(k));
    M.forEach(([key]) => {
      const v = CURVE[key][k - 1], x = sx(k), y = sy(v);
      $(`.pkm[data-k="${key}"]`, root).setAttribute("transform", `translate(${x},${y})`);
      const lab = $(`.pkl[data-k="${key}"]`, root);
      lab.textContent = v.toFixed(1);
      // keep the three labels apart: MiRL above, the others below or to the right
      const dy = key === "mirl" ? -10 : key === "base" ? (CURVE.base[k - 1] > CURVE.tmax[k - 1] ? -8 : 16) : (CURVE.tmax[k - 1] >= CURVE.base[k - 1] ? -8 : 16);
      lab.setAttribute("x", x + 9); lab.setAttribute("y", y + dy);
      $(`.pk-val[data-k="${key}"] .n`, root).textContent = v.toFixed(1) + "%";
      const R = key === "mirl" ? BLUE : GREY;
      cells[key].is.forEach((e, j) => (e.style.background = R(cnt(key, solved[j], k))));
      const sv = all.filter(t => cnt(key, t, k) > 0).length;
      cells[key].c.innerHTML = `<b>${sv}</b>/${N_TASKS} solved`;
    });
  }
  let playing = false;
  const P = player(() => setK(1), async wait => { playing = true; for (let k = 2; k <= 16; k++) { await wait(420); setK(k); } playing = false; });
  range.addEventListener("input", () => { P.stop(); setK(+range.value); });
  $(".play", root).addEventListener("click", () => P.play());
  setK(16);
  if (!STATIC) onVisible(fig, () => P.play(), 0.4);
})();

/* ------------------------------------------------------------ Figure 9: training resources (static) */
(() => {
  const ROWS = [   // name, TMax-4B total, MiRL [error profiling, RL training], labels [TMax, MiRL], overlap?, scale
    ["Rollouts", 75852, [4034, 22145], ["75.9K", "26.2K"]],
    ["Input tokens", 30895.3, [1269.8, 9236.4], ["30.9B", "10.5B"]],
    ["Output tokens", 919.2, [38.93, 222.8], ["919M", "262M"]],
    ["Environments", 2370, [2017, 189], ["2,370", "2,017"], true],
    ["GPU hours", 526.6, [38.1, 164.7], ["526.6", "202.8"]],
    ["API fee ($)", 0, [603.7, 133.4], ["$0", "$737.1"], false, 3521.5],
    ["Total cost ($)", 3521.5, [752.0, 1208.3], ["$3,521.5", "$1,960.3"]],
  ];
  const root = $("#cost");
  root.append(html(`<div class="cost-legend"><span><span class="sw" style="background:#e6e6e6"></span>TMax-4B (100%)</span><span><span class="sw" style="background-color:#fff;background-image:linear-gradient(45deg,#6f9fd0 25%,transparent 25%,transparent 75%,#6f9fd0 75%),linear-gradient(45deg,#6f9fd0 25%,transparent 25%,transparent 75%,#6f9fd0 75%);background-size:6px 6px;background-position:0 0,3px 3px"></span>MiRL: error profiling</span><span><span class="sw" style="background:var(--blue-2)"></span>MiRL: RL training</span></div>`));
  const g = html(`<div class="cost"></div>`); root.append(g);
  ROWS.forEach(([n, tm, [p, t], [aT, aM], overlap, ref], i) => {
    const base = tm > 0 ? tm : ref, fp = p / base * 100, ft = t / base * 100, tot = overlap ? Math.max(p, t) / base : (p + t) / base;
    g.append(html(`<div class="lab ${i === ROWS.length - 1 ? "tot" : ""}">${n}<small>${aT} → <b>${aM}</b></small></div>`));
    const tr = html(`<div class="ctrack ${tm > 0 ? "" : "none"}">${overlap
      ? `<span class="p" style="width:${fp}%"></span><span class="r inner" style="left:0;width:${ft}%"></span>`
      : `<span class="p" style="width:${fp}%"></span><span class="r" style="left:${fp}%;width:${ft}%"></span>`}</div>`);
    g.append(tr);
    bindTip(tr, `<b>${n}</b><br>TMax-4B: ${aT}<br>MiRL: ${aM}${overlap ? " (RL training uses 189 of them)" : ""}`);
    g.append(html(`<div class="chg">${tm > 0 ? "−" + Math.round((1 - tot) * 100) + "%" : "+" + aM}</div>`));
  });
  g.append(html(`<div class="cost-axis">${[0, 25, 50, 75, 100].map(f => `<span style="left:${f}%">${f}%</span>`).join("")}</div>`));
  g.append(html(`<div class="cost-axis-t">MiRL cost as % of TMax-4B</div>`));
})();
