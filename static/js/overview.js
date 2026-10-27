/* Teaser (Figure 1b): where rollouts start, and what the reward checks. Both rows run at the same pace per step. */
(() => {
  const root = $("#ov");
  const SEQ = ["instruction", "step 1", "step 2", "…", "step <i>k</i>", "step <i>k</i>+1", "…", "step <i>n</i>"];
  const K = 4, K1 = 5, N = 7;
  const ro = `<i>r</i><sub>o</sub>`, re = `<i>r</i><sub>e</sub>`;

  const row = (cls, head, reward) => html(`
    <div class="ov-row ${cls}">
      <div class="ov-head">${head}</div>
      <div class="ov-steps ${cls === "mirl" ? "has-brace" : ""}">
        ${SEQ.map((s, i) => `<span class="pill ${i === 0 ? "inst given" : ""} ${s === "…" ? "dots" : ""}" data-i="${i}"><span>${s}</span><span class="start">${tri(0.9)}</span></span>`).join("")}
        ${cls === "mirl" ? `<span class="brace"></span>` : ""}
      </div>
      <div class="ov-foot" style="position:relative">
        <span class="reward-tag fx">${reward}</span>
        ${cls === "mirl" ? `<span class="ov-judge fx pop" style="position:absolute">${ICON.judge()}<span>behavior ${re}</span></span>` : ""}
        <span class="ov-test fx pop" style="position:absolute"><span>outcome ${ro}</span>${ICON.test(true)}</span>
      </div>
    </div>`);

  const std = row("std", `<span class="who">Standard RL (e.g., GRPO, DPPO):</span> rollout start ${tri()} at step 1`,
    `reward = <span class="i" style="color:var(--ink-3)">only</span> outcome ${ro}`);
  const mirl = row("mirl", `<span class="who">MiRL (ours):</span> rollout start ${tri()} at <span class="mk">misbehavior-prone step <i>k</i></span>`,
    `reward = behavior ${re}<span class="and">∧</span>outcome ${ro}`);
  root.append(std, mirl);

  const pills = r => $$(".ov-steps .pill", r);
  // place the brace under steps k..k+1, the judge under the brace, and each test icon under step n
  function layout() {
    [std, mirl].forEach(r => {
      const ps = pills(r), foot = $(".ov-foot", r), t = $(".ov-test", r);
      const pn = ps[N]; t.style.left = (pn.offsetLeft + pn.offsetWidth - t.offsetWidth) + "px"; t.style.top = "2px";
    });
    const ps = pills(mirl), b = $(".brace", mirl), a = ps[K], c = ps[K1];
    b.style.left = (a.offsetLeft + 2) + "px"; b.style.width = (c.offsetLeft + c.offsetWidth - a.offsetLeft - 4) + "px";
    const j = $(".ov-judge", mirl), mid = a.offsetLeft + (c.offsetLeft + c.offsetWidth - a.offsetLeft) / 2;
    j.style.left = (mid - j.offsetWidth / 2) + "px"; j.style.top = "4px";
  }
  addEventListener("resize", layout);
  document.fonts.ready.then(layout);

  const reset = () => {
    [std, mirl].forEach(r => {
      offAll(r);
      pills(r).forEach((p, i) => { p.classList.remove("gen", "started", "mis", "given"); if (i === 0) p.classList.add("given"); });
    });
    $(".brace", mirl).classList.remove("on");
    pills(mirl)[K].classList.add("mis-label");
    layout();
  };

  const P = player(reset, async wait => {
    const STEP = 430;
    await wait(500);
    // MiRL restores the prefix (given, not generated); standard RL has nothing to restore
    pills(mirl).slice(1, K).forEach(p => p.classList.add("given"));
    await wait(450);
    pills(std)[1].classList.add("started");
    pills(mirl)[K].classList.add("started");
    await wait(350);
    const run = async (r, from, extra) => {
      const ps = pills(r);
      for (let i = from; i <= N; i++) {
        ps[i].classList.add(r === mirl && i === K ? "mis" : "gen");
        if (extra) await extra(i);
        await wait(STEP);
      }
      on($(".ov-test", r));
      await wait(500);
      on($(".reward-tag", r));
    };
    await Promise.all([
      run(std, 1),
      run(mirl, K, async i => { if (i === K1) { await wait(STEP * 0.6); $(".brace", mirl).classList.add("on"); on($(".ov-judge", mirl)); } }),
    ]);
  });

  $("#fig-ov .replay").addEventListener("click", () => P.play());
  reset();
  onVisible($("#fig-ov"), () => P.play(), 0.5);
})();
