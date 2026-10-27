/* Shared helpers: DOM, timed players, visibility triggers, tooltip, icons. */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const html = s => { const t = document.createElement("template"); t.innerHTML = s.trim(); return t.content.firstElementChild; };
// ?static=1 (or reduced motion) shows every animation in its final state at once
const STATIC = new URLSearchParams(location.search).has("static") || matchMedia("(prefers-reduced-motion: reduce)").matches;

if (STATIC) document.documentElement.classList.add("static");
const STOP = Symbol("stop");
/* A player runs one async script; play() restarts it, and any older run stops at its next wait. */
function player(reset, script) {
  let gen = 0;
  return {
    async play() {
      const my = ++gen;
      reset();
      const wait = ms => new Promise((ok, no) => setTimeout(() => (my === gen ? ok() : no(STOP)), STATIC ? 0 : ms));
      try { await script(wait); return true; } catch (e) { if (e !== STOP) throw e; return false; }
    },
    stop() { gen++; },
  };
}

/* run fn once when el scrolls into view */
function onVisible(el, fn, threshold = 0.35) {
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { io.disconnect(); fn(); } }), {threshold});
  io.observe(el);
}

const on = (el, v = true) => el && el.classList.toggle("on", v);
const offAll = root => $$(".fx", root).forEach(e => e.classList.remove("on"));

/* tooltip */
const tip = $("#tip");
function bindTip(el, text) {
  el.addEventListener("mouseenter", () => { tip.innerHTML = typeof text === "function" ? text() : text; tip.classList.add("on"); });
  el.addEventListener("mousemove", e => {
    const w = tip.offsetWidth, x = Math.min(e.clientX + 14, innerWidth - w - 8);
    tip.style.left = x + "px"; tip.style.top = (e.clientY + 16) + "px";
  });
  el.addEventListener("mouseleave", () => tip.classList.remove("on"));
}

/* icons */
const ICON = {
  check: (c = "#3f8f4f") => `<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" fill="${c}" opacity=".14"/><path d="M7 12.5l3.2 3.2L17 9" fill="none" stroke="${c}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  cross: (c = "#c0443f") => `<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" fill="${c}" opacity=".14"/><path d="M8.5 8.5l7 7M15.5 8.5l-7 7" fill="none" stroke="${c}" stroke-width="2.4" stroke-linecap="round"/></svg>`,
  test: (ok = true) => `<svg viewBox="0 0 24 24" class="icon"><rect x="3" y="4" width="18" height="16" rx="2.5" fill="#fff" stroke="#2b2b2b" stroke-width="1.5"/><path d="M3 8.5h18" stroke="#2b2b2b" stroke-width="1.5"/>${ok
    ? `<path d="M8 14l2.6 2.6L16.5 11" fill="none" stroke="#3f8f4f" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>`
    : `<path d="M9.2 11.4l5.6 5.6M14.8 11.4l-5.6 5.6" stroke="#c0443f" stroke-width="2" stroke-linecap="round"/>`}</svg>`,
  testBlank: `<svg viewBox="0 0 24 24" class="icon"><rect x="3" y="4" width="18" height="16" rx="2.5" fill="#fff" stroke="#2b2b2b" stroke-width="1.5"/><path d="M3 8.5h18" stroke="#2b2b2b" stroke-width="1.5"/></svg>`,
  judge: (fill = "#f8e3a0") => `<svg viewBox="0 0 24 24" class="icon"><path d="M12 3.5v15M7.5 18.5h9M4.5 6.5h15" stroke="#2b2b2b" stroke-width="1.4" stroke-linecap="round" fill="none"/><path d="M2.4 12 4.5 6.5 6.6 12Q4.5 14.6 2.4 12Z" fill="${fill}" stroke="#2b2b2b" stroke-width="1.2" stroke-linejoin="round"/><path d="M17.4 12 19.5 6.5 21.6 12Q19.5 14.6 17.4 12Z" fill="${fill}" stroke="#2b2b2b" stroke-width="1.2" stroke-linejoin="round"/></svg>`,
  star: (fill = "#7fb4e6") => `<svg viewBox="0 0 24 24" class="star"><path d="M12 2.2l2.9 6.6 7.1.6-5.4 4.7 1.6 7-6.2-3.7-6.2 3.7 1.6-7L2 9.4l7.1-.6z" fill="${fill}" stroke="#2a2a2a" stroke-width="1.2" stroke-linejoin="round"/></svg>`,
  container: `<svg viewBox="0 0 24 24"><rect x="2.5" y="4" width="19" height="15" rx="2.5" fill="#1e2230"/><path d="M6 9l3 2.5L6 14M11 14h5" stroke="#8fd18f" stroke-width="1.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
};
const tri = (size = 1) => `<span class="tri" style="transform:scale(${size})"></span>`;

/* misbehavior palette (rank #1 .. #8), as in the paper figures */
const PAL = ["#4A2662", "#6E3469", "#91416D", "#B04F6C", "#C86870", "#D78B7B", "#D8AA9A", "#D2C6BF"];
const mix = (h1, h2, t) => "#" + [1, 3, 5].map(j => Math.round(parseInt(h1.slice(j, j + 2), 16) * (1 - t) + parseInt(h2.slice(j, j + 2), 16) * t).toString(16).padStart(2, "0")).join("");

/* nav: show after the hero, highlight the current section */
(() => {
  const nav = $("#nav"), hero = $(".hero h1");
  const io = new IntersectionObserver(([e]) => nav.classList.toggle("show", !e.isIntersecting && e.boundingClientRect.top < 0));
  io.observe(hero);
  const links = $$(".nav .links a"), secs = links.map(a => $(a.getAttribute("href")));
  addEventListener("scroll", () => {
    let cur = -1; secs.forEach((s, i) => { if (s.getBoundingClientRect().top < 140) cur = i; });
    links.forEach((a, i) => a.classList.toggle("cur", i === cur));
  }, {passive: true});
})();

/* BibTeX copy */
$("#copy").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText($("#bib-text").textContent); $("#copy").textContent = "Copied"; }
  catch { $("#copy").textContent = "Select and copy"; }
  setTimeout(() => ($("#copy").textContent = "Copy"), 1600);
});
