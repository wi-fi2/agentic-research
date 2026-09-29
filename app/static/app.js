"use strict";

const $ = (s) => document.querySelector(s);
const STAGES = [
  ["decompose", "Decompose question"],
  ["search", "Search the web"],
  ["fetch", "Read & index sources"],
  ["judge", "Judge relevance & injection risk"],
  ["reformulate", "Fill evidence gaps"],
  ["synthesize", "Write the brief"],
  ["verify", "Verify every claim"],
  ["finalize", "Validate citations"],
];
const EXAMPLES = [
  "Compare DuckDB vs Polars for out-of-core data processing on single nodes",
  "Current regulatory status and production capacity of sodium-ion battery manufacturers",
  "Key compliance deadlines of the EU AI Act for general-purpose AI models",
  "Best practices for mitigating cold-start latency in Python serverless containers",
];

const state = { config: null, runId: null, slug: null, report: null, shareUrl: null, es: null };

// ------------------------------------------------------------------ utils

function storageGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function storageSet(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } }

function headers(extra = {}) {
  const h = { ...extra };
  const t = storageGet("accessToken");
  if (t) h["X-Access-Token"] = t;
  return h;
}

async function api(path, opts = {}) {
  const res = await fetch(path, { ...opts, headers: headers(opts.headers || {}) });
  const body = res.headers.get("content-type")?.includes("json") ? await res.json() : await res.text();
  if (res.status === 401 && state.config?.requires_token) {
    const t = prompt("This instance needs an access token:");
    if (t) { storageSet("accessToken", t); return api(path, opts); }
  }
  if (!res.ok) throw new Error(body?.error || `HTTP ${res.status}`);
  return body;
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), 2200);
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function show(id) {
  for (const s of ["#composer", "#progress", "#reportView"]) $(s).hidden = s !== id;
}

// ------------------------------------------------------------------ boot

async function boot() {
  EXAMPLES.forEach((q) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = q;
    b.onclick = () => { $("#question").value = q; $("#question").focus(); };
    $("#examples").appendChild(b);
  });

  try {
    state.config = await api("/api/config");
    const c = state.config;
    $("#providers").innerHTML = [
      `<span class="pill ${c.llm_providers.length ? "on" : "off"}">LLM: ${esc(c.llm_providers.join(", ") || "none")}</span>`,
      `<span class="pill on">Search: ${esc(c.search_providers.join(", "))}</span>`,
      `<span class="pill ${c.laya === "ready" ? "on" : c.laya === "error" ? "off" : ""}">Laya: ${esc(c.laya)}</span>`,
    ].join("");
  } catch { /* shared view still works */ }
  loadModelPicker();

  const path = location.pathname;
  let m;
  if ((m = path.match(/^\/r\/([\w-]+)/))) return loadShared(m[1]);
  if ((m = path.match(/^\/run\/(\w+)/))) return loadRun(m[1]);
  show("#composer");
}

// ------------------------------------------------------------------ model picker

const MODEL_PICKERS = [["#writerModel", "writerModel"], ["#helperModel", "helperModel"]];

async function loadModelPicker() {
  let providers;
  try {
    ({ providers } = await api("/api/models"));
  } catch { return; }  // picker is optional; runs use the server's default chain
  const names = Object.keys(providers).filter((p) => providers[p].length);
  if (!names.length) return;
  const groups = names.map((p) =>
    `<optgroup label="${esc(p)}">` + providers[p].map((m) =>
      `<option value="${esc(JSON.stringify({ provider: p, model: m }))}">${esc(m)}</option>`).join("") + "</optgroup>").join("");
  for (const [sel, key] of MODEL_PICKERS) {
    const el = $(sel);
    el.innerHTML = `<option value="">Auto (server default)</option>` + groups;
    const saved = storageGet(key);
    if (saved && [...el.options].some((o) => o.value === saved)) el.value = saved;
    el.onchange = () => storageSet(key, el.value);
  }
  $("#modelRow").hidden = false;
}

function modelPick(sel) {
  const v = $(sel).value;
  return v ? JSON.parse(v) : undefined;
}

// ------------------------------------------------------------------ start + progress

$("#form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("#question").value.trim();
  if (question.length < 8) return toast("Please write a fuller question.");
  const depth = document.querySelector("input[name=depth]:checked").value;
  $("#submitBtn").disabled = true;
  try {
    const { run_id } = await api("/api/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, depth, writer: modelPick("#writerModel"), helper: modelPick("#helperModel") }),
    });
    history.pushState({}, "", `/run/${run_id}`);
    startProgress(run_id, question);
  } catch (err) {
    toast(err.message);
  } finally {
    $("#submitBtn").disabled = false;
  }
});

function startProgress(runId, question) {
  state.runId = runId;
  show("#progress");
  $("#progressQuestion").textContent = question;
  $("#errorBox").hidden = true;
  $("#iteration").textContent = "";
  $("#stages").innerHTML = STAGES.map(([k, label]) =>
    `<li data-stage="${k}"><span class="dot"></span><div><b>${label}</b><small></small></div></li>`).join("");
  $("#facets").innerHTML = "";
  $("#sourcesLive").innerHTML = "";
  renderStats({});
  if (state.es) state.es.close();
  const es = (state.es = new EventSource(`/api/research/${runId}/events`));
  es.onmessage = (msg) => handleEvent(JSON.parse(msg.data));
  es.onerror = () => { /* browser auto-reconnects; server replays backlog */ };
}

function handleEvent(ev) {
  switch (ev.type) {
    case "stage": {
      const li = document.querySelector(`[data-stage="${ev.stage}"]`);
      if (li) { li.className = ev.status; li.querySelector("small").textContent = ev.detail || ""; }
      break;
    }
    case "iteration":
      $("#iteration").textContent = `Search loop ${ev.n} of ${ev.max}`;
      if (ev.n > 1) ["search", "fetch", "judge"].forEach((s) => {
        const li = document.querySelector(`[data-stage="${s}"]`);
        if (li) li.className = "";
      });
      break;
    case "plan":
    case "coverage":
      renderFacets(ev.sub_queries);
      break;
    case "source": {
      const li = document.createElement("li");
      li.textContent = `${ev.method === "snippet" ? "◌" : "●"} ${ev.title || ev.url}`;
      li.title = ev.url;
      $("#sourcesLive").prepend(li);
      break;
    }
    case "metrics":
      renderStats(ev.metrics);
      break;
    case "done":
      state.es?.close();
      loadRun(ev.run_id);
      break;
    case "error":
      state.es?.close();
      $("#errorBox").textContent = ev.message;
      $("#errorBox").hidden = false;
      break;
  }
}

function renderFacets(sqs = []) {
  $("#facets").innerHTML = sqs.map((s) =>
    `<li>${esc(s.query_text)}${(s.types || []).map((t) => ` <span class="route">${esc(t)}</span>`).join("")}<div class="bar"><i style="width:${Math.round((s.coverage || 0) * 100)}%"></i></div></li>`).join("");
}

function renderStats(m) {
  const cost = m.total_cost_usd ?? 0;
  const cells = [
    [m.pages_fetched ?? 0, "pages read"],
    [m.passages_indexed ?? 0, "passages"],
    [m.duplicates_removed ?? 0, "duplicates"],
    [m.llm_calls ?? 0, "LLM calls"],
    [(m.judge_input_tokens ?? 0).toLocaleString(), "Laya tokens (local)"],
    [`$${cost.toFixed(4)}`, "cost so far"],
  ];
  $("#stats").innerHTML = cells.map(([v, l]) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`).join("");
}

// ------------------------------------------------------------------ report

async function loadRun(runId) {
  try {
    const run = await api(`/api/research/${runId}`);
    state.runId = runId;
    state.slug = null;
    if (run.status === "running") return startProgress(runId, run.question);
    if (run.status === "error") {
      startProgress(runId, run.question);
      return;
    }
    state.shareUrl = run.share_slug ? `${location.origin}/r/${run.share_slug}` : null;
    renderReport(run.report, false);
  } catch (err) {
    toast(err.message);
    show("#composer");
  }
}

async function loadShared(slug) {
  try {
    const data = await api(`/api/shared/${slug}`);
    state.slug = slug;
    state.shareUrl = location.href;
    renderReport(data.report, true);
  } catch (err) {
    show("#composer");
    toast(err.message);
  }
}

function renderMarkdown(mdText, sources) {
  const titles = Object.fromEntries(sources.map((s) => [s.source_id, s.title]));
  let html = marked.parse(mdText, { mangle: false, headerIds: false });
  html = DOMPurify.sanitize(html);
  // Citation chips and verification flags (applied after sanitising; inputs are fixed patterns).
  html = html.replace(/\[(S\d+)\]/g, (_, id) =>
    titles[id] ? `<a class="cite" href="#ref-${id}" title="${esc(titles[id])}">${id}</a>` : `[${id}]`);
  html = html.replace(/<em>\[Unverified\]<\/em>/g, '<span class="flag unverified" title="The cited source did not clearly support this sentence">unverified</span>');
  html = html.replace(/<em>\[Conflicting evidence\]<\/em>/g, '<span class="flag conflict" title="Sources disagree on this point">conflicting</span>');
  html = html.replace(/\[UNVERIFIED: ([^\]]+)\]/g, (_, why) => `<span class="flag missing">${esc(why.toLowerCase())}</span>`);
  return html;
}

function renderReport(report, shared) {
  state.report = report;
  document.title = `${report.title} · Agentic Research`;
  show("#reportView");
  $("#sharedBanner").hidden = !shared;
  $("#shareBtn").textContent = shared ? "Share link" : "Share";

  const art = $("#report");
  art.innerHTML = renderMarkdown(report.markdown, report.sources);
  // Anchor the bibliography entries so citation chips can jump to them.
  art.querySelectorAll("li").forEach((li) => {
    const m = li.textContent.match(/^\[(S\d+)\]/);
    if (m) li.id = `ref-${m[1]}`;
  });
  art.querySelectorAll('a[href^="http"]').forEach((a) => { a.target = "_blank"; a.rel = "noopener noreferrer"; });
  art.querySelectorAll("a.cite").forEach((a) => a.addEventListener("click", () => {
    const target = document.getElementById(a.getAttribute("href").slice(1));
    if (target) { target.classList.add("flash"); setTimeout(() => target.classList.remove("flash"), 1400); }
  }));

  const claims = report.claims || [];
  const n = (s) => claims.filter((c) => c.status === s).length;
  const m = report.metrics || {};
  const mode = { laya: "Verified by Laya", "laya+llm": "Verified by Laya + LLM", llm: "Verified by LLM", none: "Not verified" }[report.verification_mode] || "";
  $("#reportMeta").innerHTML = [
    `<span class="badge ${report.verification_mode === "none" ? "warn" : "ok"}">${mode}</span>`,
    `<span class="badge ok">${n("supported")} supported</span>`,
    n("uncertain") ? `<span class="badge bad">${n("uncertain")} conflicting</span>` : "",
    n("unverified") ? `<span class="badge warn">${n("unverified")} unverified</span>` : "",
    n("refuted") ? `<span class="badge bad">${n("refuted")} removed</span>` : "",
    `<span class="badge">${report.sources.length} sources</span>`,
  ].join("");

  const cost = m.total_cost_usd ?? (m.llm_cost_usd || 0);
  $("#reportSide").innerHTML = `
    <div class="card"><h3>Run</h3><div class="kv">
      <span>Duration</span><span>${(m.duration_seconds || 0).toFixed(0)} s</span>
      <span>Search loops</span><span>${m.iterations || 0}</span>
      <span>Evidence coverage</span><span>${Math.round((m.coverage || 0) * 100)}%</span>
      <span>Pages read</span><span>${m.pages_fetched || 0}</span>
      <span>Passages indexed</span><span>${m.passages_indexed || 0}</span>
      <span>Quarantined</span><span>${m.passages_quarantined || 0}</span>
      <span>LLM calls</span><span>${m.llm_calls || 0}</span>
      <span>Laya batches</span><span>${m.judge_calls || 0}</span>
      <span>Claims escalated to LLM</span><span>${m.claims_escalated || 0}</span>
      <span>Quotes found in source</span><span>${(m.quotes_exact || 0) + (m.quotes_fuzzy || 0)} / ${(m.quotes_exact || 0) + (m.quotes_fuzzy || 0) + (m.quotes_missing || 0)}</span>
      <span>Total cost</span><span>$${cost.toFixed(4)}</span>
    </div></div>
    <div class="card"><h3>Stopped because</h3>${esc(report.stop_reason || "—")}</div>
    ${report.warnings?.length ? `<div class="card"><h3>Quality notes</h3><ul>${report.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></div>` : ""}`;
}

// ------------------------------------------------------------------ download

function exportUrl(fmt) {
  return state.slug ? `/api/shared/${state.slug}/export?format=${fmt}` : `/api/research/${state.runId}/export?format=${fmt}`;
}

$("#downloadBtn").addEventListener("click", (e) => {
  e.stopPropagation();
  const menu = $("#downloadMenu");
  menu.hidden = !menu.hidden;
  $("#downloadBtn").setAttribute("aria-expanded", String(!menu.hidden));
});
document.addEventListener("click", () => { $("#downloadMenu").hidden = true; });
$("#downloadMenu").addEventListener("click", async (e) => {
  const fmt = e.target.dataset?.fmt;
  if (!fmt) return;
  toast(`Preparing ${fmt.toUpperCase()}…`);
  try {
    const res = await fetch(exportUrl(fmt), { headers: headers() });
    if (!res.ok) throw new Error((await res.json()).error || "Export failed");
    const blob = await res.blob();
    const name = (res.headers.get("content-disposition") || "").match(/filename="([^"]+)"/)?.[1] || `report.${fmt}`;
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: name });
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  } catch (err) {
    toast(err.message);
  }
});
$("#printBtn").addEventListener("click", () => window.print());
$("#copyMdBtn").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(state.report.markdown); toast("Markdown copied"); }
  catch { toast("Copy failed — use Download instead"); }
});

// ------------------------------------------------------------------ share

$("#shareBtn").addEventListener("click", async () => {
  try {
    if (!state.shareUrl) {
      const { url } = await api(`/api/research/${state.runId}/share`, { method: "POST" });
      state.shareUrl = url;
    }
    openShare();
  } catch (err) {
    toast(err.message);
  }
});

function openShare() {
  const url = state.shareUrl;
  const title = state.report.title;
  const text = `Research brief: ${title}`;
  const u = encodeURIComponent(url), t = encodeURIComponent(text);
  $("#shareUrl").value = url;
  $("#revokeShare").hidden = Boolean(state.slug);
  const targets = [
    ["X / Twitter", `https://twitter.com/intent/tweet?text=${t}&url=${u}`],
    ["LinkedIn", `https://www.linkedin.com/sharing/share-offsite/?url=${u}`],
    ["WhatsApp", `https://wa.me/?text=${t}%20${u}`],
    ["Telegram", `https://t.me/share/url?url=${u}&text=${t}`],
    ["Reddit", `https://www.reddit.com/submit?url=${u}&title=${t}`],
    ["Email", `mailto:?subject=${t}&body=${encodeURIComponent(`${text}\n\n${url}`)}`],
  ];
  $("#shareTargets").innerHTML =
    (navigator.share ? `<button type="button" id="nativeShare">Share via…</button>` : "") +
    targets.map(([name, href]) => `<a href="${href}" target="_blank" rel="noopener noreferrer">${name}</a>`).join("");
  $("#nativeShare")?.addEventListener("click", () => navigator.share({ title, text, url }).catch(() => {}));
  $("#shareDialog").showModal();
}

$("#copyShare").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText($("#shareUrl").value); toast("Link copied"); }
  catch { $("#shareUrl").select(); toast("Press ⌘/Ctrl+C to copy"); }
});
$("#revokeShare").addEventListener("click", async () => {
  try {
    await api(`/api/research/${state.runId}/share`, { method: "DELETE" });
    state.shareUrl = null;
    $("#shareDialog").close();
    toast("Share link revoked");
  } catch (err) { toast(err.message); }
});

// ------------------------------------------------------------------ history

$("#historyBtn").addEventListener("click", async () => {
  $("#history").hidden = false;
  try {
    const runs = await api("/api/runs");
    $("#historyList").innerHTML = runs.length ? runs.map((r) =>
      `<li><a href="/run/${r.id}">${esc(r.question)}<small>${new Date(r.created_at * 1000).toLocaleString()} · ${r.depth} · ${r.status}</small></a></li>`).join("")
      : `<li class="muted" style="padding:12px">No research yet.</li>`;
  } catch (err) { $("#historyList").innerHTML = `<li class="muted" style="padding:12px">${esc(err.message)}</li>`; }
});
$("#closeHistory").addEventListener("click", () => { $("#history").hidden = true; });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#history").hidden = true; });
window.addEventListener("popstate", () => location.reload());

boot();
