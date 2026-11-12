const state = {
  data: null,
  view: "overview",
  seed: 52,
  episode: 0,
  step: 0,
  timer: null,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const fmt = (value, digits = 2) => Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
const signed = (value, digits = 2) => `${value > 0 ? "+" : ""}${fmt(value, digits)}`;
const titleCase = (value) => String(value).replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);

const ZONES = {
  blue_agent_0: "Restricted Zone A",
  blue_agent_1: "Operational Zone A",
  blue_agent_2: "Restricted Zone B",
  blue_agent_3: "Operational Zone B",
  blue_agent_4: "Admin Network",
};

function setView(view, updateUrl = true) {
  if (!$("#view-" + view)) return;
  state.view = view;
  $$("[data-view-panel]").forEach((panel) => panel.classList.toggle("is-visible", panel.dataset.viewPanel === view));
  $$(".nav-item").forEach((button) => button.classList.toggle("is-active", button.dataset.view === view));
  if (updateUrl) {
    const url = new URL(window.location.href);
    url.searchParams.set("view", view);
    history.replaceState({}, "", url);
  }
  window.scrollTo({ top: 0, behavior: "instant" });
}

function renderOverview() {
  const s = state.data.summary;
  $("#claimBoundary").textContent = state.data.meta.claimBoundary;
  const cards = [
    ["Held-out reward", fmt(s.rewardMean, 1), `± ${fmt(s.rewardStd, 1)} SD`],
    ["Paired evaluation", s.pairedRuns, "runs per comparison"],
    ["Executed baselines", s.baselinesCompared, "all positive CIs"],
    ["Policy validation", `${fmt(s.policyAccuracy * 100, 1)}%`, `macro-F1 ${fmt(s.policyMacroF1, 3)}`],
    ["Evidence coverage", `${fmt(s.explanationCoverage * 100, 0)}%`, "prototype explanations"],
  ];
  $("#overviewMetrics").innerHTML = cards
    .map(([label, value, note]) => `<article class="metric-card"><span>${label}</span><strong>${value}</strong><small>${note}</small></article>`)
    .join("");
}

function currentEpisode() {
  return state.data.simulation.find((item) => item.seed === Number(state.seed) && item.episode === Number(state.episode));
}

function configureSimulationSelectors() {
  const seeds = [...new Set(state.data.simulation.map((item) => item.seed))];
  $("#seedSelect").innerHTML = seeds.map((seed) => `<option value="${seed}">${seed}</option>`).join("");
  const initial = new URL(window.location.href);
  const requestedSeed = Number(initial.searchParams.get("seed"));
  if (seeds.includes(requestedSeed)) state.seed = requestedSeed;
  $("#seedSelect").value = state.seed;
  configureEpisodes();
  const requestedEpisode = Number(initial.searchParams.get("episode"));
  const validEpisodes = state.data.simulation.filter((item) => item.seed === state.seed).map((item) => item.episode);
  if (validEpisodes.includes(requestedEpisode)) state.episode = requestedEpisode;
  $("#episodeSelect").value = state.episode;
  const requestedStep = Number(initial.searchParams.get("step"));
  if (Number.isInteger(requestedStep) && requestedStep >= 0) state.step = Math.min(requestedStep, currentEpisode().trace.length - 1);
  renderSimulation();
}

function configureEpisodes() {
  const episodes = state.data.simulation.filter((item) => item.seed === Number(state.seed)).map((item) => item.episode);
  if (!episodes.includes(Number(state.episode))) state.episode = episodes[0];
  $("#episodeSelect").innerHTML = episodes.map((episode) => `<option value="${episode}">${episode}</option>`).join("");
  $("#episodeSelect").value = state.episode;
}

function renderSimulation() {
  const episode = currentEpisode();
  if (!episode) return;
  state.step = Math.max(0, Math.min(state.step, episode.trace.length - 1));
  const frame = episode.trace[state.step];
  $("#stepSlider").max = episode.trace.length - 1;
  $("#stepSlider").value = state.step;
  $("#stepNumber").textContent = String(state.step).padStart(2, "0");
  $("#stepTotal").textContent = `/ ${episode.trace.length - 1}`;
  $("#episodeReward").textContent = signed(episode.totalReward, 1);
  $("#stepReward").textContent = signed(frame.sumReward, 2);
  $("#episodeRuntime").textContent = `${fmt(episode.wallTimeSeconds, 2)} s`;
  $("#profileName").textContent = episode.profile;
  renderTopology(frame);
  renderActionBars(episode.actionCounts);
  renderRewardChart(episode.trace, state.step);
  renderActionTable(frame.actions);
  const url = new URL(window.location.href);
  url.searchParams.set("seed", state.seed);
  url.searchParams.set("episode", state.episode);
  url.searchParams.set("step", state.step);
  history.replaceState({}, "", url);
}

function renderTopology(frame) {
  const node = $("#topology");
  node.innerHTML = "";
  Object.entries(ZONES).forEach(([agent, zone]) => {
    const command = frame.actions[agent] || "No action";
    const kind = frame.actionKinds[agent] || "Unknown";
    const card = document.createElement("article");
    card.className = `zone-card is-active is-${kind.toLowerCase()}`;
    card.innerHTML = `<span class="agent">${escapeHtml(agent.replace("blue_agent_", "Defender "))}</span><strong>${escapeHtml(zone)}</strong><span class="zone-action">Action class<b>${escapeHtml(kind)}</b></span>`;
    card.title = command;
    node.appendChild(card);
  });
}

function renderActionBars(counts) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...entries.map(([, value]) => value));
  $("#actionBars").innerHTML = entries
    .map(([name, value]) => `<div class="bar-row"><span>${escapeHtml(name)}</span><div class="bar-track"><div class="bar-fill" style="width:${(value / max) * 100}%"></div></div><strong>${value}</strong></div>`)
    .join("");
}

function renderActionTable(actions) {
  $("#actionTable").innerHTML = Object.entries(actions)
    .map(([agent, command]) => {
      const kind = command.split(" ", 1)[0];
      return `<tr><td>${escapeHtml(agent)}</td><td>${escapeHtml(ZONES[agent])}</td><td><span class="action-class">${escapeHtml(kind)}</span></td><td><code>${escapeHtml(command)}</code></td></tr>`;
    })
    .join("");
}

function svgText(x, y, text, anchor = "start", fill = "#9eb2bc", size = 10) {
  return `<text x="${x}" y="${y}" text-anchor="${anchor}" fill="${fill}" font-size="${size}" font-family="Inter, sans-serif">${escapeHtml(text)}</text>`;
}

function renderRewardChart(trace, cursor) {
  const svg = $("#rewardChart");
  const values = trace.map((item) => item.sumReward);
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 0);
  const W = 1200, H = 230, L = 52, R = 18, T = 18, B = 34;
  const x = (i) => L + (i / Math.max(1, trace.length - 1)) * (W - L - R);
  const y = (v) => T + ((max - v) / Math.max(1, max - min)) * (H - T - B);
  const path = trace.map((item, index) => `${index ? "L" : "M"}${x(index).toFixed(2)},${y(item.sumReward).toFixed(2)}`).join(" ");
  const grid = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
    const value = max - (max - min) * ratio;
    const yy = y(value);
    return `<line x1="${L}" y1="${yy}" x2="${W - R}" y2="${yy}" stroke="#284452" stroke-width="1"/>${svgText(L - 8, yy + 3, fmt(value, 1), "end")}`;
  }).join("");
  const cx = x(cursor), cy = y(values[cursor]);
  svg.innerHTML = `${grid}<line x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}" stroke="#4b6571"/><path d="${path}" fill="none" stroke="#5bc6d7" stroke-width="2"/><line x1="${cx}" y1="${T}" x2="${cx}" y2="${H - B}" stroke="#e3b866" stroke-dasharray="4 4"/><circle cx="${cx}" cy="${cy}" r="5" fill="#e3b866"/>${svgText(L, H - 12, "0")}${svgText(W - R, H - 12, String(trace.length - 1), "end")}${svgText(cx + 9, Math.max(15, cy - 8), `step ${cursor}: ${signed(values[cursor], 1)}`, "start", "#e3b866", 11)}`;
}

function setupSimulationEvents() {
  $("#seedSelect").addEventListener("change", (event) => { stopPlayback(); state.seed = Number(event.target.value); state.episode = 0; state.step = 0; configureEpisodes(); renderSimulation(); });
  $("#episodeSelect").addEventListener("change", (event) => { stopPlayback(); state.episode = Number(event.target.value); state.step = 0; renderSimulation(); });
  $("#stepSlider").addEventListener("input", (event) => { stopPlayback(); state.step = Number(event.target.value); renderSimulation(); });
  $("#stepButton").addEventListener("click", () => { stopPlayback(); state.step = Math.min(state.step + 1, currentEpisode().trace.length - 1); renderSimulation(); });
  $("#resetButton").addEventListener("click", () => { stopPlayback(); state.step = 0; renderSimulation(); });
  $("#playButton").addEventListener("click", () => state.timer ? stopPlayback() : startPlayback());
}

function startPlayback() {
  if (state.step >= currentEpisode().trace.length - 1) state.step = 0;
  $("#playButton").textContent = "Pause trace";
  state.timer = setInterval(() => {
    if (state.step >= currentEpisode().trace.length - 1) return stopPlayback();
    state.step += 1;
    renderSimulation();
  }, Number($("#speedSelect").value));
}

function stopPlayback() {
  clearInterval(state.timer);
  state.timer = null;
  $("#playButton").textContent = "Play trace";
}

function configureExplainability() {
  const episodes = [...new Set(state.data.explanations.map((item) => item.episode_id))];
  $("#explainEpisode").innerHTML = episodes.map((episode) => `<option value="${episode}">${episode}</option>`).join("");
  $("#explainEpisode").addEventListener("change", () => { $("#explainStep").value = 0; renderExplanation(); });
  $("#explainStep").addEventListener("input", renderExplanation);
  renderExplanation();
}

function renderExplanation() {
  const episode = Number($("#explainEpisode").value || 0);
  const records = state.data.explanations.filter((item) => item.episode_id === episode);
  $("#explainStep").max = Math.max(0, records.length - 1);
  const index = Math.min(Number($("#explainStep").value), records.length - 1);
  const record = records[index] || state.data.explanations[0];
  $("#explainStepOutput").textContent = String(record.step).padStart(2, "0");
  $("#explainAction").textContent = titleCase(record.action);
  $("#explainRegime").textContent = titleCase(record.regime);
  $("#explainText").textContent = record.explanation;
  $("#evidenceCount").textContent = `${record.evidence_count} item${record.evidence_count === 1 ? "" : "s"}`;
  $("#evidenceCards").innerHTML = record.evidence.length
    ? record.evidence.map((item) => `<article class="evidence-card"><span>${escapeHtml(titleCase(item.kill_chain_phases))}</span><strong>${escapeHtml(item.name)}</strong></article>`).join("")
    : `<article class="evidence-card"><span>No retained evidence</span><strong>This record is included in the uncovered 9%.</strong></article>`;
}

function renderEvidence() {
  const rows = [{ policy: "CybSAGE", reward_mean: state.data.summary.rewardMean }, ...state.data.baselines]
    .sort((a, b) => Number(b.reward_mean) - Number(a.reward_mean));
  const worst = Math.min(...rows.map((row) => Number(row.reward_mean)));
  const best = Math.max(...rows.map((row) => Number(row.reward_mean)));
  $("#benchmarkChart").innerHTML = rows.map((row) => {
    const score = ((Number(row.reward_mean) - worst) / (best - worst)) * 92 + 8;
    const isCyb = row.policy === "CybSAGE";
    return `<div class="benchmark-row ${isCyb ? "is-cybsage" : ""}"><span class="benchmark-label">${escapeHtml(titleCase(row.policy))}</span><div class="benchmark-track"><div class="benchmark-fill" style="width:${score}%"></div></div><strong>${fmt(row.reward_mean, 1)}</strong></div>`;
  }).join("");
  renderEffectChart();
  const learned = state.data.ablations.filter((row) => ["cybsage_full", "no_posterior", "no_ssl", "state_only"].includes(row.method));
  const min = Math.min(...learned.map((row) => row.decision_utility_mean)) - 0.01;
  const max = Math.max(...learned.map((row) => row.decision_utility_mean)) + 0.002;
  $("#ablationChart").innerHTML = learned.map((row) => {
    const height = ((row.decision_utility_mean - min) / (max - min)) * 100;
    return `<div class="ablation-column"><span class="ablation-value">${fmt(row.decision_utility_mean, 3)}</span><div class="ablation-bar" style="height:${height}%"></div><span class="ablation-label">${escapeHtml(titleCase(row.method))}</span></div>`;
  }).join("");
  const full = learned.find((row) => row.method === "cybsage_full");
  $("#fullUtility").textContent = fmt(full.decision_utility_mean, 3);
  $("#fullMacroF1").textContent = fmt(full.macro_f1_mean, 3);
  $("#evidenceClaim").textContent = state.data.meta.claimBoundary;
}

function renderEffectChart() {
  const rows = [...state.data.effects].sort((a, b) => b.mean_difference - a.mean_difference);
  const svg = $("#effectChart");
  const W = 760, H = 390, L = 124, R = 42, T = 28, B = 40;
  const max = Math.ceil(Math.max(...rows.map((row) => row.ci95_high)) / 100) * 100;
  const x = (v) => L + (v / max) * (W - L - R);
  const rowHeight = (H - T - B) / rows.length;
  let markup = "";
  for (let tick = 0; tick <= max; tick += 100) {
    const xx = x(tick);
    markup += `<line x1="${xx}" y1="${T}" x2="${xx}" y2="${H - B}" stroke="#284452"/>${svgText(xx, H - 15, tick, "middle")}`;
  }
  rows.forEach((row, index) => {
    const yy = T + rowHeight * index + rowHeight / 2;
    markup += `${svgText(L - 12, yy + 4, titleCase(row.baseline), "end", "#9eb2bc", 10)}<line x1="${x(row.ci95_low)}" y1="${yy}" x2="${x(row.ci95_high)}" y2="${yy}" stroke="#5bc6d7" stroke-width="3"/><line x1="${x(row.ci95_low)}" y1="${yy - 5}" x2="${x(row.ci95_low)}" y2="${yy + 5}" stroke="#5bc6d7"/><line x1="${x(row.ci95_high)}" y1="${yy - 5}" x2="${x(row.ci95_high)}" y2="${yy + 5}" stroke="#5bc6d7"/><circle cx="${x(row.mean_difference)}" cy="${yy}" r="5" fill="#e3b866"/>${svgText(Math.min(W - 8, x(row.ci95_high) + 8), yy + 4, `+${fmt(row.mean_difference, 1)}`, "start", "#e3b866", 10)}`;
  });
  svg.innerHTML = markup;
}

function bytes(value) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${fmt(value / 1024, 1)} KiB`;
  return `${fmt(value / 1024 ** 2, 1)} MiB`;
}

function renderResources() {
  const resources = state.data.resources;
  const runtimeRows = [
    { policy: "CybSAGE", meanSeconds: resources.cybsageEpisodeRuntime.meanSeconds },
    ...resources.baselineRuntime,
  ].sort((a, b) => b.meanSeconds - a.meanSeconds);
  const max = Math.max(...runtimeRows.map((row) => row.meanSeconds));
  $("#runtimeChart").innerHTML = runtimeRows.map((row) => `<div class="runtime-row"><span>${escapeHtml(titleCase(row.policy))}</span><div class="runtime-track"><div class="runtime-fill" style="width:${(row.meanSeconds / max) * 100}%"></div></div><strong>${fmt(row.meanSeconds, 2)}</strong></div>`).join("");
  $("#artifactList").innerHTML = resources.artifacts.map((item) => `<div class="artifact-row"><span>${escapeHtml(item.name)}</span><strong>${bytes(item.bytes)}</strong></div>`).join("");
  $("#energyStatement").textContent = resources.energyStatement;
  $("#provenanceList").innerHTML = state.data.provenance.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
}

function setupNavigation() {
  $$(".nav-item").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
  $$('[data-go]').forEach((button) => button.addEventListener("click", () => setView(button.dataset.go)));
  $("#exportView").addEventListener("click", () => window.print());
  const requested = new URL(window.location.href).searchParams.get("view");
  setView(requested || "overview", false);
}

async function init() {
  try {
    const response = await fetch("data/research-console.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`Artifact request failed: ${response.status}`);
    state.data = await response.json();
    renderOverview();
    configureSimulationSelectors();
    setupSimulationEvents();
    configureExplainability();
    renderEvidence();
    renderResources();
    setupNavigation();
    $("#loadingState").classList.add("is-hidden");
  } catch (error) {
    $("#loadingState").innerHTML = `<p>Unable to load the research artifact bundle.</p><code>${escapeHtml(error.message)}</code>`;
  }
}

document.addEventListener("DOMContentLoaded", init);
