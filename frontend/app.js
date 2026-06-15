const API_BASE = "http://localhost:8000";

const inputs = {
  sleep: { el: document.getElementById("sleep"), display: document.getElementById("val-sleep"), format: v => parseFloat(v).toFixed(1) },
  water: { el: document.getElementById("water"), display: document.getElementById("val-water"), format: v => parseFloat(v).toFixed(1) },
  sodium: { el: document.getElementById("sodium"), display: document.getElementById("val-sodium"), format: v => v },
  standing: { el: document.getElementById("standing"), display: document.getElementById("val-standing"), format: v => parseFloat(v).toFixed(1) },
  stress: { el: document.getElementById("stress"), display: document.getElementById("val-stress"), format: v => v },
  symptom: { el: document.getElementById("symptom"), display: document.getElementById("val-symptom"), format: v => parseFloat(v).toFixed(1) },
  pressure: { el: document.getElementById("pressure"), display: document.getElementById("val-pressure"), format: v => v },
  temp: { el: document.getElementById("temp"), display: document.getElementById("val-temp"), format: v => v },
};

const gaugeFill = document.getElementById("gauge-fill");
const gaugeBg = document.getElementById("gauge-bg");
const riskPercent = document.getElementById("risk-percent");
const riskLevel = document.getElementById("risk-level");
const riskFactors = document.getElementById("risk-factors");
const factorsList = document.getElementById("factors-list");
const modelAcc = document.getElementById("model-acc");
const predictBtn = document.getElementById("predict-btn");
const btnText = predictBtn.querySelector(".btn-text");
const spinner = predictBtn.querySelector(".spinner");
const errorBanner = document.getElementById("error-banner");
const closeError = document.getElementById("close-error");
const historyList = document.getElementById("history-list");
const clearHistoryBtn = document.getElementById("clear-history");
const whatifList = document.getElementById("whatif-list");
const whatifLoading = document.getElementById("whatif-loading");

const MAX_DASH = 251.3; // length of the semicircle arc

// Scenario presets for quick demos
const SCENARIOS = {
  good: { sleep: 10, water: 4.0, sodium: 5000, standing: 1, stress: 1, pressure: 1013, temp: 20, meds: true, symptom: 1 },
  average: { sleep: 7, water: 2.0, sodium: 3000, standing: 4, stress: 5, pressure: 1013, temp: 22, meds: true, symptom: 4 },
  bad: { sleep: 4, water: 0.5, sodium: 1000, standing: 10, stress: 10, pressure: 1030, temp: 38, meds: false, symptom: 9 },
};

Object.values(inputs).forEach(({ el, display, format }) => {
  el.addEventListener("input", () => { display.textContent = format(el.value); });
});

function setLoading(isLoading) {
  predictBtn.disabled = isLoading;
  btnText.classList.toggle("hidden", isLoading);
  spinner.classList.toggle("hidden", !isLoading);
}

function updateGauge(prob, level) {
  const offset = MAX_DASH * (1 - prob);
  gaugeFill.style.strokeDashoffset = offset;
  gaugeBg.style.strokeDashoffset = MAX_DASH * (1 - Math.min(prob * 0.5, 0.15)); // subtle background fill

  let color = "#10b981";
  if (prob >= 0.3) color = "#f59e0b";
  if (prob >= 0.6) color = "#f43f5e";

  riskPercent.textContent = `${Math.round(prob * 100)}%`;
  riskPercent.style.color = color;
  riskLevel.textContent = `${level} risk`;
}

function prettyName(name) {
  const labels = {
    sleep_hours: "Sleep",
    water_intake: "Water intake",
    sodium_mg: "Sodium",
    standing_hours: "Standing / active time",
    stress_level: "Stress level",
    weather_pressure: "Barometric pressure",
    temperature: "Temperature",
    meds_taken: "Medication taken",
    symptom_score: "Current symptom score",
  };
  return labels[name] || name.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function riskClass(level) {
  return level === "Low" ? "risk-low" : level === "Moderate" ? "risk-moderate" : "risk-high";
}

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem("flareforecast_history") || "[]");
  } catch {
    return [];
  }
}

function saveHistory(history) {
  localStorage.setItem("flareforecast_history", JSON.stringify(history.slice(0, 20)));
  renderHistory(history);
}

function renderHistory(history) {
  historyList.innerHTML = "";
  if (history.length === 0) {
    historyList.innerHTML = "<li>No predictions yet.</li>";
    return;
  }

  history.slice(0, 10).forEach(entry => {
    const li = document.createElement("li");
    const date = new Date(entry.timestamp).toLocaleString();
    li.innerHTML = `
      <div style="display:flex;align-items:center;">
        <span class="risk-dot ${riskClass(entry.risk_level)}"></span>
        <span><strong>${Math.round(entry.probability * 100)}%</strong> ${entry.risk_level} risk</span>
      </div>
      <span class="time">${date}</span>
    `;
    historyList.appendChild(li);
  });
}

function addToHistory(data) {
  const history = loadHistory();
  history.unshift({
    timestamp: Date.now(),
    probability: data.flare_probability,
    risk_level: data.risk_level,
  });
  saveHistory(history);
}

async function runWhatIf() {
  whatifList.innerHTML = "";
  whatifLoading.classList.remove("hidden");

  try {
    const res = await fetch(`${API_BASE}/whatif`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });

    if (!res.ok) throw new Error("whatif failed");

    const results = await res.json();
    whatifLoading.classList.add("hidden");

    // Only show factors that would reduce risk (positive delta = risk goes down)
    const helpful = results.filter(r => r.delta > 0).slice(0, 5);

    if (helpful.length === 0) {
      whatifList.innerHTML = "<li>No single change would lower your risk much. You're doing great!</li>";
      return;
    }

    const maxDelta = Math.max(...helpful.map(r => r.delta));

    helpful.forEach((r, i) => {
      const li = document.createElement("li");
      li.style.animationDelay = `${i * 0.05}s`;

      const unit = r.factor === "sodium_mg" ? "mg" : r.factor === "water_intake" ? "L" : r.factor === "sleep_hours" || r.factor === "standing_hours" ? "hrs" : "";
      const direction = r.fixed_value > r.current_value ? "↑" : "↓";
      const percentDrop = Math.round(r.delta * 100);
      const barWidth = `${(r.delta / maxDelta) * 100}%`;

      li.innerHTML = `
        <div class="factor">
          <span class="factor-name">${prettyName(r.factor)}</span>
          <span class="factor-change">${direction} to ${r.fixed_value}${unit}</span>
        </div>
        <div style="display:flex;align-items:center;">
          <span class="delta">-${percentDrop}% <small>risk</small></span>
          <div class="whatif-bar"><span style="width:${barWidth}"></span></div>
        </div>
      `;
      whatifList.appendChild(li);
    });
  } catch (err) {
    console.error("What-if error", err);
    whatifLoading.classList.add("hidden");
  }
}

function buildPayload() {
  return {
    sleep_hours: parseFloat(inputs.sleep.el.value),
    water_intake: parseFloat(inputs.water.el.value),
    sodium_mg: parseInt(inputs.sodium.el.value),
    standing_hours: parseFloat(inputs.standing.el.value),
    stress_level: parseInt(inputs.stress.el.value),
    weather_pressure: parseFloat(inputs.pressure.el.value),
    temperature: parseFloat(inputs.temp.el.value),
    meds_taken: document.getElementById("meds").checked ? 1 : 0,
    symptom_score: parseFloat(inputs.symptom.el.value),
  };
}

async function predict() {
  setLoading(true);
  errorBanner.classList.add("hidden");
  riskLevel.textContent = "Analyzing...";

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const data = await res.json();
    console.log("Prediction response:", data);
    updateGauge(data.flare_probability, data.risk_level);
    addToHistory(data);

    factorsList.innerHTML = "";
    if (data.top_factors.length === 0) {
      factorsList.innerHTML = "<li>No major risk drivers today. Great job!</li>";
    } else {
      data.top_factors.forEach(f => {
        const li = document.createElement("li");
        const avgText = f.direction === "above" ? "above your average" : "below your average";
        const sign = f.delta > 0 ? "+" : "";
        li.textContent = `${prettyName(f.factor)} is ${avgText} (${sign}${f.delta}σ) — raises risk`;
        factorsList.appendChild(li);
      });
    }
    modelAcc.textContent = `${Math.round(data.model_accuracy * 100)}%`;
    riskFactors.classList.remove("hidden");
    runWhatIf();
  } catch (err) {
    console.error(err);
    riskLevel.textContent = "Backend unavailable";
    errorBanner.classList.remove("hidden");
  } finally {
    setLoading(false);
  }
}

// Scenario buttons
document.querySelectorAll(".scenario-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const s = SCENARIOS[btn.dataset.scenario];
    if (!s) return;

    inputs.sleep.el.value = s.sleep;
    inputs.water.el.value = s.water;
    inputs.sodium.el.value = s.sodium;
    inputs.standing.el.value = s.standing;
    inputs.stress.el.value = s.stress;
    inputs.pressure.el.value = s.pressure;
    inputs.temp.el.value = s.temp;
    inputs.symptom.el.value = s.symptom;
    document.getElementById("meds").checked = s.meds;

    Object.values(inputs).forEach(({ el, display, format }) => {
      display.textContent = format(el.value);
    });

    predict();
  });
});

// Form submit
document.getElementById("snapshot-form").addEventListener("submit", (e) => {
  e.preventDefault();
  predict();
});

closeError.addEventListener("click", () => errorBanner.classList.add("hidden"));

clearHistoryBtn.addEventListener("click", () => {
  localStorage.removeItem("flareforecast_history");
  renderHistory([]);
});

renderHistory(loadHistory());

// --- Timeline chart ---
async function drawTimeline() {
  try {
    const res = await fetch(`${API_BASE}/sample-data?n_days=30`);
    if (!res.ok) throw new Error("sample data failed");
    const data = await res.json();
    const canvas = document.getElementById("timeline");
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width, h = rect.height;
    const pad = { top: 20, right: 20, bottom: 30, left: 40 };
    const chartW = w - pad.left - pad.right;
    const chartH = h - pad.top - pad.bottom;

    ctx.clearRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = pad.top + chartH * (i / 5);
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
    }

    // Y-axis labels
    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let i = 0; i <= 5; i++) {
      const val = 10 - (i / 5) * 10;
      const y = pad.top + chartH * (i / 5);
      ctx.fillText(val.toFixed(0), pad.left - 8, y);
    }

    // X-axis labels (every 5 days)
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    data.forEach((d, i) => {
      if (i % 5 === 0 || i === data.length - 1) {
        const x = pad.left + (i / (data.length - 1)) * chartW;
        const label = d.date.slice(5); // MM-DD
        ctx.fillText(label, x, h - pad.bottom + 6);
      }
    });

    // Symptom score line
    ctx.strokeStyle = "#818cf8";
    ctx.lineWidth = 3;
    ctx.lineJoin = "round";
    ctx.beginPath();
    data.forEach((d, i) => {
      const x = pad.left + (i / (data.length - 1)) * chartW;
      const y = pad.top + chartH * (1 - d.symptom_score / 10);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Fill area
    ctx.fillStyle = "rgba(99, 102, 241, 0.15)";
    ctx.beginPath();
    data.forEach((d, i) => {
      const x = pad.left + (i / (data.length - 1)) * chartW;
      const y = pad.top + chartH * (1 - d.symptom_score / 10);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.lineTo(pad.left + chartW, pad.top + chartH);
    ctx.lineTo(pad.left, pad.top + chartH);
    ctx.closePath();
    ctx.fill();

    // Flare markers
    data.forEach((d, i) => {
      if (d.flare_next_48h) {
        const x = pad.left + (i / (data.length - 1)) * chartW;
        const y = pad.top + chartH * (1 - d.symptom_score / 10);
        ctx.fillStyle = "#f43f5e";
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  } catch (err) {
    console.error("Timeline error", err);
  }
}

window.addEventListener("resize", drawTimeline);

// Load initial state
drawTimeline();
predict();
