const labelNames = [
  "Center",
  "Donut",
  "Edge_Loc",
  "Edge_Ring",
  "Loc",
  "Near_Full",
  "Scratch",
  "Random"
];

const state = {
  wafers: [],
  selectedIndex: null,
  lastPrediction: null,
  loadingWafers: false,
  loadingPrediction: false
};

const elements = {
  waferGrid: document.getElementById("waferGrid"),
  resampleButton: document.getElementById("resampleButton"),
  predictButton: document.getElementById("predictButton"),
  statusText: document.getElementById("statusText"),
  labelsTitle: document.getElementById("labelsTitle"),
  labelBadge: document.getElementById("labelBadge"),
  selectedMeta: document.getElementById("selectedMeta"),
  binaryChart: document.getElementById("binaryChart"),
  probabilityChart: document.getElementById("probabilityChart")
};

elements.resampleButton.addEventListener("click", loadWafers);
elements.predictButton.addEventListener("click", runInference);

loadWafers();

async function loadWafers() {
  setStatus("Loading wafer samples...", true);
  setSelectedWafer(null);

  try {
    const response = await fetch("/api/wafers?count=10");
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || "Unable to load wafers");
    }

    state.wafers = payload.wafers || [];
    state.lastPrediction = null;
    renderWaferGrid();
    resetResults();
    setStatus(`Loaded ${state.wafers.length} wafer samples`, false);
  } catch (error) {
    console.error(error);
    state.wafers = [];
    renderWaferGrid();
    resetResults(error.message);
    setStatus(error.message, false, true);
  }
}

async function runInference() {
  const selectedWafer = getSelectedWafer();
  if (!selectedWafer) {
    return;
  }

  state.loadingPrediction = true;
  syncButtons();
  elements.labelsTitle.textContent = "Running inference...";
  elements.labelBadge.textContent = "Evaluating wafer sample";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ wafer: selectedWafer })
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || "Inference failed");
    }

    state.lastPrediction = payload;
    renderPrediction(payload);
  } catch (error) {
    console.error(error);
    resetResults(error.message);
  } finally {
    state.loadingPrediction = false;
    syncButtons();
  }
}

function renderWaferGrid() {
  elements.waferGrid.innerHTML = "";

  if (!state.wafers.length) {
    elements.waferGrid.innerHTML = `<p class="status">No wafer samples available.</p>`;
    return;
  }

  state.wafers.forEach((wafer, index) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "wafer-card";
    card.innerHTML = `
      <canvas width="96" height="96" aria-hidden="true"></canvas>
      <div class="wafer-card__meta">
        <h3>Wafer ${index + 1}</h3>
        <p>Click to select this sample for inference.</p>
      </div>
    `;
    card.addEventListener("click", () => setSelectedWafer(index));

    drawWaferPreview(card.querySelector("canvas"), wafer);
    elements.waferGrid.appendChild(card);
  });

  highlightSelectedWafer();
}

function setSelectedWafer(index) {
  state.selectedIndex = index;
  state.lastPrediction = null;
  highlightSelectedWafer();
  syncButtons();
  resetResults();

  if (index === null) {
    elements.selectedMeta.textContent = "No wafer selected";
    return;
  }

  elements.selectedMeta.textContent = `Selected wafer ${index + 1}`;
}

function highlightSelectedWafer() {
  const cards = [...elements.waferGrid.querySelectorAll(".wafer-card")];
  cards.forEach((card, index) => {
    card.classList.toggle("is-selected", index === state.selectedIndex);
  });
}

function renderPrediction(payload) {
  const labels = payload.labels && payload.labels.length ? payload.labels : ["None"];
  const probabilities = Array.isArray(payload.probabilities) ? payload.probabilities : [];
  const binaryOutput = labelNames.map((label) => ({
    label,
    value: payload.labels && payload.labels.includes(label) ? 1 : 0
  }));

  elements.labelsTitle.textContent = "Prediction complete";
  elements.labelBadge.textContent = labels.join(", ");
  elements.binaryChart.innerHTML = binaryOutput.map(renderMetricRow).join("");
  elements.probabilityChart.innerHTML = labelNames
    .map((label, index) => ({
      label,
      value: Number(probabilities[index] || 0)
    }))
    .map(renderMetricRow)
    .join("");
}

function resetResults(message = "No prediction yet") {
  elements.labelsTitle.textContent =
    state.selectedIndex === null ? "Choose a wafer to begin" : "Ready for inference";
  elements.labelBadge.textContent = message;
  elements.binaryChart.innerHTML = labelNames
    .map((label) => ({ label, value: 0 }))
    .map(renderMetricRow)
    .join("");
  elements.probabilityChart.innerHTML = labelNames
    .map((label) => ({ label, value: 0 }))
    .map(renderMetricRow)
    .join("");
}

function renderMetricRow(metric) {
  const percentage = Math.max(0, Math.min(100, Math.round(Number(metric.value) * 100)));

  return `
    <div class="metric-row">
      <div class="metric-row__label">${metric.label}</div>
      <div class="metric-row__track">
        <span class="metric-row__fill" style="width:${percentage}%"></span>
      </div>
      <div class="metric-row__value">${Number(metric.value).toFixed(3)}</div>
    </div>
  `;
}

function drawWaferPreview(canvas, wafer) {
  const context = canvas.getContext("2d");
  const rows = wafer.length;
  const cols = wafer[0].length;
  const cellWidth = canvas.width / cols;
  const cellHeight = canvas.height / rows;

  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      const value = Number(wafer[y][x] || 0);
      context.fillStyle = plasmaColor(value);
      context.fillRect(x * cellWidth, y * cellHeight, cellWidth, cellHeight);
    }
  }
}

function plasmaColor(value) {
  const clamped = Math.max(0, Math.min(1, value));
  const stops = [
    [13, 8, 135],
    [92, 0, 165],
    [176, 42, 143],
    [245, 136, 71],
    [240, 249, 33]
  ];
  const scaled = clamped * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(scaled));
  const ratio = scaled - index;
  const [r1, g1, b1] = stops[index];
  const [r2, g2, b2] = stops[index + 1];
  const r = Math.round(r1 + (r2 - r1) * ratio);
  const g = Math.round(g1 + (g2 - g1) * ratio);
  const b = Math.round(b1 + (b2 - b1) * ratio);
  return `rgb(${r}, ${g}, ${b})`;
}

function getSelectedWafer() {
  if (state.selectedIndex === null) {
    return null;
  }
  return state.wafers[state.selectedIndex] || null;
}

function setStatus(message, loading, isError = false) {
  state.loadingWafers = loading;
  elements.statusText.textContent = message;
  elements.statusText.style.color = isError ? "var(--danger)" : "var(--muted)";
  syncButtons();
}

function syncButtons() {
  const hasSelection = state.selectedIndex !== null;
  elements.predictButton.disabled = !hasSelection || state.loadingPrediction || state.loadingWafers;
  elements.predictButton.textContent = state.loadingPrediction ? "Running..." : "Run Inference";
  elements.resampleButton.disabled = state.loadingWafers;
}
