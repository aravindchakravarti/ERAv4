const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const captureBtn = document.getElementById("captureBtn");
const copyBtn = document.getElementById("copyBtn");
const latexOut = document.getElementById("latexOut");
const resultBox = document.getElementById("resultBox");
const statusEl = document.getElementById("status");
const ocrUrlInput = document.getElementById("ocrUrl");

let img = new Image();
let isDragging = false;
let startX = 0, startY = 0, endX = 0, endY = 0;
let screenshotDataUrl = null;

captureBtn.addEventListener("click", async () => {
  setStatus("Capturing visible tab...");
  resultBox.classList.add("hidden");
  latexOut.value = "";

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    chrome.tabs.captureVisibleTab(tab.windowId, { format: "png" }, (dataUrl) => {
      if (chrome.runtime.lastError || !dataUrl) {
        setStatus("Capture failed. " + (chrome.runtime.lastError?.message || ""));
        return;
      }
      screenshotDataUrl = dataUrl;
      loadImage(dataUrl);
    });
  } catch (e) {
    setStatus("Capture error: " + e.message);
  }
});

function loadImage(dataUrl) {
  img = new Image();
  img.onload = () => {
    // Use the real pixel size to preserve fidelity for OCR
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    setStatus("Draw a rectangle to select the math region.");
  };
  img.src = dataUrl;
}

// Mouse interactions for cropping
canvas.addEventListener("mousedown", (e) => {
  if (!img.src) return;
  const { x, y } = getRelativePos(e);
  startX = x; startY = y;
  endX = x; endY = y;
  isDragging = true;
});

canvas.addEventListener("mousemove", (e) => {
  if (!isDragging) return;
  const { x, y } = getRelativePos(e);
  endX = x; endY = y;
  redrawWithSelection();
});

canvas.addEventListener("mouseup", () => {
  if (!isDragging) return;
  isDragging = false;
  cropAndSend();
});

canvas.addEventListener("mouseleave", () => {
  if (isDragging) {
    isDragging = false;
    redrawWithSelection();
  }
});

function getRelativePos(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: Math.floor((e.clientX - rect.left) * scaleX),
    y: Math.floor((e.clientY - rect.top) * scaleY),
  };
}

function redrawWithSelection() {
  ctx.drawImage(img, 0, 0);
  const { x, y, w, h } = normalizedRect();
  // selection overlay
  ctx.save();
  ctx.fillStyle = "rgba(0,0,0,0.25)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.clearRect(x, y, w, h);
  // draw border
  ctx.strokeStyle = "white";
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, w, h);
  ctx.restore();
}

function normalizedRect() {
  const x = Math.min(startX, endX);
  const y = Math.min(startY, endY);
  const w = Math.max(1, Math.abs(endX - startX));
  const h = Math.max(1, Math.abs(endY - startY));
  return { x, y, w, h };
}

async function cropAndSend() {
  if (!img.src) return;
  const { x, y, w, h } = normalizedRect();

  // Guard: avoid zero-size crops
  if (w < 4 || h < 4) {
    setStatus("Selection too small. Try again.");
    ctx.drawImage(img, 0, 0);
    return;
  }

  // Crop to a new canvas at 1:1 pixels
  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = w;
  cropCanvas.height = h;
  const cropCtx = cropCanvas.getContext("2d");
  cropCtx.drawImage(img, x, y, w, h, 0, 0, w, h);
  const croppedDataUrl = cropCanvas.toDataURL("image/png");

  setStatus("Sending to OCR...");
  try {
    const url = ocrUrlInput.value.trim();
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: croppedDataUrl })
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status}: ${text}`);
    }
    const data = await res.json();
    const latex = (data.latex || "").toString();
    latexOut.value = latex.length ? latex : "(empty)";
    resultBox.classList.remove("hidden");
    setStatus("Done.");
  } catch (err) {
    setStatus("OCR error: " + err.message);
  } finally {
    // Reset the overlay, keep screenshot visible
    ctx.drawImage(img, 0, 0);
  }
}

copyBtn.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(latexOut.value);
    setStatus("Copied to clipboard.");
  } catch {
    setStatus("Copy failed. Select the text and copy manually.");
  }
});

function setStatus(msg) { statusEl.textContent = msg || ""; }
