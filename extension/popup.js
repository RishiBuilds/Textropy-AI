const DEFAULT_API_BASE = "http://localhost:8000";
let API_BASE = DEFAULT_API_BASE;
let API_KEY = "";

chrome.storage.local.get({ apiBase: DEFAULT_API_BASE, apiKey: "" }, (items) => {
  API_BASE = items.apiBase;
  API_KEY = items.apiKey;
  checkAPI();
});

function authHeaders() {
  return API_KEY ? { "X-API-Key": API_KEY } : {};
}

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const captureBtn = document.getElementById("captureBtn");
const btnIcon = document.getElementById("btnIcon");
const btnText = document.getElementById("btnText");
const subjectSelect = document.getElementById("subjectSelect");
const resultArea = document.getElementById("resultArea");
const resultText = document.getElementById("resultText");
const copyBtn = document.getElementById("copyBtn");
const errorMsg = document.getElementById("errorMsg");
const openDashboard = document.getElementById("openDashboard");

openDashboard.addEventListener("click", () => {
  chrome.tabs.create({ url: "http://localhost:8501" });
});

document.getElementById("openSettings").addEventListener("click", () => {
  chrome.runtime.openOptionsPage();
});

async function checkAPI() {
  try {
    const res = await fetch(`${API_BASE}/health`, { method: "GET", headers: authHeaders() });
    if (res.ok) {
      statusDot.classList.add("connected");
      statusText.textContent = "API Connected";
      captureBtn.disabled = false;
      return true;
    }
  } catch (e) {}

  statusDot.classList.remove("connected");
  statusText.textContent = "API Offline \u2014 Start the backend first";
  captureBtn.disabled = true;
  return false;
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.classList.add("visible");
}

function hideError() {
  errorMsg.classList.remove("visible");
}

function setLoading(loading) {
  if (loading) {
    btnIcon.innerHTML = '<span class="spinner"></span>';
    btnText.textContent = "Extracting...";
    captureBtn.disabled = true;
  } else {
    btnIcon.innerHTML = "&#128247;";
    btnText.textContent = "Capture & Extract";
    captureBtn.disabled = false;
  }
}

function dataURLtoBlob(dataUrl) {
  const parts = dataUrl.split(",");
  const mime = parts[0].match(/:(.*?);/)[1];
  const b64 = atob(parts[1]);
  const arr = new Uint8Array(b64.length);
  for (let i = 0; i < b64.length; i++) {
    arr[i] = b64.charCodeAt(i);
  }
  return new Blob([arr], { type: mime });
}

captureBtn.addEventListener("click", async () => {
  hideError();
  setLoading(true);
  resultArea.classList.remove("visible");

  chrome.runtime.sendMessage({ action: "capture-screenshot" }, async (response) => {
    if (!response || response.error) {
      showError(response?.error || "Failed to capture screenshot.");
      setLoading(false);
      return;
    }

    try {
      const blob = dataURLtoBlob(response.dataUrl);
      const formData = new FormData();
      formData.append("file", blob, "screenshot.png");
      formData.append("subject", subjectSelect.value);
      formData.append("model", "auto");
      formData.append("enhance_img", "true");

      const apiRes = await fetch(`${API_BASE}/ocr`, {
        method: "POST",
        headers: authHeaders(),
        body: formData
      });

      if (!apiRes.ok) {
        const err = await apiRes.json();
        throw new Error(err.detail || `API error ${apiRes.status}`);
      }

      const data = await apiRes.json();
      renderResult(data.text);
      resultArea.classList.add("visible");
    } catch (e) {
      showError(e.message);
    }

    setLoading(false);
  });
});

let lastRawText = "";
function renderResult(text) {
  lastRawText = text;
  if (typeof katex === "undefined") {
    resultText.textContent = text;
    return;
  }
  let html = "";
  const blocks = text.split(/\$\$([\s\S]+?)\$\$/g);
  for (let i = 0; i < blocks.length; i++) {
    if (i % 2 === 1) {
      html += `<div class="math-block">${katex.renderToString(blocks[i], { displayMode: true, throwOnError: false })}</div>`;
    } else {
      const inline = blocks[i].split(/\$([^$]+?)\$/g);
      for (let j = 0; j < inline.length; j++) {
        if (j % 2 === 1) {
          html += katex.renderToString(inline[j], { displayMode: false, throwOnError: false });
        } else {
          html += escapeHtml(inline[j]);
        }
      }
    }
  }
  resultText.innerHTML = html;
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

copyBtn.addEventListener("click", () => {
  navigator.clipboard.writeText(lastRawText || resultText.textContent).then(() => {
    copyBtn.textContent = "\u2705 Copied!";
    setTimeout(() => {
      copyBtn.innerHTML = "&#128203; Copy to Clipboard";
    }, 2000);
  });
});