const apiBaseInput = document.getElementById("apiBase");
const apiKeyInput = document.getElementById("apiKey");
const saveBtn = document.getElementById("saveBtn");
const statusEl = document.getElementById("status");

chrome.storage.local.get(
  { apiBase: "http://localhost:8000", apiKey: "" },
  (items) => {
    apiBaseInput.value = items.apiBase;
    apiKeyInput.value = items.apiKey;
  }
);

saveBtn.addEventListener("click", () => {
  let base = apiBaseInput.value.trim() || "http://localhost:8000";
  base = base.replace(/\/+$/, ""); 
  chrome.storage.local.set(
    { apiBase: base, apiKey: apiKeyInput.value.trim() },
    () => {
      statusEl.textContent = "Saved. Reopen the popup to apply.";
      setTimeout(() => (statusEl.textContent = ""), 2500);
    }
  );
});
