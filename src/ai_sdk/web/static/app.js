const state = {
  mode: "single",
  activeRunId: null,
  running: false,
  citations: [],
};

const elements = {
  activityList: document.querySelector("#activity-list"),
  cancelButton: document.querySelector("#cancel-button"),
  chatForm: document.querySelector("#chat-form"),
  composerState: document.querySelector("#composer-state"),
  documentCount: document.querySelector("#document-count"),
  documentList: document.querySelector("#document-list"),
  emptyState: document.querySelector("#empty-state"),
  fileInput: document.querySelector("#file-input"),
  messageInput: document.querySelector("#message-input"),
  messages: document.querySelector("#messages"),
  modeCopy: document.querySelector("#mode-copy"),
  provider: document.querySelector("#provider"),
  resetButton: document.querySelector("#reset-button"),
  routeSummary: document.querySelector("#route-summary"),
  sendButton: document.querySelector("#send-button"),
  sourceList: document.querySelector("#source-list"),
  systemStatus: document.querySelector("#system-status"),
  toast: document.querySelector("#toast"),
  uploadMessage: document.querySelector("#upload-message"),
  uploadZone: document.querySelector("#upload-zone"),
};

document.querySelectorAll(".segment").forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});

document.querySelectorAll(".prompt-card").forEach((button) => {
  button.addEventListener("click", () => {
    elements.messageInput.value = button.textContent.trim();
    elements.messageInput.focus();
    resizeComposer();
  });
});

elements.fileInput.addEventListener("change", () => {
  if (elements.fileInput.files[0]) uploadFile(elements.fileInput.files[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
  elements.uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.uploadZone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  elements.uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.uploadZone.classList.remove("dragging");
  });
});

elements.uploadZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) uploadFile(file);
});

elements.messageInput.addEventListener("input", resizeComposer);
elements.messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.chatForm.requestSubmit();
  }
});
elements.chatForm.addEventListener("submit", sendMessage);
elements.cancelButton.addEventListener("click", cancelRun);
elements.resetButton.addEventListener("click", resetConversation);

async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `Request failed (${response.status}).`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (_) {
      // Keep the status-based message when the response is not JSON.
    }
    throw new Error(detail);
  }
  if (response.status === 204) return null;
  return response.json();
}

async function refreshStatus() {
  try {
    const status = await request("/api/status");
    const readyProviders = status.providers.filter((provider) => provider.ready);
    elements.systemStatus.className = `status-pill ${readyProviders.length ? "ready" : "warning"}`;
    elements.systemStatus.innerHTML = "";
    const dot = document.createElement("span");
    dot.className = "status-dot";
    elements.systemStatus.append(dot, document.createTextNode(
      readyProviders.length ? `${readyProviders.length}/3 providers ready` : "Add provider keys"
    ));

    status.providers.forEach((provider) => {
      const option = elements.provider.querySelector(`option[value="${provider.provider}"]`);
      if (option) {
        option.disabled = !provider.ready;
        option.textContent = `${provider.display_name}${provider.ready ? " · Ready" : " · Not configured"}`;
      }
    });
    if (elements.provider.selectedOptions[0]?.disabled && readyProviders[0]) {
      elements.provider.value = readyProviders[0].provider;
    }
    const metrics = status.adaptive_metrics;
    document.querySelector("#metric-runs").textContent = metrics.total_runs;
    document.querySelector("#metric-success").textContent = metrics.total_runs
      ? `${Math.round((metrics.successful_runs / metrics.total_runs) * 100)}%`
      : "—";
  } catch (error) {
    elements.systemStatus.className = "status-pill warning";
    elements.systemStatus.textContent = "Server unavailable";
  }
}

async function refreshDocuments() {
  try {
    const documents = await request("/api/documents");
    elements.documentCount.textContent = documents.length;
    elements.documentList.replaceChildren();
    documents.forEach((documentItem) => {
      const item = document.createElement("div");
      item.className = "document-item";

      const type = document.createElement("span");
      type.className = "document-type";
      type.textContent = documentItem.format;

      const copy = document.createElement("div");
      copy.className = "document-copy";
      const name = document.createElement("strong");
      name.textContent = documentItem.source;
      const metadata = document.createElement("small");
      metadata.textContent = documentItem.page_count
        ? `${documentItem.page_count} pages · ${documentItem.chunk_count} chunks`
        : `${documentItem.chunk_count} chunks`;
      copy.append(name, metadata);

      const remove = document.createElement("button");
      remove.className = "document-remove";
      remove.type = "button";
      remove.title = "Remove document";
      remove.textContent = "×";
      remove.addEventListener("click", () => removeDocument(documentItem.document_id));
      item.append(type, copy, remove);
      elements.documentList.append(item);
    });
  } catch (error) {
    showToast(error.message, true);
  }
}

async function uploadFile(file) {
  elements.uploadMessage.className = "inline-message";
  elements.uploadMessage.textContent = `Indexing ${file.name}…`;
  const body = new FormData();
  body.append("file", file);
  try {
    await request("/api/documents", { method: "POST", body });
    elements.uploadMessage.textContent = `${file.name} is ready for retrieval.`;
    await refreshDocuments();
    showToast("Document indexed successfully.");
  } catch (error) {
    elements.uploadMessage.className = "inline-message error";
    elements.uploadMessage.textContent = error.message;
  } finally {
    elements.fileInput.value = "";
  }
}

async function removeDocument(documentId) {
  try {
    await request(`/api/documents/${encodeURIComponent(documentId)}`, { method: "DELETE" });
    await refreshDocuments();
    showToast("Document removed.");
  } catch (error) {
    showToast(error.message, true);
  }
}

function setMode(mode) {
  if (state.running) return;
  state.mode = mode;
  document.querySelectorAll(".segment").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  const adaptive = mode === "adaptive";
  elements.provider.disabled = adaptive;
  elements.modeCopy.textContent = adaptive
    ? "A deterministic route selects one, two, or three providers for the request."
    : "One provider answers with the shared RAG context.";
}

async function sendMessage(event) {
  event.preventDefault();
  const message = elements.messageInput.value.trim();
  if (!message || state.running) return;

  appendMessage("user", message);
  const answerBubble = appendMessage("assistant", "Working through the request…", true);
  elements.messageInput.value = "";
  resizeComposer();
  resetEvidence();
  setRunning(true);

  const payload = { message, mode: state.mode };
  if (state.mode === "single") payload.provider = elements.provider.value;

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const body = await response.json();
      throw new Error(body.detail || `Request failed (${response.status}).`);
    }
    state.activeRunId = response.headers.get("X-Run-ID");
    await consumeEventStream(response.body, (type, data) => {
      handleStreamEvent(type, data, answerBubble);
    });
  } catch (error) {
    answerBubble.className = "bubble error";
    answerBubble.textContent = error.message;
  } finally {
    setRunning(false);
    state.activeRunId = null;
    refreshStatus();
  }
}

async function consumeEventStream(stream, onEvent) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";
    blocks.forEach((block) => {
      let type = "message";
      let data = null;
      block.split("\n").forEach((line) => {
        if (line.startsWith("event: ")) type = line.slice(7);
        if (line.startsWith("data: ")) data = JSON.parse(line.slice(6));
      });
      if (data !== null) onEvent(type, data);
    });
    if (done) break;
  }
}

function handleStreamEvent(type, data, answerBubble) {
  if (type === "run") state.activeRunId = data.run_id;
  if (type === "route") {
    const routeName = data.route.replaceAll("_", " ");
    elements.routeSummary.className = "route-summary active";
    elements.routeSummary.querySelector("strong").textContent = routeName;
    elements.composerState.textContent = `${routeName} route selected`;
  }
  if (type === "stage") addActivity(data);
  if (type === "citations") {
    state.citations = data;
    renderSources(data);
  }
  if (type === "answer") {
    answerBubble.className = "bubble";
    answerBubble.textContent = data.content;
  }
  if (type === "cancelled") {
    answerBubble.className = "bubble error";
    answerBubble.textContent = "The workflow was cancelled at a safe boundary.";
  }
  if (type === "error") {
    answerBubble.className = "bubble error";
    answerBubble.textContent = data.message;
  }
}

function addActivity(data) {
  if (elements.activityList.querySelector(".activity-placeholder")) {
    elements.activityList.replaceChildren();
  }
  const item = document.createElement("div");
  const shortStatus = data.status.replace("stage_", "").replace("workflow_", "");
  item.className = `activity-item ${shortStatus}`;
  const node = document.createElement("span");
  node.className = "activity-node";
  const label = document.createElement("strong");
  label.textContent = data.stage_id || "workflow";
  const status = document.createElement("small");
  status.textContent = shortStatus;
  item.append(node, label, status);
  elements.activityList.append(item);
}

function renderSources(citations) {
  elements.sourceList.replaceChildren();
  if (!citations.length) {
    const empty = document.createElement("div");
    empty.className = "source-placeholder";
    empty.textContent = "No indexed source was retrieved for this answer.";
    elements.sourceList.append(empty);
    return;
  }
  citations.forEach((citation) => {
    const card = document.createElement("article");
    card.className = "source-card";
    const header = document.createElement("header");
    const name = document.createElement("strong");
    name.textContent = citation.source;
    const position = document.createElement("span");
    position.textContent = `[${citation.position}]`;
    header.append(name, position);
    const detail = document.createElement("small");
    const page = citation.page ? `Page ${citation.page} · ` : "";
    detail.textContent = `${page}relevance ${citation.score.toFixed(3)}`;
    card.append(header, detail);
    elements.sourceList.append(card);
  });
}

function resetEvidence() {
  elements.routeSummary.className = "route-summary neutral";
  elements.routeSummary.querySelector("strong").textContent = state.mode === "adaptive"
    ? "Selecting route"
    : `${elements.provider.value} · single`;
  elements.activityList.innerHTML = '<div class="activity-placeholder">Waiting for execution progress…</div>';
  elements.sourceList.innerHTML = '<div class="source-placeholder">Retrieving relevant context…</div>';
}

function appendMessage(role, content, pending = false) {
  elements.emptyState.classList.add("hidden");
  const row = document.createElement("div");
  row.className = `message ${role}`;
  const avatar = document.createElement("span");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "YOU" : "AK";
  const bubble = document.createElement("div");
  bubble.className = `bubble${pending ? " pending" : ""}`;
  bubble.textContent = content;
  row.append(avatar, bubble);
  elements.messages.append(row);
  elements.messages.scrollTop = elements.messages.scrollHeight;
  return bubble;
}

async function cancelRun() {
  if (!state.activeRunId) return;
  try {
    const result = await request(`/api/runs/${state.activeRunId}/cancel`, { method: "POST" });
    elements.composerState.textContent = result.accepted
      ? "Cancellation requested"
      : "This stage cannot be interrupted";
  } catch (error) {
    showToast(error.message, true);
  }
}

async function resetConversation() {
  if (state.running) return;
  const payload = { mode: state.mode };
  if (state.mode === "single") payload.provider = elements.provider.value;
  try {
    await request("/api/conversations/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    elements.messages.replaceChildren();
    elements.emptyState.classList.remove("hidden");
    resetEvidence();
    showToast("Conversation reset.");
  } catch (error) {
    showToast(error.message, true);
  }
}

function setRunning(running) {
  state.running = running;
  elements.sendButton.disabled = running;
  elements.provider.disabled = running || state.mode === "adaptive";
  elements.cancelButton.classList.toggle("hidden", !running || state.mode !== "adaptive");
  elements.composerState.textContent = running ? "Running" : "Ready";
}

function resizeComposer() {
  elements.messageInput.style.height = "auto";
  elements.messageInput.style.height = `${Math.min(elements.messageInput.scrollHeight, 160)}px`;
}

let toastTimer = null;
function showToast(message, error = false) {
  clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.className = `toast visible${error ? " error" : ""}`;
  toastTimer = setTimeout(() => { elements.toast.className = "toast"; }, 3200);
}

setMode("single");
refreshStatus();
refreshDocuments();
