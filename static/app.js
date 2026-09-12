// TripCraft frontend — chat input + draft panel + approve/revise controls.
// Talks to POST /api/travel and POST /api/travel/approve. No frameworks,
// no build step, just fetch() + small DOM helpers.

const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");

const statusPill = document.getElementById("status-pill");
const draftContent = document.getElementById("draft-content");
const approvalControls = document.getElementById("approval-controls");
const finalizedControls = document.getElementById("finalized-controls");
const approveBtn = document.getElementById("approve-btn");
const reviseToggleBtn = document.getElementById("revise-toggle-btn");
const reviseForm = document.getElementById("revise-form");
const reviseFeedback = document.getElementById("revise-feedback");
const reviseSubmitBtn = document.getElementById("revise-submit-btn");
const planAnotherBtn = document.getElementById("plan-another-btn");
const providerBadge = document.getElementById("provider-badge");

let currentThreadId = null;

function addMessage(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `msg msg-${role}`;
  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);
  chatLog.appendChild(wrap);
  chatLog.scrollTop = chatLog.scrollHeight;
  return bubble;
}

function addTypingIndicator() {
  const wrap = document.createElement("div");
  wrap.className = "msg msg-assistant";
  wrap.id = "typing-indicator";
  wrap.innerHTML = `<div class="msg-bubble typing-dots"><span></span><span></span><span></span></div>`;
  chatLog.appendChild(wrap);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function removeTypingIndicator() {
  document.getElementById("typing-indicator")?.remove();
}

function setStatus(status) {
  statusPill.className = `status-pill status-${status}`;
  statusPill.textContent = status.replace(/_/g, " ");
}

// Small, dependency-free Markdown -> HTML for the draft panel (headers,
// bold/italic, blockquotes, unordered lists, GFM-style pipe tables,
// paragraphs). Good enough for the structured plan text the Trip Compiler
// produces; not a full parser.
function renderMarkdown(md) {
  const escape = (s) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  // LLMs sometimes emit a literal HTML <br> inside table cells for a soft
  // line break -- allow just that one tag back through after escaping
  // (everything else stays escaped, so this isn't a general HTML injection
  // hole, just un-escaping a tag we ourselves put back).
  const lines = escape(md)
    .replace(/&lt;br\s*\/?&gt;/gi, "<br>")
    .split("\n");

  let html = "";
  let inList = false;
  let inQuote = false;

  const inline = (line) =>
    line
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.+?)\*/g, "<em>$1</em>")
      .replace(/`(.+?)`/g, "<code>$1</code>");

  const closeList = () => { if (inList) { html += "</ul>"; inList = false; } };
  const closeQuote = () => { if (inQuote) { html += "</blockquote>"; inQuote = false; } };

  const isTableSeparator = (line) => {
    const s = line.trim();
    return s.length > 0 && /^[\s|:-]+$/.test(s) && s.includes("-") && s.includes("|");
  };
  const splitRow = (line) => {
    let s = line.trim();
    if (s.startsWith("|")) s = s.slice(1);
    if (s.endsWith("|")) s = s.slice(0, -1);
    return s.split("|").map((c) => c.trim());
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trimEnd();

    if (/^###\s+/.test(line)) { closeList(); closeQuote(); html += `<h3>${inline(line.replace(/^###\s+/, ""))}</h3>`; continue; }
    if (/^##\s+/.test(line)) { closeList(); closeQuote(); html += `<h2>${inline(line.replace(/^##\s+/, ""))}</h2>`; continue; }
    if (/^#\s+/.test(line)) { closeList(); closeQuote(); html += `<h1>${inline(line.replace(/^#\s+/, ""))}</h1>`; continue; }

    // GFM pipe table: a row line immediately followed by a |---|---| separator
    if (line.includes("|") && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      closeList(); closeQuote();
      const headerCells = splitRow(line);
      html += "<div class=\"table-wrap\"><table><thead><tr>" + headerCells.map((c) => `<th>${inline(c)}</th>`).join("") + "</tr></thead><tbody>";
      i += 2; // skip header + separator line
      while (i < lines.length && lines[i].trim() !== "" && lines[i].includes("|")) {
        html += "<tr>" + splitRow(lines[i]).map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>";
        i++;
      }
      html += "</tbody></table></div>";
      i--; // the for-loop's i++ accounts for the line the while-loop stopped on
      continue;
    }

    if (/^>\s?/.test(line)) {
      closeList();
      if (!inQuote) { html += "<blockquote>"; inQuote = true; }
      html += `<p>${inline(line.replace(/^>\s?/, ""))}</p>`;
      continue;
    }
    closeQuote();
    if (/^[-*]\s+/.test(line)) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${inline(line.replace(/^[-*]\s+/, ""))}</li>`;
      continue;
    }
    closeList();
    if (line.trim() === "") continue;
    html += `<p>${inline(line)}</p>`;
  }
  closeList();
  closeQuote();
  return html;
}

function showDraft(markdown) {
  draftContent.innerHTML = renderMarkdown(markdown || "_No draft available._");
}

function resetApprovalUI() {
  approvalControls.classList.add("hidden");
  finalizedControls.classList.add("hidden");
  reviseForm.classList.add("hidden");
}

async function fetchHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    providerBadge.textContent = `provider: ${data.llm_provider}`;
  } catch {
    providerBadge.textContent = "provider: unknown";
  }
}

async function handleResponse(status, draftPlan, finalPlan, message) {
  setStatus(status);

  if (status === "blocked" || status === "pending_clarification") {
    addMessage("system", message || "Something needs your attention before I can continue.");
    draftContent.innerHTML = `<p class="placeholder">${message ? "" : "Waiting for a valid trip request…"}</p>`;
    resetApprovalUI();
    return;
  }

  if (status === "pending_approval") {
    showDraft(draftPlan);
    addMessage("assistant", "Here's a draft trip plan — review it and Approve, or Request Changes with feedback.");
    resetApprovalUI();
    approvalControls.classList.remove("hidden");
    return;
  }

  if (status === "finalized") {
    showDraft(finalPlan);
    addMessage("assistant", "🎉 Your trip plan is finalized. Have a great trip!");
    resetApprovalUI();
    finalizedControls.classList.remove("hidden");
    return;
  }

  addMessage("system", message || `Unexpected status: ${status}`);
  resetApprovalUI();
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;

  addMessage("user", text);
  chatInput.value = "";
  sendBtn.disabled = true;
  setStatus("working");
  addTypingIndicator();

  try {
    const res = await fetch("/api/travel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    removeTypingIndicator();
    if (!res.ok) throw new Error(data.detail || "Request failed");
    currentThreadId = data.thread_id;
    await handleResponse(data.status, data.draft_plan, null, data.message);
  } catch (err) {
    removeTypingIndicator();
    setStatus("error");
    addMessage("system", `Error: ${err.message}`);
  } finally {
    sendBtn.disabled = false;
  }
});

approveBtn.addEventListener("click", () => submitDecision("approve"));
reviseToggleBtn.addEventListener("click", () => reviseForm.classList.toggle("hidden"));
reviseSubmitBtn.addEventListener("click", () => {
  const feedback = reviseFeedback.value.trim();
  if (!feedback) {
    reviseFeedback.focus();
    return;
  }
  addMessage("user", `Requested changes: ${feedback}`);
  reviseFeedback.value = "";
  submitDecision("revise", feedback);
});

planAnotherBtn.addEventListener("click", () => {
  currentThreadId = null;
  setStatus("idle");
  draftContent.innerHTML = `<p class="placeholder">Your draft itinerary will appear here once TripCraft's agents finish researching weather, itinerary, flights and hotels for your trip.</p>`;
  resetApprovalUI();
  chatInput.focus();
});

async function submitDecision(decision, feedback) {
  if (!currentThreadId) return;
  resetApprovalUI();
  setStatus("working");
  addTypingIndicator();
  try {
    const res = await fetch("/api/travel/approve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ thread_id: currentThreadId, decision, feedback }),
    });
    const data = await res.json();
    removeTypingIndicator();
    if (!res.ok) throw new Error(data.detail || "Request failed");
    await handleResponse(data.status, data.draft_plan, data.final_plan, null);
  } catch (err) {
    removeTypingIndicator();
    setStatus("error");
    addMessage("system", `Error: ${err.message}`);
  }
}

fetchHealth();
