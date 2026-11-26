(() => {
  "use strict";

  const appState = {
    statePayload: null,
    messages: [],
    taskFilter: "all",
    activeTargetMsgId: null,
    qaHistory: [],
  };

  const els = {
    metaFile: document.getElementById("meta-file"),
    metaMessages: document.getElementById("meta-messages"),
    metaOrder: document.getElementById("meta-order"),
    metaLastTs: document.getElementById("meta-last-ts"),
    metaProvider: document.getElementById("meta-provider"),
    metaCache: document.getElementById("meta-cache"),
    metaParticipants: document.getElementById("meta-participants"),
    globalStatus: document.getElementById("global-status"),
    summaryCounts: document.getElementById("summary-counts"),
    tasksTbody: document.getElementById("tasks-tbody"),
    decisionsTbody: document.getElementById("decisions-tbody"),
    deadlinesTbody: document.getElementById("deadlines-tbody"),
    openTbody: document.getElementById("open-tbody"),
    filterChips: Array.from(document.querySelectorAll("[data-task-filter]")),
    qaForm: document.getElementById("qa-form"),
    questionInput: document.getElementById("question-input"),
    questionError: document.getElementById("question-error"),
    askSubmitBtn: document.getElementById("ask-submit-btn"),
    starterChips: Array.from(document.querySelectorAll(".starter-chip")),
    qaResult: document.getElementById("qa-result"),
    qaHistoryList: document.getElementById("qa-history-list"),
    rebuildBtn: document.getElementById("rebuild-btn"),
    openTranscriptBtn: document.getElementById("open-transcript-btn"),
    transcriptDialog: document.getElementById("transcript-dialog"),
    closeTranscriptBtn: document.getElementById("close-transcript-btn"),
    transcriptContextNote: document.getElementById("transcript-context-note"),
    transcriptFilter: document.getElementById("transcript-filter"),
    transcriptList: document.getElementById("transcript-list"),
  };

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text !== undefined && text !== null) {
      node.textContent = String(text);
    }
    return node;
  }

  function formatLineRef(lineStart, lineEnd) {
    return lineStart === lineEnd ? `L${lineStart}` : `L${lineStart}-L${lineEnd}`;
  }

  function formatEvidenceRefLabel(ref) {
    const senderLabel = ref.sender ? ref.sender : "system";
    const lineRef = formatLineRef(ref.line_start, ref.line_end);
    return `[${ref.msg_id} | ${lineRef} | ${ref.timestamp_raw} | ${senderLabel}]`;
  }

  function statusToneClass(status) {
    if (["confirmed", "completed", "resolved", "verified"].includes(status)) {
      return "status-ok";
    }
    if (
      [
        "reversed",
        "proposed",
        "tentative",
        "unconfirmed",
        "deferred",
        "open",
        "verified partial",
      ].includes(status)
    ) {
      return "status-warn";
    }
    if (["unassigned", "blocked", "abstained"].includes(status)) {
      return "status-bad";
    }
    if (["assigned", "in_progress", "rule_based"].includes(status)) {
      return "status-info";
    }
    return "status-muted";
  }

  function createStatusBadge(label) {
    return el("span", `status-badge ${statusToneClass(label)}`, `[${label}]`);
  }

  function createEmptyRow(colSpan, message) {
    const tr = el("tr");
    const td = el("td", "empty-note", message);
    td.colSpan = colSpan;
    tr.appendChild(td);
    return tr;
  }

  function createCell(labelText) {
    const td = el("td");
    td.dataset.label = labelText;
    return td;
  }

  function createCitationList(refs) {
    const wrap = el("div", "citation-list");
    if (!refs || refs.length === 0) {
      wrap.appendChild(el("span", "empty-note", "[no verified citations]"));
      return wrap;
    }
    for (const ref of refs) {
      const btn = el("button", "citation-chip", formatEvidenceRefLabel(ref));
      btn.type = "button";
      btn.addEventListener("click", () => openTranscriptAt(ref.msg_id));
      wrap.appendChild(btn);
    }
    return wrap;
  }

  function createHistoryDisclosure(historyEntries, listId, renderEntryText) {
    const container = el("div");
    const btn = el("button", "history-toggle-btn", `Show history (${historyEntries.length})`);
    btn.type = "button";
    btn.setAttribute("aria-expanded", "false");
    btn.setAttribute("aria-controls", listId);

    const list = el("ul", "history-trail");
    list.id = listId;
    list.hidden = true;

    for (const entry of historyEntries) {
      const li = el("li");
      const summary = el("div", "", renderEntryText(entry));
      li.appendChild(summary);
      if (entry.evidence && entry.evidence.length > 0) {
        const meta = el("div", "history-trail-meta");
        meta.appendChild(createCitationList(entry.evidence));
        li.appendChild(meta);
      }
      list.appendChild(li);
    }

    btn.addEventListener("click", () => {
      const nextExpanded = btn.getAttribute("aria-expanded") !== "true";
      btn.setAttribute("aria-expanded", nextExpanded ? "true" : "false");
      list.hidden = !nextExpanded;
      btn.textContent = nextExpanded
        ? `Hide history (${historyEntries.length})`
        : `Show history (${historyEntries.length})`;
    });

    container.appendChild(btn);
    container.appendChild(list);
    return container;
  }

  function setGlobalStatus(message, isError = false) {
    els.globalStatus.textContent = message;
    els.globalStatus.classList.toggle("is-error", Boolean(isError));
  }

  function baseFilename(pathStr) {
    if (!pathStr) return "unknown";
    const parts = pathStr.split(/[\\/]/);
    return parts[parts.length - 1] || pathStr;
  }

  function renderMetadata(payload) {
    els.metaFile.textContent = `File: ${baseFilename(payload.chat_file)}`;
    els.metaMessages.textContent = `Messages: ${payload.total_messages} (${payload.total_source_lines} lines)`;
    els.metaOrder.textContent = `Date order: ${payload.date_order}`;
    els.metaLastTs.textContent = `Last message: ${payload.last_message_timestamp || "n/a"}`;
    els.metaProvider.textContent = `Model: ${payload.provider} (${payload.model})`;
    els.metaCache.textContent = `Source: ${payload.from_cache ? "cache" : "compiled"}`;

    const participants =
      payload.participants && payload.participants.length > 0
        ? payload.participants.join(", ")
        : "none";
    els.metaParticipants.textContent = `Participants: ${participants}`;

    const openCount = collectOpenItems(payload).length;
    const unassignedCount = (payload.tasks || []).filter(
      (t) => t.status === "unassigned"
    ).length;
    const pills = [
      `Decisions: ${(payload.decisions || []).length}`,
      `Tasks: ${(payload.tasks || []).length}`,
      `Deadlines: ${(payload.deadlines || []).length}`,
      `Open items: ${openCount}`,
      `Unassigned tasks: ${unassignedCount}`,
    ];
    els.summaryCounts.replaceChildren(
      ...pills.map((label) => el("span", "summary-pill", label))
    );
  }

  function renderTasksTable(tasks) {
    const filtered = tasks.filter((task) => {
      if (appState.taskFilter === "all") return true;
      return task.status === appState.taskFilter;
    });

    if (filtered.length === 0) {
      const msg =
        tasks.length === 0
          ? "No tasks extracted from this chat."
          : `No tasks match filter "${appState.taskFilter}".`;
      els.tasksTbody.replaceChildren(createEmptyRow(5, msg));
      return;
    }

    const rows = filtered.map((task, idx) => {
      const tr = el("tr");

      const taskCell = createCell("Task");
      taskCell.appendChild(el("span", "cell-title", task.title));
      taskCell.appendChild(el("span", "cell-subkey", task.task_key));
      tr.appendChild(taskCell);

      const ownerCell = createCell("Owner");
      ownerCell.textContent = task.current_owner || "unassigned";
      tr.appendChild(ownerCell);

      const statusCell = createCell("Status");
      statusCell.appendChild(createStatusBadge(task.status));
      tr.appendChild(statusCell);

      const detailsCell = createCell("Details and history");
      if (task.status_note) {
        detailsCell.appendChild(el("div", "", `Note: ${task.status_note}`));
      } else if (!task.history || task.history.length <= 1) {
        detailsCell.appendChild(el("span", "empty-note", "No additional notes"));
      }
      if (task.history && task.history.length > 1) {
        detailsCell.appendChild(
          createHistoryDisclosure(
            task.history,
            `task-history-${idx}`,
            (entry) =>
              `${entry.timestamp_raw}: ${entry.action} (owner: ${
                entry.owner || "unassigned"
              })` + (entry.note ? `, ${entry.note}` : "")
          )
        );
      }
      tr.appendChild(detailsCell);

      const citationsCell = createCell("Citations");
      citationsCell.appendChild(createCitationList(task.evidence));
      tr.appendChild(citationsCell);

      return tr;
    });

    els.tasksTbody.replaceChildren(...rows);
  }

  function renderDecisionsTable(decisions) {
    if (!decisions || decisions.length === 0) {
      els.decisionsTbody.replaceChildren(
        createEmptyRow(5, "No decisions extracted from this chat.")
      );
      return;
    }

    const rows = decisions.map((decision, idx) => {
      const tr = el("tr");

      const topicCell = createCell("Topic");
      topicCell.appendChild(el("span", "cell-title", decision.title));
      topicCell.appendChild(el("span", "cell-subkey", decision.topic_key));
      tr.appendChild(topicCell);

      const valueCell = createCell("Current decision");
      valueCell.textContent = decision.current_value;
      tr.appendChild(valueCell);

      const statusCell = createCell("Status");
      statusCell.appendChild(createStatusBadge(decision.status));
      tr.appendChild(statusCell);

      const historyCell = createCell("History");
      if (decision.history && decision.history.length > 1) {
        const prior = decision.history
          .slice(0, -1)
          .map((e) => `${e.value} (${e.action})`)
          .join(" -> ");
        historyCell.appendChild(el("div", "", `Prior: ${prior}`));
        historyCell.appendChild(
          createHistoryDisclosure(
            decision.history,
            `decision-history-${idx}`,
            (entry) =>
              `${entry.timestamp_raw}: [${entry.action}] ${entry.value}` +
              (entry.note ? ` (${entry.note})` : "")
          )
        );
      } else if (decision.history && decision.history[0] && decision.history[0].note) {
        historyCell.textContent = decision.history[0].note;
      } else {
        historyCell.appendChild(el("span", "empty-note", "Single entry"));
      }
      tr.appendChild(historyCell);

      const citationsCell = createCell("Citations");
      citationsCell.appendChild(createCitationList(decision.evidence));
      tr.appendChild(citationsCell);

      return tr;
    });

    els.decisionsTbody.replaceChildren(...rows);
  }

  function renderDeadlinesTable(deadlines) {
    if (!deadlines || deadlines.length === 0) {
      els.deadlinesTbody.replaceChildren(
        createEmptyRow(5, "No deadlines extracted from this chat.")
      );
      return;
    }

    const rows = deadlines.map((deadline, idx) => {
      const tr = el("tr");

      const milestoneCell = createCell("Milestone");
      milestoneCell.appendChild(el("span", "cell-title", deadline.title));
      milestoneCell.appendChild(el("span", "cell-subkey", deadline.item_key));
      tr.appendChild(milestoneCell);

      const dateCell = createCell("Current date");
      dateCell.textContent = deadline.current_date_text;
      tr.appendChild(dateCell);

      const statusCell = createCell("Status");
      statusCell.appendChild(createStatusBadge(deadline.status));
      tr.appendChild(statusCell);

      const ruleCell = createCell("Rule and shifts");
      if (deadline.dependency_rule) {
        ruleCell.appendChild(el("div", "", `Rule: ${deadline.dependency_rule}`));
      }
      if (deadline.history && deadline.history.length > 1) {
        const shifts = deadline.history
          .slice(0, -1)
          .map((e) => `${e.value} (${e.timestamp_raw})`)
          .join(" -> ");
        ruleCell.appendChild(el("div", "", `Prior shifts: ${shifts}`));
        ruleCell.appendChild(
          createHistoryDisclosure(
            deadline.history,
            `deadline-history-${idx}`,
            (entry) =>
              `${entry.timestamp_raw}: [${entry.action}] ${entry.value}` +
              (entry.note ? ` (rule: ${entry.note})` : "")
          )
        );
      } else if (!deadline.dependency_rule) {
        ruleCell.appendChild(el("span", "empty-note", "No dependency rule"));
      }
      tr.appendChild(ruleCell);

      const citationsCell = createCell("Citations");
      citationsCell.appendChild(createCitationList(deadline.evidence));
      tr.appendChild(citationsCell);

      return tr;
    });

    els.deadlinesTbody.replaceChildren(...rows);
  }

  function collectOpenItems(payload) {
    const items = [];
    for (const item of payload.open_items || []) {
      if (item.status === "resolved") continue;
      const detail = item.owner
        ? `${item.description} (owner: ${item.owner})`
        : item.description;
      items.push({
        category: `issue:${item.status}`,
        title: item.title,
        key: item.issue_key,
        detail,
        evidence: item.evidence || [],
      });
    }
    for (const task of payload.tasks || []) {
      if (["unassigned", "unconfirmed", "blocked"].includes(task.status)) {
        const ownerText = task.current_owner
          ? `owner: ${task.current_owner}`
          : "unassigned";
        const detail = task.status_note
          ? `${ownerText}; ${task.status_note}`
          : ownerText;
        items.push({
          category: `task:${task.status}`,
          title: task.title,
          key: task.task_key,
          detail,
          evidence: task.evidence || [],
        });
      }
    }
    for (const decision of payload.decisions || []) {
      if (["uncertain", "proposed"].includes(decision.status)) {
        items.push({
          category: `decision:${decision.status}`,
          title: decision.title,
          key: decision.topic_key,
          detail: decision.current_value,
          evidence: decision.evidence || [],
        });
      }
    }
    return items;
  }

  function renderOpenTable(payload) {
    const openRows = collectOpenItems(payload);
    if (openRows.length === 0) {
      els.openTbody.replaceChildren(
        createEmptyRow(4, "No open, unassigned, or unconfirmed items.")
      );
      return;
    }

    const rows = openRows.map((entry) => {
      const tr = el("tr");

      const categoryCell = createCell("Category");
      categoryCell.appendChild(createStatusBadge(entry.category));
      tr.appendChild(categoryCell);

      const itemCell = createCell("Item");
      itemCell.appendChild(el("span", "cell-title", entry.title));
      itemCell.appendChild(el("span", "cell-subkey", entry.key));
      tr.appendChild(itemCell);

      const detailCell = createCell("Details");
      detailCell.textContent = entry.detail;
      tr.appendChild(detailCell);

      const citationsCell = createCell("Citations");
      citationsCell.appendChild(createCitationList(entry.evidence));
      tr.appendChild(citationsCell);

      return tr;
    });

    els.openTbody.replaceChildren(...rows);
  }

  function renderProjectState(payload) {
    appState.statePayload = payload;
    renderMetadata(payload);
    renderTasksTable(payload.tasks || []);
    renderDecisionsTable(payload.decisions || []);
    renderDeadlinesTable(payload.deadlines || []);
    renderOpenTable(payload);
  }

  function verificationDisplayBadge(result) {
    if (!result.supported) return "abstained";
    if (result.partial_status_only) return "verified partial";
    return "verified";
  }

  function renderQaResult(result) {
    const container = document.createDocumentFragment();
    const header = el("div", "qa-result-header");
    const badgeLabel = verificationDisplayBadge(result);
    header.appendChild(createStatusBadge(badgeLabel));

    const retrievalText = result.used_dense ? "hybrid (bm25 + embeddings)" : "bm25 fallback";
    header.appendChild(
      el(
        "span",
        "qa-meta-line",
        `status: ${result.verification_status} | retrieval: ${retrievalText}`
      )
    );
    container.appendChild(header);

    container.appendChild(el("p", "qa-answer-text", result.answer));

    if (result.rejected_msg_ids && result.rejected_msg_ids.length > 0) {
      container.appendChild(
        el(
          "p",
          "qa-meta-line",
          `Filtered unverified IDs: ${result.rejected_msg_ids.join(", ")}`
        )
      );
    }

    if (result.evidence && result.evidence.length > 0) {
      container.appendChild(el("strong", "", "Verified evidence:"));
      const list = el("ul", "evidence-items");
      for (const ref of result.evidence) {
        const li = el("li", "evidence-item");
        const chip = el("button", "citation-chip", formatEvidenceRefLabel(ref));
        chip.type = "button";
        chip.addEventListener("click", () => openTranscriptAt(ref.msg_id));
        li.appendChild(chip);
        li.appendChild(el("p", "evidence-quote", ref.quote));
        list.appendChild(li);
      }
      container.appendChild(list);
    } else {
      container.appendChild(el("p", "empty-note", "No verified citations."));
    }

    els.qaResult.replaceChildren(container);
  }

  function renderQaHistory() {
    if (appState.qaHistory.length === 0) {
      els.qaHistoryList.replaceChildren(
        el("li", "empty-note", "No questions asked in this session yet.")
      );
      return;
    }

    const items = appState.qaHistory.map((entry) => {
      const li = el("li");
      const btn = el("button", "qa-history-btn");
      btn.type = "button";
      const badge = verificationDisplayBadge(entry);
      btn.textContent = `[${badge}] ${entry.question}`;
      btn.addEventListener("click", () => {
        els.questionInput.value = entry.question;
        renderQaResult(entry);
      });
      li.appendChild(btn);
      return li;
    });

    els.qaHistoryList.replaceChildren(...items);
  }

  function renderTranscriptMessages(targetMsgId = null, filterQuery = "") {
    const messages = appState.messages || [];
    if (messages.length === 0) {
      els.transcriptList.replaceChildren(
        el("p", "empty-note", "No parsed messages available.")
      );
      return;
    }

    const normalizedFilter = filterQuery.trim().toLowerCase();
    const targetIdx = targetMsgId
      ? messages.findIndex((m) => m.msg_id === targetMsgId)
      : -1;

    const visibleMessages = normalizedFilter
      ? messages.filter((m) => {
          const sender = (m.sender || "system").toLowerCase();
          return (
            m.msg_id.toLowerCase().includes(normalizedFilter) ||
            sender.includes(normalizedFilter) ||
            m.text.toLowerCase().includes(normalizedFilter)
          );
        })
      : messages;

    if (visibleMessages.length === 0) {
      els.transcriptList.replaceChildren(
        el("p", "empty-note", "No transcript messages match that filter.")
      );
      return;
    }

    const fragment = document.createDocumentFragment();
    let targetNode = null;

    for (const msg of visibleMessages) {
      const idxInFull = messages.indexOf(msg);
      const isTarget = Boolean(targetMsgId && msg.msg_id === targetMsgId);
      const isContext = Boolean(
        targetIdx !== -1 &&
          !isTarget &&
          Math.abs(idxInFull - targetIdx) <= 2
      );

      const card = el("article", "transcript-item");
      card.dataset.msgId = msg.msg_id;
      if (isTarget) {
        card.classList.add("is-target");
        targetNode = card;
      } else if (isContext) {
        card.classList.add("is-context");
      }

      const head = el("div", "transcript-item-head");
      const lineRef = formatLineRef(msg.line_start, msg.line_end);
      const senderLabel = msg.sender || "system";
      head.appendChild(
        el("span", "", `[${msg.msg_id} | ${lineRef} | ${msg.timestamp_raw} | ${senderLabel}]`)
      );
      if (msg.kind !== "chat") {
        head.appendChild(createStatusBadge(msg.kind));
      }
      if (msg.is_edited) {
        head.appendChild(createStatusBadge("edited"));
      }
      card.appendChild(head);

      const body = el("p", "transcript-item-body", msg.text);
      card.appendChild(body);
      fragment.appendChild(card);
    }

    els.transcriptList.replaceChildren(fragment);

    if (targetNode) {
      requestAnimationFrame(() => {
        targetNode.scrollIntoView({ block: "center", behavior: "smooth" });
      });
    }
  }

  function openTranscriptAt(targetMsgId = null) {
    appState.activeTargetMsgId = targetMsgId;
    els.transcriptFilter.value = "";
    els.transcriptContextNote.textContent = targetMsgId
      ? `Focused on ${targetMsgId} with surrounding context`
      : `All ${appState.messages.length} chronological messages`;

    renderTranscriptMessages(targetMsgId, "");
    if (!els.transcriptDialog.open) {
      els.transcriptDialog.showModal();
    }
  }

  function setupDialogFallbacks() {
    if (!("closedBy" in HTMLDialogElement.prototype)) {
      const isOutside = (event, dialog) => {
        const rect = dialog.getBoundingClientRect();
        return (
          event.clientX < rect.left ||
          event.clientX > rect.right ||
          event.clientY < rect.top ||
          event.clientY > rect.bottom
        );
      };
      let pressedOutside = false;
      els.transcriptDialog.addEventListener("pointerdown", (event) => {
        pressedOutside = isOutside(event, els.transcriptDialog);
      });
      els.transcriptDialog.addEventListener("pointerup", (event) => {
        if (pressedOutside && isOutside(event, els.transcriptDialog)) {
          els.transcriptDialog.close();
        }
        pressedOutside = false;
      });
    }

    els.closeTranscriptBtn.addEventListener("click", () => {
      els.transcriptDialog.close();
    });

    els.openTranscriptBtn.addEventListener("click", () => {
      openTranscriptAt(null);
    });

    els.transcriptFilter.addEventListener("input", () => {
      renderTranscriptMessages(
        appState.activeTargetMsgId,
        els.transcriptFilter.value
      );
    });
  }

  async function fetchJson(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail =
        typeof data.detail === "string"
          ? data.detail
          : Array.isArray(data.detail) && data.detail[0] && data.detail[0].msg
            ? data.detail[0].msg
            : `Request failed with HTTP ${response.status}`;
      throw new Error(detail);
    }
    return data;
  }

  async function loadInitialData() {
    setGlobalStatus("Loading project state and transcript...");
    try {
      const [statePayload, messagesPayload] = await Promise.all([
        fetchJson("/api/state"),
        fetchJson("/api/messages"),
      ]);
      appState.messages = messagesPayload.messages || [];
      renderProjectState(statePayload);
      setGlobalStatus("");
    } catch (err) {
      setGlobalStatus(`Failed to load dashboard data: ${err.message}`, true);
    }
  }

  function setQuestionError(message) {
    els.questionError.textContent = message;
    if (message) {
      els.questionInput.setAttribute("aria-invalid", "true");
    } else {
      els.questionInput.removeAttribute("aria-invalid");
    }
  }

  async function submitQuestion(rawQuestion) {
    const question = rawQuestion.trim();
    if (!question) {
      setQuestionError("Please enter a question before submitting.");
      els.questionInput.focus();
      return;
    }

    setQuestionError("");
    els.askSubmitBtn.disabled = true;
    els.askSubmitBtn.textContent = "Verifying...";
    els.qaResult.replaceChildren(
      el("p", "empty-note", "Retrieving evidence and verifying citations...")
    );

    try {
      const result = await fetchJson("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      renderQaResult(result);
      appState.qaHistory = [
        result,
        ...appState.qaHistory.filter((item) => item.question !== result.question),
      ].slice(0, 8);
      renderQaHistory();
    } catch (err) {
      els.qaResult.replaceChildren(
        el("p", "field-error", `Query failed: ${err.message}`)
      );
    } finally {
      els.askSubmitBtn.disabled = false;
      els.askSubmitBtn.textContent = "Ask question";
    }
  }

  async function handleRebuild() {
    els.rebuildBtn.disabled = true;
    els.rebuildBtn.textContent = "Rebuilding...";
    setGlobalStatus("Recompiling project state and rebuilding retrieval index...");
    try {
      const [rebuiltState, messagesPayload] = await Promise.all([
        fetchJson("/api/rebuild", { method: "POST" }),
        fetchJson("/api/messages"),
      ]);
      appState.messages = messagesPayload.messages || [];
      renderProjectState(rebuiltState);
      setGlobalStatus("Project state and retrieval index rebuilt.");
    } catch (err) {
      setGlobalStatus(`Rebuild failed: ${err.message}`, true);
    } finally {
      els.rebuildBtn.disabled = false;
      els.rebuildBtn.textContent = "Rebuild state";
    }
  }

  function setupEvents() {
    for (const chip of els.filterChips) {
      chip.addEventListener("click", () => {
        appState.taskFilter = chip.dataset.taskFilter || "all";
        for (const other of els.filterChips) {
          const active = other === chip;
          other.classList.toggle("is-active", active);
          other.setAttribute("aria-pressed", active ? "true" : "false");
        }
        if (appState.statePayload) {
          renderTasksTable(appState.statePayload.tasks || []);
        }
      });
    }

    for (const chip of els.starterChips) {
      chip.addEventListener("click", () => {
        const question = chip.dataset.question || chip.textContent || "";
        els.questionInput.value = question.trim();
        submitQuestion(els.questionInput.value);
      });
    }

    els.questionInput.addEventListener("input", () => {
      if (els.questionInput.value.trim()) {
        setQuestionError("");
      }
    });

    els.questionInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        submitQuestion(els.questionInput.value);
      }
    });

    els.qaForm.addEventListener("submit", (event) => {
      event.preventDefault();
      submitQuestion(els.questionInput.value);
    });

    els.rebuildBtn.addEventListener("click", handleRebuild);
    setupDialogFallbacks();
  }

  setupEvents();
  loadInitialData();
})();
