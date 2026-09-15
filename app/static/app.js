// ClipForge shared header + live layer (vanilla JS, no build step).
// Wires: SSE live updates, notification bell + dropdown (with confirm-gated
// actions), schedule countdown, desktop notification on batch completion.
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));
  const escape = (x) => (x == null ? "" : String(x)).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  const CONFIRM_LABEL = { discard: "Confirm discard", cleanup_sources: "Confirm delete",
    cleanup_stale: "Confirm delete", update_ytdlp: "Confirm update" };

  async function api(url, opts) {
    const r = await fetch(url, opts);
    if (!r.ok) { let d = {}; try { d = await r.json(); } catch (e) {} throw new Error(d.detail || r.status); }
    return r.json();
  }

  // ---- notifications bell -------------------------------------------------
  const bell = $("#bell-btn"), dropdown = $("#notif-panel");
  async function loadNotifs() {
    if (!bell) return;
    try {
      const d = await api("/api/notifications");
      $("#bell-count").textContent = d.unread;
      $("#bell-count").style.display = d.unread ? "" : "none";
      renderNotifs(d.notifications);
    } catch (e) { /* ignore */ }
  }
  function renderNotifs(list) {
    if (!dropdown) return;
    dropdown.innerHTML = list.length ? list.map(n => `
      <div class="notif ${n.read_at ? 'read' : ''}" data-id="${n.id}">
        <span class="sev sev-${escape(n.severity)}"></span>
        <div class="notif-body">
          <div class="notif-title">${escape(n.title)}</div>
          <div class="notif-text">${escape(n.body || "")}</div>
          <div class="notif-actions">
            ${(n.actions || []).map(a =>
              `<button class="sm ${a.confirm ? 'danger-armed' : ''} ${a.action === 'nav' ? 'primary' : ''}"
                 data-action="${escape(a.action)}" data-confirm="${a.confirm ? 1 : 0}"
                 data-args='${escape(JSON.stringify(a.args || {}))}'>${escape(a.label)}</button>`).join("")}
          </div>
        </div>
      </div>`).join("") : '<div class="notif-empty">No notifications</div>';
    bindNotifActions();
  }
  function bindNotifActions() {
    $$("#notif-panel .notif-actions button").forEach(btn => {
      btn.addEventListener("click", async () => {
        const action = btn.dataset.action, args = JSON.parse(btn.dataset.args || "{}");
        const needsConfirm = btn.dataset.confirm === "1";
        const notif = btn.closest(".notif"); const id = notif && notif.dataset.id;
        if (action === "nav") { window.location.href = (args.to || "/dashboard"); return; }
        let confirm = false;
        if (needsConfirm) {
          if (!btn.classList.contains("armed")) {          // 2-step, no native confirm()
            btn.classList.add("armed"); btn.textContent = CONFIRM_LABEL[action] || "Confirm";
            setTimeout(() => { btn.classList.remove("armed"); btn.textContent = btn.dataset.label || btn.textContent; }, 4000);
            if (!btn.dataset.label) btn.dataset.label = btn.textContent;
            return;
          }
          confirm = true;
        }
        btn.disabled = true;
        try {
          const r = await api(`/api/actions/${encodeURIComponent(action)}`,
            { method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ args, confirm, notif_id: id ? +id : null }) });
          btn.closest(".notif").classList.add("acted");
          if (r.restart_required) setBanner("Restart ClipForge to apply the update.");
          loadNotifs();
        } catch (e) { btn.disabled = false; btn.textContent = "Failed: " + e.message; }
      });
    });
  }
  if (bell && dropdown) {
    bell.addEventListener("click", (e) => { e.stopPropagation(); dropdown.classList.toggle("open"); });
    document.addEventListener("click", (e) => { if (!dropdown.contains(e.target) && !bell.contains(e.target)) dropdown.classList.remove("open"); });
    loadNotifs();
    window.addEventListener("clipforge:reload-notifs", loadNotifs);
  }
  function setBanner(msg) { const b = $("#restart-banner"); if (b) { b.textContent = msg; b.classList.remove("hidden"); } }

  // ---- SSE live updates ---------------------------------------------------
  let es, esRetries = 0;
  function connectSSE() {
    try { es = new EventSource("/events"); } catch (e) { return; }
    es.onopen = () => { esRetries = 0; document.body.classList.add("live"); };
    es.addEventListener("job_progress", (ev) => {
      const d = JSON.parse(ev.data);
      const q = $("#quota-line"); if (q) q.textContent = `${d.detail || d.stage}`;
      document.title = `▶ ${d.stage} — ${d.detail || ""}`.slice(0, 60);
    });
    es.addEventListener("notification_new", () => loadNotifs());
    es.addEventListener("clip_rendered", (ev) => {
      const c = JSON.parse(ev.data);
      document.title = `✓ ${c.id} rendered`;
      if (window.onClipRendered) window.onClipRendered(c);   // review/briefing hook
      loadNotifs();
    });
    es.addEventListener("schedule_changed", (ev) => {
      const d = JSON.parse(ev.data);
      const nl = $("#next-run"); if (nl && d.next_run) nl.dataset.next = d.next_run;
    });
    es.onerror = () => { esRetries++; try { es.close(); } catch (e) {}
      setTimeout(connectSSE, Math.min(15000, 1000 * (esRetries || 1))); };  // EventSource auto-reconnects too
  }
  connectSSE();

  // ---- schedule countdown tick -------------------------------------------
  function tick() {
    const el = $("#next-run"); if (!el) return;
    const next = el.dataset.next; if (!next) { el.textContent = "manual"; return; }
    const ms = new Date(next) - Date.now();
    if (isNaN(ms)) { el.textContent = "—"; return; }
    const tot = Math.max(0, Math.floor(ms / 1000));
    const h = Math.floor(tot / 3600), m = Math.floor((tot % 3600) / 60), s = tot % 60;
    el.textContent = h ? `in ${h}h ${m}m` : `in ${m}m ${s}s`;
  }
  if ($("#next-run")) { tick(); setInterval(tick, 1000); }

  // ---- desktop notification permission (opt-in button, not on load) -------
  const dnBtn = $("#enable-desktop");
  if (dnBtn) dnBtn.addEventListener("click", async () => {
    if (!("Notification" in window)) { dnBtn.textContent = "Not supported"; return; }
    const p = await Notification.requestPermission();
    dnBtn.textContent = p === "granted" ? "Desktop alerts on" : "Blocked";
    dnBtn.disabled = true;
  });
  document.addEventListener("visibilitychange", () => {});
  window.addEventListener("clipforge_batch_done", () => {
    if (document.hidden && "Notification" in window && Notification.permission === "granted")
      new Notification("ClipForge", { body: "A batch finished — clips are ready to review." });
  });
})();
