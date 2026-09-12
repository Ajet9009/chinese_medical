const BASE = "/api";
const TOKEN_KEY = "cm_token";

function authHeaders() {
  const token = localStorage.getItem(TOKEN_KEY);
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function goLogin() {
  if (window.location.pathname === "/login") return;
  window.location.href = "/login";
}

async function json(path, options = {}) {
  const { headers: extraHeaders, ...rest } = options;
  const res = await fetch(BASE + path, {
    ...rest,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
      ...(extraHeaders || {}),
    },
  });
  if (res.status === 401 && path !== "/auth/login") {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem("cm_user");
    goLogin();
  }
  if (res.status === 204) return null;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data.detail || res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

export function login(username, password) {
  return json("/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export function me() {
  return json("/auth/me");
}

export function listConversations(keyword = "") {
  const q = keyword ? `?keyword=${encodeURIComponent(keyword)}` : "";
  return json("/conversations" + q);
}

export function createConversation(title = "") {
  return json("/conversations", { method: "POST", body: JSON.stringify({ title }) });
}

export function renameConversation(id, title) {
  return json(`/conversations/${id}`, { method: "PATCH", body: JSON.stringify({ title }) });
}

export function deleteConversation(id) {
  return json(`/conversations/${id}`, { method: "DELETE" });
}

export function batchDeleteConversations(ids) {
  return json("/conversations/batch-delete", { method: "POST", body: JSON.stringify({ ids }) });
}

export function listMessages(id) {
  return json(`/conversations/${id}/messages`);
}

export function batchDeleteMessages(conversationId, ids) {
  return json(`/conversations/${conversationId}/messages/batch-delete`, {
    method: "POST",
    body: JSON.stringify({ ids }),
  });
}

export function patchMessage(conversationId, messageId, body) {
  return json(`/conversations/${conversationId}/messages/${messageId}`, {
    method: "PATCH",
    body: JSON.stringify(body),
  });
}

export function listFavorites(keyword = "") {
  const q = keyword ? `?keyword=${encodeURIComponent(keyword)}` : "";
  return json("/favorites" + q);
}

export function addFavorite(query, answer) {
  return json("/favorites", { method: "POST", body: JSON.stringify({ query, answer }) });
}

export function deleteFavorite(id) {
  return json(`/favorites/${id}`, { method: "DELETE" });
}

export function adminFeedbacks(feedback = "") {
  const q = feedback ? `?feedback=${encodeURIComponent(feedback)}` : "";
  return json("/admin/feedbacks" + q);
}

export function adminMarkGolden(messageId) {
  return json(`/admin/feedbacks/${messageId}/golden`, { method: "POST", body: "{}" });
}

export function adminGolden() {
  return json("/admin/golden");
}

export function adminPrompts() {
  return json("/admin/prompts");
}

export function adminImportStatus() {
  return json("/admin/import-status");
}

export function adminIngestDocs() {
  return json("/admin/docs/ingest", { method: "POST", body: "{}" });
}

export async function adminUploadDoc(file) {
  const res = await fetch(BASE + "/admin/docs/upload", {
    method: "POST",
    headers: authHeaders(),
    body: (() => {
      const fd = new FormData();
      fd.append("file", file);
      return fd;
    })(),
  });
  if (res.status === 401) {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem("cm_user");
    goLogin();
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data.detail || res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

export function adminUsers() {
  return json("/admin/users");
}

export function adminCreateUser({ username, password, role, dept }) {
  return json("/admin/users", {
    method: "POST",
    body: JSON.stringify({ username, password, role, dept }),
  });
}

export function adminPatchUser(id, body) {
  return json(`/admin/users/${id}`, { method: "PATCH", body: JSON.stringify(body) });
}

export function patchMe(dept) {
  return json("/auth/me", { method: "PATCH", body: JSON.stringify({ dept }) });
}

export function listDocs(page = 1, size = 20, keyword = "") {
  const q = new URLSearchParams({ page, size, keyword });
  return json("/document/list?" + q.toString());
}

export function getDocStats() {
  return json("/document/stats");
}

export function uploadDocs(form, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", BASE + "/document/upload");
    const token = localStorage.getItem(TOKEN_KEY);
    if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    if (onProgress) xhr.upload.onprogress = (e) => onProgress(e);
    xhr.onload = () => {
      if (xhr.status === 401) {
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem("cm_user");
        goLogin();
        reject(new Error("未登录"));
        return;
      }
      let data = {};
      try {
        data = JSON.parse(xhr.responseText || "{}");
      } catch {
        data = {};
      }
      if (xhr.status < 200 || xhr.status >= 300) {
        const detail = data.detail || xhr.statusText;
        reject(new Error(typeof detail === "string" ? detail : JSON.stringify(detail)));
        return;
      }
      resolve(data);
    };
    xhr.onerror = () => reject(new Error("上传失败"));
    xhr.send(form);
  });
}

export function parseDocs(ids) {
  return json("/document/parse", { method: "POST", body: JSON.stringify({ ids }) });
}

export function vectorize(docId) {
  return json("/document/vector/generate", { method: "POST", body: JSON.stringify({ docId }) });
}

export function vectorizeBatch(ids) {
  return json("/document/vector/batch", { method: "POST", body: JSON.stringify({ ids }) });
}

export function deleteDocs(ids) {
  return json("/document/batch-delete", { method: "POST", body: JSON.stringify({ ids }) });
}

export function listDocVersions(docId) {
  return json(`/document/${docId}/versions`);
}

export function rollbackDoc(docId, version) {
  return json("/document/rollback", { method: "POST", body: JSON.stringify({ docId, version }) });
}

export function govDocuments(page = 1, size = 20, keyword = "") {
  const q = new URLSearchParams({ page, size, keyword });
  return json("/knowledge-governance/documents?" + q.toString());
}

export function govProfile(docId) {
  return json(`/knowledge-governance/documents/${docId}/profile`);
}

export function saveGovProfile(docId, body) {
  return json(`/knowledge-governance/documents/${docId}/profile`, {
    method: "PUT",
    body: JSON.stringify(body),
  });
}

export function govScan(docIds = []) {
  return json("/knowledge-governance/scan", { method: "POST", body: JSON.stringify({ docIds }) });
}

export function govIssues(params = {}) {
  const q = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v) q.set(k, v);
  });
  return json("/knowledge-governance/issues?" + q.toString());
}

export function govIssueStats() {
  return json("/knowledge-governance/issues/stats");
}

export function govReview(id, status, note = "") {
  return json(`/knowledge-governance/issues/${id}/review`, {
    method: "POST",
    body: JSON.stringify({ status, note }),
  });
}

export function changePassword(oldPassword, newPassword) {
  return json("/auth/password", {
    method: "POST",
    body: JSON.stringify({ old_password: oldPassword, new_password: newPassword }),
  });
}

export function adminDeleteUser(id) {
  return json(`/admin/users/${id}`, { method: "DELETE" });
}

export function adminResetPassword(id, password) {
  return json(`/admin/users/${id}/reset-password`, {
    method: "POST",
    body: JSON.stringify({ password }),
  });
}

export function adminLogs(limit = 50, offset = 0) {
  return json(`/admin/logs?limit=${limit}&offset=${offset}`);
}

export async function streamAsk(question, conversationId, onEvent, options = {}) {
  const { signal, omitUserMessage } = options;
  const res = await fetch(BASE + "/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    signal,
    body: JSON.stringify({
      question,
      conversation_id: conversationId || undefined,
      omit_user_message: Boolean(omitUserMessage),
    }),
  });
  if (res.status === 401) {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem("cm_user");
    goLogin();
  }
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "请求失败");
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let eventName = "message";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split("\n");
      buf = parts.pop() || "";
      for (const line of parts) {
        if (line.startsWith("event:")) {
          eventName = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          const raw = line.slice(5).trim();
          let data = {};
          try {
            data = JSON.parse(raw);
          } catch {
            data = { raw };
          }
          onEvent(eventName, data);
          eventName = "message";
        }
      }
    }
  } catch (err) {
    if (signal?.aborted) {
      onEvent("abort", {});
      return;
    }
    throw err;
  }
}
