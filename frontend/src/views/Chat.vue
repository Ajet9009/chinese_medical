<template>
  <div class="shell" :class="{ collapsed: convCollapsed }">
    <aside class="side">
      <div class="side-actions">
        <button class="btn" type="button" @click="newChat">＋ 新对话</button>
        <button class="btn ghost" type="button" @click="showFavPanel = !showFavPanel">
          {{ showFavPanel ? "⭐ 收起收藏" : "⭐ 我的收藏" }}
        </button>
      </div>
      <div v-if="showFavPanel" class="fav-panel">
        <div v-if="!favList.length" class="hint">暂无收藏，可在回答下点击「⭐ 收藏」</div>
        <div
          v-for="f in favList"
          :key="f.id"
          class="fav-item"
          @click="useFav(f)"
        >
          <span class="fav-q">{{ f.query }}</span>
          <button class="icon-btn" type="button" title="取消收藏" @click.stop="delFav(f.id)">✕</button>
        </div>
      </div>
      <div class="side-search">
        <input class="input" v-model="searchKw" placeholder="🔍 搜索对话" @input="onSearch" />
      </div>
      <div class="conv-batch" v-if="conversations.length">
        <label>
          <input type="checkbox" :checked="allConvsSelected" @change="toggleAllConvs($event.target.checked)" />
          全选
        </label>
        <span class="hint">{{ selectedConvs.size }}/{{ conversations.length }}</span>
        <button
          class="btn danger sm"
          type="button"
          :disabled="!selectedConvs.size"
          @click="batchRemoveConvs"
        >🗑️ 删选中</button>
      </div>
      <div class="conv-list">
        <div
          v-for="c in conversations"
          :key="c.id"
          class="conv-item"
          :class="{ active: c.id === currentId }"
          @click="selectConv(c.id)"
        >
          <input
            v-if="editingId === c.id"
            class="input rename-input"
            v-model="editingTitle"
            @click.stop
            @keyup.enter="saveRename(c)"
            @keyup.esc="editingId = ''"
          />
          <template v-else>
            <input
              type="checkbox"
              class="conv-check"
              :checked="selectedConvs.has(c.id)"
              @click.stop
              @change="toggleConv(c.id)"
            />
            <div class="conv-meta">
              <div class="conv-title">{{ c.title || "未命名" }}</div>
              <div class="conv-time">{{ fmtTime(c.updated_at || c.created_at) }}</div>
            </div>
            <div class="conv-ops" @click.stop>
              <button class="icon-btn" type="button" title="重命名" @click="startRename(c)">✏️</button>
              <button class="icon-btn danger" type="button" title="删除" @click="removeConv(c.id)">🗑️</button>
            </div>
          </template>
        </div>
        <div v-if="!conversations.length" class="hint empty-list">
          {{ searchKw ? "无匹配对话" : "暂无历史对话" }}
        </div>
      </div>
      <button class="collapse-btn" type="button" @click="convCollapsed = true" title="收起">‹</button>
    </aside>

    <section class="main">
      <header class="chat-head">
        <button v-if="convCollapsed" class="icon-btn menu" type="button" @click="convCollapsed = false">☰</button>
        <h2>{{ currentTitle }}</h2>
        <div class="hint">图谱检索用消解后的问句；回答保留你的原话与上文。</div>
      </header>

      <div class="msgs" ref="listEl">
        <div class="msg-batch" v-if="selectableMsgIds.length">
          <label>
            <input type="checkbox" :checked="allMsgsSelected" @change="toggleAllMsgs($event.target.checked)" />
            全选
          </label>
          <span class="hint">已选 {{ selectedMsgs.size }}/{{ selectableMsgIds.length }}</span>
          <button class="btn danger sm" type="button" :disabled="!selectedMsgs.size" @click="batchRemoveMsgs">🗑️ 删选中</button>
        </div>

        <div v-if="!messages.length" class="empty">
          <div class="seal">问</div>
          <p>从一味药、一方剂问起。可追问「它由哪些药组成？」</p>
          <div class="examples">
            <button
              v-for="ex in examples"
              :key="ex"
              class="chip"
              type="button"
              @click="send(ex)"
            >{{ ex }}</button>
          </div>
        </div>

        <div v-for="(m, i) in messages" :key="m.id || i" class="msg" :class="m.role">
          <div class="bubble" :class="m.role === 'user' ? 'user' : 'ai'">
            <template v-if="m.role === 'user'">
              <template v-if="m.editing">
                <textarea class="input edit-area" v-model="m.editText" rows="2"></textarea>
                <div class="edit-ops">
                  <button class="btn send" type="button" @click="resendEdit(m)">➤ 重提</button>
                  <button class="btn ghost" type="button" @click="m.editing = false">取消</button>
                </div>
              </template>
              <template v-else>
                <label v-if="m.id" class="msg-check" @click.stop>
                  <input type="checkbox" :checked="selectedMsgs.has(m.id)" @change="toggleMsg(m.id)" />
                </label>
                {{ m.content }}
                <button class="icon-btn edit-btn" type="button" title="编辑后重提" @click="startEdit(m)">✏️</button>
              </template>
            </template>
            <template v-else>
              <div v-if="!m.streaming" class="bubble-actions">
                <button class="text-btn" type="button" @click="regenerate(m)">🔄 重生成</button>
                <button class="text-btn" type="button" @click="copyAnswer(m)">📋 复制</button>
                <button class="text-btn" type="button" @click="exportWord(m)">📄 导出</button>
                <button class="text-btn" type="button" @click="saveFavorite(m)">⭐ 收藏</button>
              </div>
              <ul v-if="m.progress && m.progress.length" class="progress">
                <li v-for="(p, j) in m.progress" :key="j">{{ p }}</li>
              </ul>
              <div v-if="m.role === 'assistant' && !m.streaming" class="md" v-html="renderMd(m.content)"></div>
              <div v-else>{{ m.content }}<span v-if="m.streaming" class="cursor">▍</span></div>
              <div v-if="m.aborted" class="hint">已停止生成（仅显示已接收内容）</div>
              <div class="meta" v-if="!m.streaming && (m.details?.elapsed_ms || m.id)">
                <span v-if="m.details?.elapsed_ms">{{ Math.round(m.details.elapsed_ms) }} ms</span>
                <span class="fb">
                  <button class="text-btn" :class="{ on: m.details?.feedback === 'like' }" type="button" @click="setFeedback(m, 'like')" title="赞">👍</button>
                  <button class="text-btn" :class="{ on: m.details?.feedback === 'dislike' }" type="button" @click="setFeedback(m, 'dislike')" title="踩">👎</button>
                </span>
              </div>
              <details v-if="m.details && hasDetails(m.details)" class="details">
                <summary>🔍 推理过程</summary>
                <div>意图：{{ m.details.is_zhongyi_intent ? "中医" : "普通" }} · {{ m.details.intent_reason }}</div>
                <div v-if="m.details.search_question">检索问句：{{ m.details.search_question }}</div>
                <div v-if="matchedLine(m.details)">实体：{{ matchedLine(m.details) }}</div>
                <div v-if="m.details.refused">拒答：图谱与文献均无依据</div>
                <div v-if="m.details.crag_grade">
                  文献护栏：{{ m.details.crag_grade }}
                  <template v-if="m.details.crag_action"> · {{ m.details.crag_action }}</template>
                  <template v-if="m.details.crag_confidence"> · {{ m.details.crag_confidence }}</template>
                  <template v-if="m.details.citation_ok === false"> · 引用未通过</template>
                </div>
                <div v-if="(m.details.doc_chunks || []).length" class="src-head">文献</div>
                <div v-for="(d, k) in (m.details.doc_chunks || [])" :key="'d'+k">
                  {{ d.doc_name }}#{{ d.chunk_idx }}（{{ Number(d.score || 0).toFixed(2) }}）{{ d.text }}
                </div>
                <pre
                  v-for="(q, k) in (m.details.cypher_queries || [])"
                  :key="k"
                  class="cypher"
                >{{ q }}</pre>
              </details>
              <div class="related" v-if="!m.streaming && relatedOf(m).length">
                <div class="src-head">相关追问</div>
                <div class="examples">
                  <button
                    v-for="rq in relatedOf(m)"
                    :key="rq"
                    class="rq"
                    type="button"
                    @click="send(rq)"
                  >{{ rq }}</button>
                </div>
              </div>
            </template>
          </div>
        </div>
      </div>

      <div class="composer">
        <form @submit.prevent="onSubmit">
          <textarea
            v-model="draft"
            rows="2"
            :disabled="busy"
            placeholder="输入中医问题，Enter 发送，Shift+Enter 换行"
            @keydown.enter.exact.prevent="onSubmit"
          />
          <button v-if="busy" class="btn send danger" type="button" @click="stopGen">⏹ 停止</button>
          <button v-else class="btn send" type="submit" :disabled="!draft.trim()">➤ 发送</button>
        </form>
        <div v-if="error" class="err">{{ error }}</div>
      </div>
    </section>
    <div class="toast" v-if="toastMsg">{{ toastMsg }}</div>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, ref } from "vue";
import MarkdownIt from "markdown-it";
import {
  addFavorite,
  batchDeleteConversations,
  batchDeleteMessages,
  createConversation,
  deleteConversation,
  deleteFavorite,
  listConversations,
  listFavorites,
  listMessages,
  patchMessage,
  renameConversation,
  streamAsk,
} from "../api.js";

const md = new MarkdownIt({ html: false, linkify: true, breaks: true });
const examples = [
  "四君子汤有什么功效？",
  "人参的性味归经是什么？",
  "咳嗽应该吃什么中药？",
];

const conversations = ref([]);
const currentId = ref("");
const messages = ref([]);
const draft = ref("");
const busy = ref(false);
const error = ref("");
const editingId = ref("");
const editingTitle = ref("");
const listEl = ref(null);
const searchKw = ref("");
const convCollapsed = ref(false);
const showFavPanel = ref(false);
const favList = ref([]);
const selectedConvs = ref(new Set());
const selectedMsgs = ref(new Set());
const toastMsg = ref("");
let abortCtl = null;
let toastTimer = null;

const currentTitle = computed(() => {
  const found = conversations.value.find((c) => c.id === currentId.value);
  return found?.title || "新对话";
});

const allConvsSelected = computed(
  () => conversations.value.length > 0 && selectedConvs.value.size === conversations.value.length
);

const selectableMsgIds = computed(() => messages.value.filter((m) => m.id).map((m) => m.id));
const allMsgsSelected = computed(
  () => selectableMsgIds.value.length > 0 && selectedMsgs.value.size === selectableMsgIds.value.length
);

function renderMd(text) {
  return md.render(text || "");
}

function fmtTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const m = d.getMonth() + 1;
  const day = d.getDate();
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${m}/${day} ${hh}:${mm}`;
}

function toast(text) {
  toastMsg.value = text;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toastMsg.value = "";
  }, 1800);
}

function hasDetails(d) {
  if (!d) return false;
  return Boolean(
    d.intent_reason ||
      d.search_question ||
      d.refused ||
      d.crag_grade ||
      d.citation_ok === false ||
      (d.doc_chunks && d.doc_chunks.length) ||
      (d.cypher_queries && d.cypher_queries.length) ||
      matchedLine(d)
  );
}

function matchedLine(d) {
  const me = d?.matched_entities || {};
  const bits = [];
  for (const [k, arr] of Object.entries(me)) {
    if (!arr || !arr.length) continue;
    bits.push(arr.map((e) => e.name).join("、"));
  }
  return bits.filter(Boolean).join(" · ");
}

function relatedOf(m) {
  const me = m.details?.matched_entities || {};
  const qs = [];
  if (me.matched_formulas?.length) qs.push("它由哪些药组成？", "还治哪些证？");
  if (me.matched_herbs?.length) qs.push("性味归经是什么？", "有哪些配伍？");
  if (me.matched_symptoms?.length) qs.push("常用哪些方剂？");
  return [...new Set(qs)].slice(0, 3);
}

function mapMsg(m) {
  return {
    id: m.id,
    role: m.role,
    content: m.content,
    details: m.details || null,
    streaming: false,
    progress: [],
    editing: false,
    editText: "",
    aborted: false,
  };
}

async function refreshConvs() {
  conversations.value = await listConversations(searchKw.value);
}

async function onSearch() {
  selectedConvs.value = new Set();
  await refreshConvs();
}

async function refreshFavs() {
  favList.value = await listFavorites();
}

async function newChat() {
  const c = await createConversation("新对话");
  currentId.value = c.id;
  messages.value = [];
  selectedMsgs.value = new Set();
  await refreshConvs();
}

async function selectConv(id) {
  currentId.value = id;
  selectedMsgs.value = new Set();
  const rows = await listMessages(id);
  messages.value = rows.map(mapMsg);
  await scrollBottom();
}

function startRename(c) {
  editingId.value = c.id;
  editingTitle.value = c.title;
}

async function saveRename(c) {
  const title = editingTitle.value.trim();
  if (title) await renameConversation(c.id, title);
  editingId.value = "";
  await refreshConvs();
}

async function removeConv(id) {
  await deleteConversation(id);
  const next = new Set(selectedConvs.value);
  next.delete(id);
  selectedConvs.value = next;
  if (currentId.value === id) {
    currentId.value = "";
    messages.value = [];
  }
  await refreshConvs();
}

function toggleConv(id) {
  const next = new Set(selectedConvs.value);
  if (next.has(id)) next.delete(id);
  else next.add(id);
  selectedConvs.value = next;
}

function toggleAllConvs(on) {
  selectedConvs.value = on ? new Set(conversations.value.map((c) => c.id)) : new Set();
}

async function batchRemoveConvs() {
  const ids = [...selectedConvs.value];
  if (!ids.length) return;
  await batchDeleteConversations(ids);
  selectedConvs.value = new Set();
  if (ids.includes(currentId.value)) {
    currentId.value = "";
    messages.value = [];
  }
  await refreshConvs();
}

function toggleMsg(id) {
  const next = new Set(selectedMsgs.value);
  if (next.has(id)) next.delete(id);
  else next.add(id);
  selectedMsgs.value = next;
}

function toggleAllMsgs(on) {
  selectedMsgs.value = on ? new Set(selectableMsgIds.value) : new Set();
}

async function batchRemoveMsgs() {
  const ids = [...selectedMsgs.value];
  if (!ids.length || !currentId.value) return;
  await batchDeleteMessages(currentId.value, ids);
  selectedMsgs.value = new Set();
  await selectConv(currentId.value);
}

function startEdit(m) {
  m.editing = true;
  m.editText = m.content;
}

async function resendEdit(m) {
  const text = (m.editText || "").trim();
  if (!text) return;
  const idx = messages.value.indexOf(m);
  if (m.id && currentId.value) {
    await patchMessage(currentId.value, m.id, { content: text });
  }
  m.content = text;
  m.editing = false;
  messages.value = messages.value.slice(0, idx + 1);
  await send(text, { omitUser: true });
}

async function regenerate(aiMsg) {
  const idx = messages.value.indexOf(aiMsg);
  const user = [...messages.value.slice(0, idx)].reverse().find((x) => x.role === "user");
  if (!user) return;
  if (aiMsg.id && currentId.value) {
    await batchDeleteMessages(currentId.value, [aiMsg.id]);
  }
  messages.value.splice(idx, 1);
  await send(user.content, { omitUser: true });
}

async function copyAnswer(m) {
  await navigator.clipboard.writeText(m.content || "");
  toast("已复制");
}

function exportWord(m) {
  const html = `<html><head><meta charset="utf-8"></head><body>${md.render(m.content || "")}</body></html>`;
  const blob = new Blob(["\ufeff", html], { type: "application/msword" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "草本通回答.doc";
  a.click();
  URL.revokeObjectURL(a.href);
}

async function saveFavorite(m) {
  const user = [...messages.value].reverse().find((x) => x.role === "user");
  await addFavorite(user?.content || currentTitle.value, m.content || "");
  await refreshFavs();
  toast("已收藏");
}

async function delFav(id) {
  await deleteFavorite(id);
  await refreshFavs();
}

function useFav(f) {
  showFavPanel.value = false;
  send(f.query);
}

async function setFeedback(m, kind) {
  if (!m.id || !currentId.value) return;
  const next = m.details?.feedback === kind ? null : kind;
  if (!next) return;
  const updated = await patchMessage(currentId.value, m.id, { feedback: next });
  m.details = updated.details;
}

async function scrollBottom() {
  await nextTick();
  const el = listEl.value;
  if (el) el.scrollTop = el.scrollHeight;
}

function stopGen() {
  abortCtl?.abort();
}

async function send(text, opts = {}) {
  const question = (text || "").trim();
  if (!question || busy.value) return;
  error.value = "";
  busy.value = true;
  draft.value = "";
  if (!opts.omitUser) {
    messages.value.push({
      role: "user",
      content: question,
      details: null,
      streaming: false,
      progress: [],
    });
  }
  const ai = {
    role: "assistant",
    content: "",
    details: null,
    streaming: true,
    progress: [],
    aborted: false,
  };
  messages.value.push(ai);
  await scrollBottom();
  abortCtl = new AbortController();
  try {
    await streamAsk(
      question,
      currentId.value || undefined,
      (event, data) => {
        if (event === "session" && data.conversation_id) {
          currentId.value = data.conversation_id;
        } else if (event === "progress") {
          const mark = data.status === "done" ? "成" : "…";
          ai.progress.push(`${mark} ${data.label}`);
        } else if (event === "token") {
          ai.content += data.text || "";
        } else if (event === "done") {
          ai.streaming = false;
          ai.content = data.answer || ai.content;
          ai.details = {
            is_zhongyi_intent: data.is_zhongyi_intent,
            intent_reason: data.intent_reason,
            elapsed_ms: data.elapsed_ms,
            matched_entities: data.matched_entities,
            cypher_queries: data.cypher_queries,
            search_question: data.search_question,
            doc_chunks: data.doc_chunks || [],
            refused: Boolean(data.refused),
            crag_grade: data.crag_grade || "",
            crag_action: data.crag_action || "",
            crag_confidence: data.crag_confidence || "",
            citation_ok: data.citation_ok !== false,
          };
        } else if (event === "error") {
          ai.streaming = false;
          ai.content = data.error || "出错了";
        } else if (event === "abort") {
          ai.streaming = false;
          ai.aborted = true;
        }
        scrollBottom();
      },
      { signal: abortCtl.signal, omitUserMessage: Boolean(opts.omitUser) }
    );
    await refreshConvs();
    if (!ai.aborted && currentId.value) {
      const rows = await listMessages(currentId.value);
      messages.value = rows.map(mapMsg);
    }
  } catch (e) {
    if (abortCtl?.signal?.aborted) {
      ai.streaming = false;
      ai.aborted = true;
    } else {
      error.value = e.message || String(e);
      ai.streaming = false;
      if (!ai.content) ai.content = error.value;
    }
  } finally {
    busy.value = false;
    ai.streaming = false;
    abortCtl = null;
    await scrollBottom();
  }
}

function onSubmit() {
  send(draft.value);
}

onMounted(async () => {
  try {
    await refreshConvs();
    await refreshFavs();
    if (conversations.value[0]) await selectConv(conversations.value[0].id);
  } catch (e) {
    error.value = "无法连接 API（http://localhost:8000）。请先启动 python -m _005_fastapi.main";
  }
});
</script>
