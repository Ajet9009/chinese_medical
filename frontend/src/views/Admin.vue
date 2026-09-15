<template>
  <div class="page admin-page">
    <div class="page-inner">
    <header class="page-head">
      <div>
        <h2>管理</h2>
        <p class="hint">反馈可写入黄金集。仅 user / admin 两角色，不开放自行注册。</p>
      </div>
    </header>
    <div class="tabs">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        class="tab"
        :class="{ on: current === tab.id }"
        type="button"
        @click="current = tab.id"
      >{{ tab.label }}</button>
    </div>

    <section v-if="current === 'feedback'" class="card">
      <div class="admin-toolbar">
        <select class="select" v-model="fbFilter" @change="loadFeedbacks">
          <option value="">全部反馈</option>
          <option value="like">👍 赞</option>
          <option value="dislike">👎 踩</option>
        </select>
        <span class="hint">共 {{ feedbacks.total || 0 }} 条</span>
        <span v-if="goldenHint" class="hint">{{ goldenHint }}</span>
      </div>
      <table class="tbl">
        <thead>
          <tr>
            <th>用户</th>
            <th>问句</th>
            <th>回答</th>
            <th>反馈</th>
            <th>证据</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="item in feedbacks.items || []" :key="item.message_id">
            <td>{{ item.username || "—" }}</td>
            <td>{{ item.question }}</td>
            <td class="clip">{{ item.answer }}</td>
            <td>
              <span class="badge" :class="item.feedback === 'like' ? 'badge-success' : 'badge-danger'">
                {{ item.feedback === "like" ? "👍 赞" : "👎 踩" }}
              </span>
            </td>
            <td>
              <span v-if="item.evidence_gap" class="badge badge-warning">证据缺口</span>
              <span v-else class="hint">—</span>
            </td>
            <td>
              <button
                class="btn ghost sm"
                type="button"
                :disabled="item.in_golden || goldenBusy === item.message_id"
                @click="markGolden(item)"
              >{{ item.in_golden ? "已在黄金集" : (goldenBusy === item.message_id ? "写入中…" : "标为黄金集") }}</button>
            </td>
          </tr>
          <tr v-if="!(feedbacks.items || []).length">
            <td colspan="6" class="hint">暂无点赞或点踩。踩一条后可标入黄金集待标注。</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section v-else-if="current === 'prompts'" class="card">
      <p class="hint" style="margin-top:0">
        Langfuse：{{ prompts.langfuse_enabled ? "已启用" : "未启用" }}。本页只读，改模板去 Langfuse → Prompts（约 60 秒后生效）。
      </p>
      <table class="tbl">
        <thead>
          <tr>
            <th>节点</th>
            <th>名称</th>
            <th>来源</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="p in prompts.items || []" :key="p.name">
            <td>{{ p.label }}</td>
            <td><code>{{ p.name }}</code></td>
            <td>
              <span class="badge" :class="p.source === 'langfuse' ? 'badge-info' : 'badge-neutral'">
                {{ p.source }}
              </span>
            </td>
          </tr>
        </tbody>
      </table>
    </section>

    <section v-else-if="current === 'import'" class="stat-grid">
      <article
        class="stat stat-accent stat-zoomable"
        role="button"
        tabindex="0"
        title="点击放大"
        @click="openZoom('redis')"
        @keydown.enter.prevent="openZoom('redis')"
      >
        <h3>Redis</h3>
        <p class="stat-val">{{ importStatus.redis || "—" }}</p>
        <p class="hint">会话热缓存通不通</p>
      </article>
      <article
        class="stat stat-accent stat-zoomable"
        role="button"
        tabindex="0"
        title="点击放大"
        @click="openZoom('neo4j')"
        @keydown.enter.prevent="openZoom('neo4j')"
      >
        <h3>Neo4j</h3>
        <p class="stat-val">{{ importStatus.neo4j?.ok ? "连通" : "未连通" }}</p>
        <p v-if="importStatus.neo4j?.error" class="err">{{ importStatus.neo4j.error }}</p>
        <ul class="hint">
          <li v-for="(n, label) in importStatus.neo4j?.labels || {}" :key="label">
            {{ label }}：{{ n }}
          </li>
        </ul>
      </article>
      <article
        class="stat stat-accent stat-zoomable"
        role="button"
        tabindex="0"
        title="点击放大"
        @click="openZoom('faiss')"
        @keydown.enter.prevent="openZoom('faiss')"
      >
        <h3>FAISS</h3>
        <p class="stat-val">{{ importStatus.faiss?.index_exists ? "索引已找到" : "索引缺失" }}</p>
        <p class="hint clip">{{ importStatus.faiss?.index_path }}</p>
        <p>{{ importStatus.faiss?.metadata_exists ? "元数据已找到" : "元数据缺失" }}</p>
        <p class="hint clip">{{ importStatus.faiss?.metadata_path }}</p>
      </article>
      <article
        class="stat stat-accent stat-zoomable"
        role="button"
        tabindex="0"
        title="点击放大"
        @click="openZoom('docs')"
        @keydown.enter.prevent="openZoom('docs')"
      >
        <h3>文献 RAG</h3>
        <p class="stat-val">{{ importStatus.docs?.index_exists ? "索引已找到" : "索引缺失" }}</p>
        <p class="hint">{{ importStatus.docs?.enabled ? "已启用" : "已关闭" }} · {{ importStatus.docs?.chunk_count || 0 }} 块</p>
        <p class="hint clip">{{ importStatus.docs?.source_dir }}</p>
        <router-link class="btn sm" to="/documents" @click.stop>去知识库上传</router-link>
        <button class="btn sm" type="button" :disabled="ingestBusy" @click.stop="ingestDocs">
          {{ ingestBusy ? "重建中…" : "📚 重建文献索引" }}
        </button>
        <p v-if="ingestMsg" class="hint">{{ ingestMsg }}</p>
      </article>
      <article
        class="stat stat-accent stat-zoomable"
        role="button"
        tabindex="0"
        title="点击放大"
        @click="openZoom('degraded')"
        @keydown.enter.prevent="openZoom('degraded')"
      >
        <h3>降级打点</h3>
        <p class="stat-val">{{ importStatus.degraded?.total || 0 }}</p>
        <p class="hint">Redis / 索引失败会计数，不抛给用户</p>
        <ul class="hint">
          <li v-for="(n, tag) in importStatus.degraded?.byTag || {}" :key="tag">
            {{ tag }}：{{ n }}
          </li>
        </ul>
        <p v-if="!(importStatus.degraded?.total)" class="hint">暂无降级</p>
      </article>
      <article
        class="stat stat-accent stat-zoomable"
        role="button"
        tabindex="0"
        title="点击放大"
        @click="openZoom('golden')"
        @keydown.enter.prevent="openZoom('golden')"
      >
        <h3>黄金集</h3>
        <p class="stat-val">{{ importStatus.golden?.total || 0 }}</p>
        <p class="hint">已标注 {{ importStatus.golden?.labeled || 0 }} · 待标注 {{ importStatus.golden?.pending || 0 }}</p>
        <p class="hint clip">{{ importStatus.golden?.path }}</p>
      </article>
    </section>

    <section v-else-if="current === 'eval'" class="card">
      <p class="hint" style="margin-top:0">
        黄金集：离线关键词全命中，CI <code>python scripts/eval_golden.py</code>。
        文献 RAG：嵌入生成评测集后写出 RAGAS dataset（五列 JSON）并打分，CI <code>python scripts/eval_ragas.py</code>（默认不调 LLM）。
      </p>
      <p class="hint">
        共 {{ golden.total || 0 }} 条 · 已标注 {{ golden.labeled || 0 }} · 待标注 {{ golden.pending || 0 }}
      </p>
      <table class="tbl">
        <thead>
          <tr>
            <th>问句</th>
            <th>类别</th>
            <th>来源</th>
            <th>期望关键词</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="item in golden.items || []" :key="item.query">
            <td>{{ item.query }}</td>
            <td><span class="badge badge-info">{{ item.category }}</span></td>
            <td>
              <span class="badge" :class="item.source === 'feedback' ? 'badge-warning' : 'badge-neutral'">
                {{ item.source === "feedback" ? "反馈" : "种子" }}
              </span>
            </td>
            <td class="clip">{{ (item.expect || []).join("、") || "待标注" }}</td>
          </tr>
          <tr v-if="!(golden.items || []).length">
            <td colspan="4" class="hint">黄金集为空</td>
          </tr>
        </tbody>
      </table>
      <h3 style="margin-top:1.5rem">RAGAS 文献 RAG</h3>
      <p class="hint">
        评测集 {{ ragas.testset?.n_items || 0 }} 条
        · dataset {{ ragas.dataset?.n_items || 0 }} 条
        （{{ (ragas.dataset?.columns || []).join(" / ") || "user_input / retrieved_contexts / response / reference / reference_contexts" }}）
        <span v-if="ragas.report">
          · 最近 {{ ragas.report.n_samples || 0 }} 条 · {{ ragas.report.answer_mode || "extractive" }}
        </span>
        <span v-else> · 尚无报告，本机运行脚本后刷新</span>
      </p>
      <table class="tbl">
        <thead>
          <tr>
            <th>指标 English（中文）</th>
            <th>含义</th>
            <th>得分</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in ragasRows" :key="row.id">
            <td>{{ row.name }}</td>
            <td class="clip">{{ row.why }}</td>
            <td>{{ row.score == null ? "—" : Number(row.score).toFixed(3) }}</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section v-else-if="current === 'logs'" class="card">
      <form class="admin-toolbar log-filters" @submit.prevent="loadLogs">
        <label class="hint">
          从
          <input class="input" type="datetime-local" v-model="logStart" />
        </label>
        <label class="hint">
          到
          <input class="input" type="datetime-local" v-model="logEnd" />
        </label>
        <input
          class="input"
          v-model="logUser"
          list="log-users"
          placeholder="用户名"
        />
        <datalist id="log-users">
          <option v-for="u in users" :key="u.id" :value="u.username" />
        </datalist>
        <select class="select" v-model="logType">
          <option value="">全部类型</option>
          <option v-for="t in logTypeOptions" :key="t" :value="t">{{ t }}</option>
        </select>
        <button class="btn sm" type="submit">🔍 查询</button>
        <button class="btn ghost sm" type="button" @click="resetLogFilters">重置</button>
        <span class="hint">共 {{ logs.total || 0 }} 条</span>
      </form>
      <table class="tbl">
        <thead>
          <tr>
            <th>时间</th>
            <th>用户</th>
            <th>类型</th>
            <th>内容</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="item in logs.items || []" :key="item.id">
            <td>{{ formatTime(item.created_at) }}</td>
            <td>{{ item.username }}</td>
            <td><span class="badge badge-info">{{ item.operate_type }}</span></td>
            <td class="clip">{{ item.content }}</td>
          </tr>
          <tr v-if="!(logs.items || []).length">
            <td colspan="4" class="hint">暂无匹配日志</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section v-else-if="current === 'users'" class="card">
      <form class="user-form" @submit.prevent="createUser">
        <input class="input" v-model="newUser.username" placeholder="用户名" />
        <input class="input" v-model="newUser.password" type="password" placeholder="密码至少 6 位" />
        <input class="input" v-model="newUser.dept" placeholder="科室（可空）" />
        <select class="select" v-model="newUser.role">
          <option value="user">user</option>
          <option value="admin">admin</option>
        </select>
        <button class="btn" type="submit">＋ 新建</button>
      </form>
      <p v-if="userErr" class="err">{{ userErr }}</p>
      <table class="tbl">
        <thead>
          <tr>
            <th>用户名</th>
            <th>角色</th>
            <th>科室</th>
            <th>状态</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="u in users" :key="u.id">
            <td>{{ u.username }}</td>
            <td>
              <select class="select" :value="u.role" @change="changeRole(u, $event.target.value)">
                <option value="user">user</option>
                <option value="admin">admin</option>
              </select>
            </td>
            <td>
              <input
                class="input"
                :value="u.dept || ''"
                placeholder="公开"
                style="min-width:96px"
                @change="changeDept(u, $event.target.value)"
              />
            </td>
            <td>
              <span class="badge" :class="u.status === 'active' ? 'badge-success' : 'badge-neutral'">
                {{ u.status === "active" ? "可用" : "禁用" }}
              </span>
            </td>
            <td>
              <div class="tbl-ops">
                <button
                  v-if="u.id !== meId"
                  class="btn ghost sm"
                  type="button"
                  @click="toggleUser(u)"
                >
                  {{ u.status === "active" ? "⏸ 禁用" : "▶ 启用" }}
                </button>
                <button class="btn ghost sm" type="button" @click="resetPwd(u)">🔑 重置密码</button>
                <button
                  v-if="u.id !== meId"
                  class="btn ghost sm"
                  type="button"
                  @click="removeUser(u)"
                >🗑️ 删除</button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </section>
    <div v-if="zoomId" class="modal-bg stat-zoom-bg" @click.self="closeZoom">
      <div class="modal stat-zoom-modal" role="dialog" aria-modal="true">
        <div class="modal-head">
          <span>{{ zoomTitles[zoomId] }}</span>
          <button class="icon-btn" type="button" aria-label="关闭" @click="closeZoom">✕</button>
        </div>
        <div class="stat-zoom-body">
          <template v-if="zoomId === 'redis'">
            <p class="stat-val">{{ importStatus.redis || "—" }}</p>
            <p class="hint">会话热缓存。ok 才能按条数裁最近对话；挂了改走 SQLite，问答不中断。</p>
          </template>
          <template v-else-if="zoomId === 'neo4j'">
            <p class="stat-val">{{ importStatus.neo4j?.ok ? "连通" : "未连通" }}</p>
            <p v-if="importStatus.neo4j?.error" class="err">{{ importStatus.neo4j.error }}</p>
            <p class="hint">图节点库存。数字是数量，不是正确率。</p>
            <ul class="hint stat-zoom-list">
              <li v-for="(n, label) in importStatus.neo4j?.labels || {}" :key="label">
                {{ label }}：{{ n }}
              </li>
            </ul>
          </template>
          <template v-else-if="zoomId === 'faiss'">
            <p class="stat-val">{{ importStatus.faiss?.index_exists ? "索引已找到" : "索引缺失" }}</p>
            <p class="hint">实体标准化索引（口语对齐到图谱标准名），与文献 RAG 不是同一套文件。</p>
            <p class="hint">索引 {{ importStatus.faiss?.index_exists ? "已找到" : "缺失" }}</p>
            <p>{{ importStatus.faiss?.index_path || "—" }}</p>
            <p class="hint">元数据 {{ importStatus.faiss?.metadata_exists ? "已找到" : "缺失" }}</p>
            <p>{{ importStatus.faiss?.metadata_path || "—" }}</p>
          </template>
          <template v-else-if="zoomId === 'docs'">
            <p class="stat-val">{{ importStatus.docs?.index_exists ? "索引已找到" : "索引缺失" }}</p>
            <p class="hint">{{ importStatus.docs?.enabled ? "已启用" : "已关闭" }} · {{ importStatus.docs?.chunk_count || 0 }} 块</p>
            <p>{{ importStatus.docs?.source_dir || "—" }}</p>
            <div class="stat-zoom-ops">
              <router-link class="btn sm" to="/documents" @click="closeZoom">去知识库上传</router-link>
              <button class="btn sm" type="button" :disabled="ingestBusy" @click="ingestDocs">
                {{ ingestBusy ? "重建中…" : "📚 重建文献索引" }}
              </button>
            </div>
            <p v-if="ingestMsg" class="hint">{{ ingestMsg }}</p>
          </template>
          <template v-else-if="zoomId === 'degraded'">
            <p class="stat-val">{{ importStatus.degraded?.total || 0 }}</p>
            <p class="hint">备胎路径被踩的次数。重启进程清零。不替代 Langfuse Trace。</p>
            <ul class="hint stat-zoom-list">
              <li v-for="(n, tag) in importStatus.degraded?.byTag || {}" :key="tag">
                {{ tag }}：{{ n }}
              </li>
            </ul>
            <p v-if="!(importStatus.degraded?.total)" class="hint">暂无降级</p>
          </template>
          <template v-else-if="zoomId === 'golden'">
            <p class="stat-val">{{ importStatus.golden?.total || 0 }}</p>
            <p class="hint">已标注 {{ importStatus.golden?.labeled || 0 }} · 待标注 {{ importStatus.golden?.pending || 0 }}</p>
            <p>{{ importStatus.golden?.path || "—" }}</p>
          </template>
        </div>
      </div>
    </div>
    <div v-if="error" class="err">{{ error }}</div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import {
  adminCreateUser,
  adminDeleteUser,
  adminFeedbacks,
  adminGolden,
  adminImportStatus,
  adminRagas_step_01,
  adminIngestDocs,
  adminLogs,
  adminMarkGolden,
  adminPatchUser,
  adminPrompts,
  adminResetPassword,
  adminUsers,
  me,
} from "../api.js";
import { useAuthStore } from "../stores/auth.js";

const auth = useAuthStore();
const tabs = [
  { id: "feedback", label: "👍 反馈" },
  { id: "prompts", label: "📝 Prompt" },
  { id: "import", label: "💚 系统运行状态" },
  { id: "eval", label: "🧪 评测" },
  { id: "users", label: "👥 用户" },
  { id: "logs", label: "📋 操作日志" },
];
const current = ref("feedback");
const error = ref("");
const feedbacks = ref({ total: 0, items: [] });
const fbFilter = ref("");
const prompts = ref({ langfuse_enabled: false, items: [] });
const importStatus = ref({});
const ingestBusy = ref(false);
const ingestMsg = ref("");
const golden = ref({ total: 0, labeled: 0, pending: 0, items: [] });
const ragas = ref({ metrics: [], testset: {}, dataset: { columns: [] }, report: null });
const ragasRows = computed(() => {
  const scores = Object.fromEntries(
    (ragas.value.report?.metrics || []).map((m) => [m.id, m.score]),
  );
  return (ragas.value.metrics || []).map((m) => ({
    ...m,
    score: scores[m.id],
  }));
});
const goldenBusy = ref("");
const goldenHint = ref("");
const users = ref([]);
const logs = ref({ total: 0, items: [], operate_types: [] });
const logStart = ref("");
const logEnd = ref("");
const logUser = ref("");
const logType = ref("");
const LOG_TYPE_FALLBACK = [
  "登录",
  "改资料",
  "改密码",
  "用户管理",
  "文档上传",
  "文档回滚",
  "文档解析",
  "文档删除",
  "文档授权",
  "知识治理",
];
const logTypeOptions = computed(() => {
  const fromApi = logs.value.operate_types || [];
  return [...new Set([...fromApi, ...LOG_TYPE_FALLBACK])];
});
const newUser = ref({ username: "", password: "", role: "user", dept: "" });
const userErr = ref("");
const meId = ref(auth.userId || "");
const zoomId = ref("");
const zoomTitles = {
  redis: "Redis",
  neo4j: "Neo4j",
  faiss: "FAISS",
  docs: "文献 RAG",
  degraded: "降级打点",
  golden: "黄金集",
};

function openZoom(id) {
  zoomId.value = id;
}

function closeZoom() {
  zoomId.value = "";
}

function onZoomKey(e) {
  if (e.key === "Escape") closeZoom();
}

function formatTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}

async function loadFeedbacks() {
  feedbacks.value = await adminFeedbacks(fbFilter.value);
}

async function loadUsers() {
  users.value = await adminUsers();
}

function localToIso(value, asEnd = false) {
  if (!value) return "";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return "";
  if (asEnd && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(value)) {
    d.setSeconds(59, 999);
  }
  return d.toISOString();
}

async function loadLogs() {
  logs.value = await adminLogs({
    username: logUser.value.trim(),
    operateType: logType.value,
    start: localToIso(logStart.value),
    end: localToIso(logEnd.value, true),
  });
}

async function resetLogFilters() {
  logStart.value = "";
  logEnd.value = "";
  logUser.value = "";
  logType.value = "";
  await loadLogs();
}

async function loadGolden() {
  golden.value = await adminGolden();
}

async function loadRagas_step_01() {
  ragas.value = await adminRagas_step_01();
}

async function markGolden(item) {
  goldenHint.value = "";
  goldenBusy.value = item.message_id;
  try {
    const data = await adminMarkGolden(item.message_id);
    goldenHint.value = data.added
      ? `已写入黄金集（共 ${data.total} 条）`
      : (data.reason || "已在黄金集");
    await loadFeedbacks();
    await loadGolden();
    if (importStatus.value.golden) {
      importStatus.value = {
        ...importStatus.value,
        golden: {
          ...importStatus.value.golden,
          total: data.total,
          pending: data.added
            ? (importStatus.value.golden.pending || 0) + 1
            : importStatus.value.golden.pending,
        },
      };
    }
  } catch (e) {
    goldenHint.value = e.message || String(e);
  } finally {
    goldenBusy.value = "";
  }
}

async function loadAll() {
  error.value = "";
  try {
    const info = await me();
    meId.value = info.id;
    auth.setSession({
      token: auth.token,
      id: info.id,
      username: info.username,
      role: info.role,
      dept: info.dept || "",
    });
    await loadFeedbacks();
    prompts.value = await adminPrompts();
    importStatus.value = await adminImportStatus();
    await loadGolden();
    await loadRagas_step_01();
    await loadUsers();
    await loadLogs();
  } catch (e) {
    error.value = e.message || String(e);
  }
}

async function ingestDocs() {
  ingestMsg.value = "";
  ingestBusy.value = true;
  try {
    const data = await adminIngestDocs();
    importStatus.value = { ...importStatus.value, docs: data };
    ingestMsg.value = `已写入 ${data.chunks} 块`;
  } catch (e) {
    ingestMsg.value = e.message || String(e);
  } finally {
    ingestBusy.value = false;
  }
}

async function createUser() {
  userErr.value = "";
  try {
    await adminCreateUser(newUser.value);
    newUser.value = { username: "", password: "", role: "user", dept: "" };
    await loadUsers();
  } catch (e) {
    userErr.value = e.message || String(e);
  }
}

async function changeDept(u, dept) {
  userErr.value = "";
  try {
    await adminPatchUser(u.id, { dept });
    await loadUsers();
  } catch (e) {
    userErr.value = e.message || String(e);
    await loadUsers();
  }
}

async function toggleUser(u) {
  userErr.value = "";
  const status = u.status === "active" ? "inactive" : "active";
  try {
    await adminPatchUser(u.id, { status });
    await loadUsers();
  } catch (e) {
    userErr.value = e.message || String(e);
  }
}

async function changeRole(u, role) {
  userErr.value = "";
  try {
    await adminPatchUser(u.id, { role });
    await loadUsers();
  } catch (e) {
    userErr.value = e.message || String(e);
    await loadUsers();
  }
}

async function resetPwd(u) {
  const pwd = window.prompt(`重置 ${u.username} 的密码（至少 6 位）`);
  if (pwd == null) return;
  userErr.value = "";
  try {
    await adminResetPassword(u.id, pwd);
  } catch (e) {
    userErr.value = e.message || String(e);
  }
}

async function removeUser(u) {
  if (!window.confirm(`确认删除用户 ${u.username}？此操作不可恢复。`)) return;
  userErr.value = "";
  try {
    await adminDeleteUser(u.id);
    await loadUsers();
  } catch (e) {
    userErr.value = e.message || String(e);
  }
}

watch(current, (id) => {
  closeZoom();
  if (id === "logs") loadLogs().catch((e) => { error.value = e.message || String(e); });
  if (id === "users") loadUsers().catch((e) => { error.value = e.message || String(e); });
  if (id === "eval") loadGolden().catch((e) => { error.value = e.message || String(e); });
  if (id === "import") adminImportStatus().then((d) => { importStatus.value = d; }).catch((e) => { error.value = e.message || String(e); });
});

onMounted(() => {
  window.addEventListener("keydown", onZoomKey);
  loadAll();
});
onUnmounted(() => window.removeEventListener("keydown", onZoomKey));
</script>
