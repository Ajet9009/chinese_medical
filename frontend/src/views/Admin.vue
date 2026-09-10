<template>
  <div class="admin-page">
    <h2>管理</h2>
    <p class="hint admin-lead">反馈、Prompt 来源与图谱导入健康。仅 user / admin 两角色，不开放自行注册。</p>
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
      </div>
      <table class="tbl">
        <thead>
          <tr>
            <th>用户</th>
            <th>问句</th>
            <th>回答</th>
            <th>反馈</th>
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
          </tr>
          <tr v-if="!(feedbacks.items || []).length">
            <td colspan="4" class="hint">暂无点赞或点踩</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section v-else-if="current === 'prompts'" class="card">
      <p class="hint" style="margin-top:0">
        Langfuse：{{ prompts.langfuse_enabled ? "已启用" : "未启用" }}。本页只读，不在此改模板。
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
      <div class="stat stat-accent">
        <h3>Redis</h3>
        <p class="stat-val">{{ importStatus.redis || "—" }}</p>
      </div>
      <div class="stat stat-accent">
        <h3>Neo4j</h3>
        <p class="stat-val">{{ importStatus.neo4j?.ok ? "连通" : "未连通" }}</p>
        <p v-if="importStatus.neo4j?.error" class="err">{{ importStatus.neo4j.error }}</p>
        <ul class="hint">
          <li v-for="(n, label) in importStatus.neo4j?.labels || {}" :key="label">
            {{ label }}：{{ n }}
          </li>
        </ul>
      </div>
      <div class="stat stat-accent">
        <h3>FAISS</h3>
        <p class="stat-val">{{ importStatus.faiss?.index_exists ? "索引已找到" : "索引缺失" }}</p>
        <p class="hint clip">{{ importStatus.faiss?.index_path }}</p>
        <p>{{ importStatus.faiss?.metadata_exists ? "元数据已找到" : "元数据缺失" }}</p>
        <p class="hint clip">{{ importStatus.faiss?.metadata_path }}</p>
      </div>
    </section>

    <section v-else-if="current === 'logs'" class="card">
      <p class="hint" style="margin-top:0">共 {{ logs.total || 0 }} 条。登录、改密与用户管理会写入这里。</p>
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
            <td colspan="4" class="hint">暂无日志</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section v-else class="card">
      <form class="user-form" @submit.prevent="createUser">
        <input class="input" v-model="newUser.username" placeholder="用户名" />
        <input class="input" v-model="newUser.password" type="password" placeholder="密码至少 6 位" />
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
    <div v-if="error" class="err">{{ error }}</div>
  </div>
</template>

<script setup>
import { onMounted, ref, watch } from "vue";
import {
  adminCreateUser,
  adminDeleteUser,
  adminFeedbacks,
  adminImportStatus,
  adminLogs,
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
  { id: "import", label: "💚 导入健康" },
  { id: "users", label: "👥 用户" },
  { id: "logs", label: "📋 操作日志" },
];
const current = ref("feedback");
const error = ref("");
const feedbacks = ref({ total: 0, items: [] });
const fbFilter = ref("");
const prompts = ref({ langfuse_enabled: false, items: [] });
const importStatus = ref({});
const users = ref([]);
const logs = ref({ total: 0, items: [] });
const newUser = ref({ username: "", password: "", role: "user" });
const userErr = ref("");
const meId = ref(auth.userId || "");

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

async function loadLogs() {
  logs.value = await adminLogs();
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
    });
    await loadFeedbacks();
    prompts.value = await adminPrompts();
    importStatus.value = await adminImportStatus();
    await loadUsers();
    await loadLogs();
  } catch (e) {
    error.value = e.message || String(e);
  }
}

async function createUser() {
  userErr.value = "";
  try {
    await adminCreateUser(newUser.value);
    newUser.value = { username: "", password: "", role: "user" };
    await loadUsers();
  } catch (e) {
    userErr.value = e.message || String(e);
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
  if (id === "logs") loadLogs().catch((e) => { error.value = e.message || String(e); });
  if (id === "users") loadUsers().catch((e) => { error.value = e.message || String(e); });
});

onMounted(loadAll);
</script>
