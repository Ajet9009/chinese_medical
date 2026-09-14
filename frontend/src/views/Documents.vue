<template>
  <div class="page doc-page">
    <div class="page-inner">
    <header class="page-head">
      <div>
        <h2>知识库</h2>
        <p class="hint">解析并向量化后才能被问答检索。同名换版会归档，可回滚。</p>
      </div>
      <div class="mini-stats mini-stats-inline">
        <div class="mini-stat"><div class="mini-val">{{ stats?.docTotal ?? "-" }}</div><div class="mini-lbl">文档</div></div>
        <div class="mini-stat"><div class="mini-val">{{ stats?.chunkTotal ?? "-" }}</div><div class="mini-lbl">分块</div></div>
        <div class="mini-stat"><div class="mini-val">{{ stats?.vectorTotal ?? "-" }}</div><div class="mini-lbl">向量</div></div>
        <div class="mini-stat"><div class="mini-val">{{ stats?.byStatus?.vectorized ?? 0 }}</div><div class="mini-lbl">已向量化</div></div>
      </div>
    </header>

    <section v-if="auth.isAdmin" class="card upload-panel">
        <div class="card-header">
          <h3 class="card-title">上传文献</h3>
          <span class="hint">Markdown / 文本 / PDF · 批量≤5 · 单份≤8MB</span>
        </div>
        <div
          class="dropzone"
          :class="{ over: dragOver }"
          @dragover.prevent="dragOver = true"
          @dragleave.prevent="dragOver = false"
          @drop.prevent="onDrop"
          @click="$refs.fileInput.click()"
        >
          <input
            ref="fileInput"
            type="file"
            multiple
            accept=".md,.txt,.pdf"
            style="display:none"
            @change="onFile"
          />
          <div v-if="!files.length" class="dz-empty">
            <div class="dz-icon">📂</div>
            <div>拖拽文件到此处，或<span class="dz-link">点击选择</span></div>
          </div>
          <ul v-else class="file-list">
            <li v-for="(f, i) in files" :key="i">
              <span class="ficon">📄</span>{{ f.name }}
              <span class="hint">({{ fmtSize(f.size) }})</span>
            </li>
          </ul>
        </div>
        <div class="row" style="margin-top: 14px">
          <select class="select" v-model="docType">
            <option v-for="t in DOC_TYPES" :key="t">{{ t }}</option>
          </select>
          <input
            class="input"
            v-model="dept"
            placeholder="科室(脾胃/伤寒,空=公开)"
            title="文档级 ACL：限定某科室可见，空=全员公开"
          />
          <input
            class="input"
            v-model="allowedRoles"
            placeholder="授权角色(user,admin,空=全员)"
            title="限定可读角色，空=科室内全员"
          />
          <button class="btn" type="button" :disabled="!files.length || uploading" @click="upload">
            {{ uploading ? "上传中…" : "上传" }}
          </button>
          <button v-if="files.length" class="btn ghost" type="button" @click="files = []">清空</button>
        </div>
        <details class="gov-opt">
          <summary>治理元数据（可选：上传即填，免后续扫描补录）</summary>
          <div class="row" style="margin-top: 6px">
            <input class="input" v-model="effectiveAt" type="datetime-local" title="生效时间" style="width:200px" />
            <input class="input" v-model="expiresAt" type="datetime-local" title="失效时间" style="width:200px" />
            <label class="chk"><input type="checkbox" v-model="isPermanent" /> 永久有效</label>
            <input
              class="input"
              v-model="versionOf"
              placeholder="版本继承(可选,旧 doc_id)"
              style="width:220px"
              title="新版替代旧版时，填旧文档 id 以建立 supersede 链"
            />
          </div>
        </details>
        <div v-if="uploading" class="upload-progress">
          <div class="progress-bar" :style="{ width: progress + '%' }"></div>
          <span>{{ progress }}%</span>
        </div>
    </section>

    <div class="card doc-list">
      <div class="card-header">
        <h3 class="card-title">文献列表 <span class="badge badge-neutral">{{ total }}</span></h3>
        <div class="row">
          <input class="input" v-model="filterKw" placeholder="🔍 按文件名筛选" style="width:200px" />
          <select class="select" v-model="filterType" style="width:auto" @change="resetPage">
            <option value="">全部类型</option>
            <option v-for="t in DOC_TYPES" :key="t">{{ t }}</option>
          </select>
          <select class="select" v-model="filterStatus" style="width:auto" @change="resetPage">
            <option value="">全部状态</option>
            <option value="pending">待解析</option>
            <option value="parsed">已解析</option>
            <option value="vectorized">已向量化</option>
          </select>
          <template v-if="auth.isAdmin">
            <button class="btn ghost sm" type="button" :disabled="!selected.length" @click="batchParse">
              批量解析({{ selected.length }})
            </button>
            <button class="btn ghost sm" type="button" :disabled="!selected.length" @click="batchVectorize">
              批量向量化({{ selected.length }})
            </button>
            <button class="btn danger sm" type="button" :disabled="!selected.length" @click="batchDelete">
              批量删除
            </button>
          </template>
        </div>
      </div>
      <div class="tbl-wrap">
        <table class="tbl">
          <thead>
            <tr>
              <th v-if="auth.isAdmin" style="width:36px">
                <input type="checkbox" :checked="allChecked" @change="toggleAll($event.target.checked)" />
              </th>
              <th>文件名</th>
              <th>类型</th>
              <th>状态</th>
              <th>分块</th>
              <th>关联方药</th>
              <th v-if="auth.isAdmin" style="width:240px">操作</th>
            </tr>
          </thead>
          <tbody v-if="loading">
            <tr v-for="i in 5" :key="i"><td :colspan="auth.isAdmin ? 7 : 5"><div class="skeleton"></div></td></tr>
          </tbody>
          <tbody v-else>
            <tr v-for="d in filtered" :key="d.docId">
              <td v-if="auth.isAdmin"><input type="checkbox" :value="d.docId" v-model="selected" /></td>
              <td>
                <a class="doc-name" :title="'预览 ' + d.docName" @click="previewDoc(d)">{{ d.docName }}</a>
              </td>
              <td><span class="badge badge-neutral">{{ d.docType }}</span></td>
              <td><span class="badge" :class="statusBadge(d.status)">{{ statusMap[d.status] || d.status }}</span></td>
              <td>{{ d.chunkCount }}</td>
              <td class="eq-cell">{{ herbLabel(d) }}</td>
              <td v-if="auth.isAdmin">
                <button class="btn ghost sm" type="button" :disabled="busy[d.docId]" @click="parseDoc(d.docId)">解析</button>
                <button class="btn ghost sm" type="button" :disabled="busy[d.docId]" @click="vectorizeDoc(d.docId)">向量化</button>
                <button class="btn ghost sm" type="button" :disabled="busy[d.docId]" @click="openVersions(d)">版本</button>
                <button class="btn danger sm" type="button" :disabled="busy[d.docId]" @click="removeDoc(d.docId)">删除</button>
              </td>
            </tr>
            <tr v-if="!filtered.length">
              <td :colspan="auth.isAdmin ? 7 : 5" class="hint">无匹配文献</td>
            </tr>
          </tbody>
        </table>
        <div class="row pager">
          <span class="hint">共 {{ total }} 条 · 第 {{ page }}/{{ totalPages }} 页</span>
          <div class="row" style="gap:6px">
            <button class="btn ghost sm" type="button" :disabled="page <= 1" @click="goPage(page - 1)">上一页</button>
            <button class="btn ghost sm" type="button" :disabled="page >= totalPages" @click="goPage(page + 1)">下一页</button>
          </div>
        </div>
      </div>
    </div>

    <div v-if="preview.show" class="modal-bg" @click.self="closePreview">
      <div class="modal">
        <div class="modal-head">
          <span>📄 文献预览</span>
          <button class="icon-btn" type="button" @click="closePreview">✕</button>
        </div>
        <iframe v-if="preview.type === 'pdf'" :src="preview.url" style="flex:1;border:0"></iframe>
        <pre v-else class="preview-text">{{ preview.text }}</pre>
      </div>
    </div>

    <div v-if="versions.show" class="modal-bg" @click.self="versions.show = false">
      <div class="modal versions-modal">
        <div class="modal-head">
          <span>版本 · {{ versions.doc?.docName }}</span>
          <button class="icon-btn" type="button" @click="versions.show = false">✕</button>
        </div>
        <p class="hint" style="margin:0 18px 8px">
          现行是当前文件。归档只在<strong>同名再上传</strong>时产生，才能回滚。
        </p>
        <table class="tbl">
          <thead>
            <tr><th>版本</th><th>大小</th><th>上传人</th><th>时间</th><th></th></tr>
          </thead>
          <tbody>
            <tr v-if="versions.current">
              <td>v{{ versions.current.version }} <span class="badge badge-success">现行</span></td>
              <td>{{ fmtSize(versions.current.fileSize) }}</td>
              <td>{{ versions.current.createdBy || "—" }}</td>
              <td>{{ formatTime(versions.current.createdAt) }}</td>
              <td class="hint">当前</td>
            </tr>
            <tr v-for="v in versions.list" :key="v.version">
              <td>v{{ v.version }}</td>
              <td>{{ fmtSize(v.fileSize) }}</td>
              <td>{{ v.createdBy || "—" }}</td>
              <td>{{ formatTime(v.createdAt) }}</td>
              <td>
                <button class="btn ghost sm" type="button" @click="doRollback(v.version)">回滚</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
    </div>
    <div v-if="toastMsg" class="toast">{{ toastMsg }}</div>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from "vue";
import {
  deleteDocs,
  getDocStats,
  listDocs,
  listDocVersions,
  parseDocs,
  rollbackDoc,
  uploadDocs,
  vectorize,
  vectorizeBatch,
} from "../api.js";
import { useAuthStore } from "../stores/auth.js";

const DOC_TYPES = ["方剂", "本草", "典籍", "医案", "其他"];
const auth = useAuthStore();
const docs = ref([]);
const stats = ref(null);
const files = ref([]);
const docType = ref("方剂");
const dept = ref("");
const allowedRoles = ref("");
const effectiveAt = ref("");
const expiresAt = ref("");
const isPermanent = ref(false);
const versionOf = ref("");
const uploading = ref(false);
const progress = ref(0);
const dragOver = ref(false);
const busy = reactive({});
const selected = ref([]);
const filterKw = ref("");
const filterType = ref("");
const filterStatus = ref("");
const page = ref(1);
const pageSize = ref(20);
const total = ref(0);
const loading = ref(false);
const statusMap = { pending: "待解析", parsed: "已解析", vectorized: "已向量化" };
const preview = reactive({ show: false, type: "", url: "", text: "" });
const versions = reactive({ show: false, doc: null, current: null, list: [] });
const toastMsg = ref("");
let toastTimer = null;

const totalPages = computed(() => Math.max(1, Math.ceil(total.value / pageSize.value)));
const filtered = computed(() =>
  docs.value.filter(
    (d) =>
      (!filterKw.value || (d.docName || "").includes(filterKw.value)) &&
      (!filterType.value || d.docType === filterType.value) &&
      (!filterStatus.value || d.status === filterStatus.value)
  )
);
const allChecked = computed(
  () => filtered.value.length > 0 && selected.value.length === filtered.value.length
);

function toast(m) {
  toastMsg.value = m;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toastMsg.value = "";
  }, 1800);
}
function statusBadge(s) {
  return { pending: "badge-neutral", parsed: "badge-info", vectorized: "badge-success" }[s] || "badge-neutral";
}
function herbLabel(d) {
  const raw = d.herbTags || d.equipmentTags || "";
  const tags = raw.split(",").filter(Boolean).slice(0, 3);
  return tags.join("、") || "—";
}
function fmtSize(b) {
  if (!b) return "0B";
  if (b < 1024) return b + "B";
  if (b < 1048576) return (b / 1024).toFixed(1) + "KB";
  return (b / 1048576).toFixed(1) + "MB";
}
function formatTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}

async function load() {
  loading.value = true;
  try {
    const [d, s] = await Promise.all([listDocs(page.value, pageSize.value), getDocStats()]);
    docs.value = d.list || [];
    total.value = d.total || 0;
    stats.value = s;
  } finally {
    loading.value = false;
  }
}
function goPage(p) {
  if (p < 1 || p > totalPages.value) return;
  page.value = p;
  load();
}
function resetPage() {
  page.value = 1;
  load();
}
function onFile(e) {
  files.value = [...files.value, ...Array.from(e.target.files || [])].slice(0, 5);
  e.target.value = "";
}
function onDrop(e) {
  dragOver.value = false;
  files.value = [...files.value, ...Array.from(e.dataTransfer.files || [])].slice(0, 5);
}
async function upload() {
  if (!files.value.length) return;
  const form = new FormData();
  files.value.forEach((f) => form.append("files", f));
  form.append("docType", docType.value);
  form.append("dept", dept.value);
  form.append("allowedRoles", allowedRoles.value);
  if (effectiveAt.value) form.append("effectiveAt", effectiveAt.value);
  if (expiresAt.value) form.append("expiresAt", expiresAt.value);
  if (isPermanent.value) form.append("isPermanent", "true");
  if (versionOf.value) form.append("versionOf", versionOf.value);
  uploading.value = true;
  progress.value = 0;
  try {
    const data = await uploadDocs(form, (e) => {
      if (e.total) progress.value = Math.round((e.loaded / e.total) * 100);
    });
    await load();
    const fail = (data.failList || []).length;
    toast(fail ? `上传完成，失败 ${fail} 份` : "上传成功");
    files.value = [];
  } catch (e) {
    toast(e.message || "上传失败");
  }
  uploading.value = false;
}
function toggleAll(c) {
  selected.value = c ? filtered.value.map((d) => d.docId) : [];
}
async function parseDoc(id) {
  busy[id] = true;
  try {
    await parseDocs([id]);
    await load();
    toast("解析完成");
  } catch (e) {
    toast(e.message || "解析失败");
  } finally {
    busy[id] = false;
  }
}
async function vectorizeDoc(id) {
  busy[id] = true;
  try {
    await vectorize(id);
    await load();
    toast("向量化完成");
  } catch (e) {
    toast(e.message || "向量化失败");
  } finally {
    busy[id] = false;
  }
}
async function removeDoc(id) {
  if (!confirm("确认删除该文献（含向量）？")) return;
  busy[id] = true;
  try {
    await deleteDocs([id]);
    await load();
    toast("已删除");
  } finally {
    busy[id] = false;
  }
}
async function batchParse() {
  if (!selected.value.length) return;
  try {
    await parseDocs(selected.value);
    await load();
    toast(`已解析 ${selected.value.length} 份`);
    selected.value = [];
  } catch (e) {
    toast(e.message || "批量解析失败");
  }
}
async function batchVectorize() {
  if (!selected.value.length) return;
  try {
    const resp = await vectorizeBatch(selected.value);
    const ok = resp.successList?.length || 0;
    const fail = resp.failList?.length || 0;
    await load();
    toast(fail ? `向量化完成 ${ok} 份，失败 ${fail} 份` : `已向量化 ${ok} 份`);
    selected.value = [];
  } catch (e) {
    toast(e.message || "批量向量化失败");
  }
}
async function batchDelete() {
  if (!selected.value.length || !confirm(`删除 ${selected.value.length} 份？`)) return;
  try {
    await deleteDocs(selected.value);
    await load();
    toast("已删除");
    selected.value = [];
  } catch (e) {
    toast(e.message || "批量删除失败");
  }
}
async function previewDoc(d) {
  const ext = (d.docName.split(".").pop() || "").toLowerCase();
  try {
    const resp = await fetch(`/api/document/preview/${d.docId}`, {
      headers: { Authorization: `Bearer ${auth.token}` },
    });
    if (!resp.ok) {
      toast("该格式不支持预览");
      return;
    }
    const blob = await resp.blob();
    if (ext === "pdf") {
      preview.type = "pdf";
      preview.url = URL.createObjectURL(blob);
    } else {
      preview.type = "text";
      preview.text = await blob.text();
    }
    preview.show = true;
  } catch {
    toast("预览失败");
  }
}
function closePreview() {
  if (preview.url) URL.revokeObjectURL(preview.url);
  preview.show = false;
  preview.url = "";
  preview.text = "";
}
async function openVersions(d) {
  versions.doc = d;
  const data = await listDocVersions(d.docId);
  if (Array.isArray(data)) {
    versions.current = null;
    versions.list = data;
  } else {
    versions.current = data.current || null;
    versions.list = data.list || [];
  }
  versions.show = true;
}
async function doRollback(version) {
  if (!versions.doc) return;
  if (!confirm(`回滚到 v${version}？当前文件将被覆盖。`)) return;
  try {
    await rollbackDoc(versions.doc.docId, version);
    versions.show = false;
    await load();
    toast("已回滚，请重新解析");
  } catch (e) {
    toast(e.message || "回滚失败");
  }
}

onMounted(load);
</script>
