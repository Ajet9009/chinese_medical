<template>
  <div class="page governance-page">
    <div class="page-inner">
    <header class="page-head">
      <div>
        <h2>知识治理</h2>
        <p class="hint">补录责任人、有效期与版本；扫描只生成问题线索，不改原文。</p>
      </div>
      <div class="row">
        <button class="btn" type="button" @click="runScan()">发起扫描</button>
        <button class="btn ghost" type="button" :disabled="refreshing" @click="refreshAll">
          {{ refreshing ? "刷新中…" : "刷新" }}
        </button>
      </div>
    </header>
    <ol class="lifecycle-inline">
      <li v-for="(stage, index) in lifecycleStages" :key="stage.title">
        <b>{{ index + 1 }}</b>
        <span>{{ stage.title }} · {{ stage.desc }}</span>
      </li>
    </ol>

    <div class="mini-stats mini-stats-inline">
      <div class="mini-stat">
        <div class="mini-val">{{ stats.documents ?? 0 }}</div>
        <div class="mini-lbl">文献</div>
      </div>
      <div class="mini-stat">
        <div class="mini-val">{{ coveragePercent }}%</div>
        <div class="mini-lbl">元数据覆盖</div>
      </div>
      <div class="mini-stat">
        <div class="mini-val">{{ unresolvedCount }}</div>
        <div class="mini-lbl">待处置</div>
      </div>
      <div class="mini-stat">
        <div class="mini-val">{{ lastScan?.findings ?? 0 }}</div>
        <div class="mini-lbl">最近扫描</div>
      </div>
    </div>

    <section v-if="lastScan" class="scan-result">
      <div>
        <strong>治理扫描完成</strong>
        <span>
          扫描 {{ lastScan.documentsScanned || 0 }} 份，发现 {{ lastScan.findings || 0 }} 个问题；
          新增 {{ lastScan.created || 0 }}，更新 {{ lastScan.updated || 0 }}。
        </span>
      </div>
      <button class="icon-btn" type="button" @click="lastScan = null">×</button>
    </section>

    <div class="workspace-tabs">
      <button class="workspace-tab" :class="{ on: activeTab === 'documents' }" type="button" @click="activeTab = 'documents'">
        治理档案 <b>{{ docs.total || 0 }}</b>
      </button>
      <button class="workspace-tab" :class="{ on: activeTab === 'issues' }" type="button" @click="activeTab = 'issues'">
        问题工作台 <b>{{ unresolvedCount }}</b>
      </button>
    </div>

    <section v-show="activeTab === 'documents'" class="card workspace-card">
      <div class="card-header workspace-head">
        <div>
          <h2 class="card-title">文献治理档案</h2>
        </div>
        <div class="row">
          <input class="input" v-model.trim="docKeyword" placeholder="搜索文档名称" style="width:200px" @keyup.enter="searchDocuments" />
          <button class="btn ghost sm" type="button" @click="searchDocuments">查询</button>
        </div>
      </div>
      <div v-if="selectedDocIds.length" class="selection-bar">
        <span>已选择 <b>{{ selectedDocIds.length }}</b> 份</span>
        <button class="btn sm" type="button" @click="runScan(true)">扫描所选</button>
        <button class="btn ghost sm" type="button" @click="selectedDocIds = []">取消选择</button>
      </div>
      <div class="tbl-wrap">
        <table class="tbl">
          <thead>
            <tr>
              <th style="width:36px">
                <input type="checkbox" :checked="allCurrentDocsSelected" @change="toggleCurrentDocs($event.target.checked)" />
              </th>
              <th>文档</th>
              <th>治理状态</th>
              <th>责任人与范围</th>
              <th>版本</th>
              <th>待补字段</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody v-if="docsLoading">
            <tr v-for="i in 5" :key="i"><td colspan="7"><div class="skeleton"></div></td></tr>
          </tbody>
          <tbody v-else>
            <tr v-for="doc in docs.list" :key="doc.docId">
              <td><input v-model="selectedDocIds" type="checkbox" :value="doc.docId" /></td>
              <td>
                <div class="doc-title">{{ doc.docName }}</div>
                <div class="cell-meta">{{ doc.docType || "未分类" }} · {{ documentStatusLabel(doc.documentStatus) }}</div>
              </td>
              <td>
                <span class="badge" :class="effectiveStateBadge(doc.effectiveState)">
                  {{ effectiveStateLabel(doc.effectiveState) }}
                </span>
              </td>
              <td>
                <div>{{ doc.metadata?.owner || "未指定责任人" }}</div>
                <div class="cell-meta">{{ doc.metadata?.applicableRegion || "适用范围待补录" }}</div>
              </td>
              <td>
                <div>{{ doc.metadata?.versionLabel || "未标注" }}</div>
                <div class="cell-meta">{{ versionStatusLabel(doc.metadata?.versionStatus) }}</div>
              </td>
              <td>
                <span v-if="doc.missingFields?.length" class="badge badge-warning">{{ doc.missingFields.length }} 项</span>
                <span v-else class="badge badge-success">档案完整</span>
              </td>
              <td>
                <button class="btn ghost sm" type="button" @click="openProfile(doc)">
                  {{ doc.metadata ? "编辑档案" : "建立档案" }}
                </button>
              </td>
            </tr>
            <tr v-if="!docs.list?.length">
              <td colspan="7" class="hint">没有找到文档。请先在知识库中上传文献。</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="row pager">
        <span class="hint">共 {{ docs.total || 0 }} 份</span>
        <button class="btn ghost sm" type="button" :disabled="docPage <= 1" @click="changeDocPage(docPage - 1)">上一页</button>
        <b>{{ docPage }} / {{ docTotalPages }}</b>
        <button class="btn ghost sm" type="button" :disabled="docPage >= docTotalPages" @click="changeDocPage(docPage + 1)">下一页</button>
      </div>
    </section>

    <section v-show="activeTab === 'issues'" class="card workspace-card">
      <div class="card-header workspace-head">
        <div>
          <h2 class="card-title">治理问题工作台</h2>
          <div class="hint">扫描只生成线索，最终状态由审核人确认。不自动改原文。</div>
        </div>
        <button class="btn ghost sm" type="button" :disabled="issuesLoading" @click="loadIssues">刷新问题</button>
      </div>
      <div class="row issue-filters">
        <input class="input" v-model.trim="issueFilters.keyword" placeholder="搜索标题或摘要" style="width:200px" @keyup.enter="loadIssues" />
        <select class="select" v-model="issueFilters.status" style="width:auto" @change="loadIssues">
          <option value="">全部状态</option>
          <option value="open">待确认</option>
          <option value="confirmed">已确认</option>
          <option value="resolved">已解决</option>
          <option value="ignored">已忽略</option>
        </select>
        <select class="select" v-model="issueFilters.type" style="width:auto" @change="loadIssues">
          <option value="">全部类型</option>
          <option v-for="type in issueTypes" :key="type" :value="type">{{ issueTypeLabel(type) }}</option>
        </select>
        <select class="select" v-model="issueFilters.severity" style="width:auto" @change="loadIssues">
          <option value="">全部等级</option>
          <option value="critical">严重</option>
          <option value="warning">警告</option>
          <option value="info">提示</option>
        </select>
      </div>
      <div v-if="issuesLoading" class="hint">加载中…</div>
      <div v-else-if="issues.list?.length" class="issue-list">
        <article v-for="issue in issues.list" :key="issue.id" class="issue-card">
          <div class="issue-tags">
            <span class="badge" :class="severityBadge(issue.severity)">{{ severityLabel(issue.severity) }}</span>
            <span class="badge badge-neutral">{{ issueTypeLabel(issue.type) }}</span>
            <span class="badge" :class="issueStatusBadge(issue.status)">{{ issueStatusLabel(issue.status) }}</span>
          </div>
          <strong>{{ issue.title }}</strong>
          <p class="hint">{{ issue.summary }}</p>
          <div v-if="issue.evidence?.missingFields?.length" class="cell-meta">
            缺失：{{ issue.evidence.missingFields.map(missingFieldLabel).join("、") }}
          </div>
          <div v-if="issue.reviewNote" class="review-note">审核说明：{{ issue.reviewNote }}</div>
          <div class="row" style="margin-top:8px">
            <template v-if="['open', 'confirmed'].includes(issue.status)">
              <button v-if="issue.status === 'open'" class="btn ghost sm" type="button" @click="openReview(issue, 'confirmed')">确认问题</button>
              <button class="btn sm" type="button" @click="openReview(issue, 'resolved')">标记解决</button>
              <button class="btn ghost sm" type="button" @click="openReview(issue, 'ignored')">忽略</button>
            </template>
            <button v-else class="btn ghost sm" type="button" @click="openReview(issue, 'open')">重新打开</button>
          </div>
        </article>
      </div>
      <div v-else class="hint">当前筛选下没有治理问题。可以发起一次扫描。</div>
    </section>

    <div v-if="profileModal.show" class="modal-bg" @click.self="closeProfile">
      <div class="modal profile-modal">
        <div class="modal-head">
          <span>{{ profileModal.doc?.docName || "治理档案" }}</span>
          <button class="icon-btn" type="button" @click="closeProfile">×</button>
        </div>
        <form class="profile-form" @submit.prevent="submitProfile">
          <p class="hint">
            当前状态
            <span class="badge" :class="effectiveStateBadge(profileModal.effectiveState)">
              {{ effectiveStateLabel(profileModal.effectiveState) }}
            </span>
          </p>
          <div class="form-grid">
            <label class="field">
              <span class="field-label">责任人</span>
              <input v-model.trim="profileForm.owner" class="input" maxlength="64" placeholder="例如：脾胃科 / 张医师" />
            </label>
            <label class="field">
              <span class="field-label">适用区域</span>
              <input v-model.trim="profileForm.applicableRegion" class="input" maxlength="256" placeholder="例如：脾胃气虚" />
            </label>
            <label class="field">
              <span class="field-label">生效时间</span>
              <input v-model="profileForm.effectiveAt" class="input" type="datetime-local" />
            </label>
            <label class="field">
              <span class="field-label">失效时间</span>
              <input v-model="profileForm.expiresAt" class="input" type="datetime-local" :disabled="profileForm.isPermanent" />
            </label>
            <label class="chk field-span"><input v-model="profileForm.isPermanent" type="checkbox" @change="onPermanentChange" /> 永久有效</label>
            <label class="field">
              <span class="field-label">复审周期（天）</span>
              <input v-model.number="profileForm.reviewIntervalDays" class="input" type="number" min="1" max="3650" />
            </label>
            <label class="field">
              <span class="field-label">下次复审时间</span>
              <input v-model="profileForm.nextReviewAt" class="input" type="datetime-local" />
            </label>
            <label class="field">
              <span class="field-label">版本标识</span>
              <input v-model.trim="profileForm.versionLabel" class="input" maxlength="64" placeholder="例如：伤寒论宋本" />
            </label>
            <label class="field">
              <span class="field-label">版本状态</span>
              <select v-model="profileForm.versionStatus" class="select">
                <option value="">请选择</option>
                <option value="draft">草稿</option>
                <option value="active">现行有效</option>
                <option value="superseded">已被替代</option>
                <option value="withdrawn">已撤回</option>
              </select>
            </label>
          </div>
          <p v-if="['superseded', 'withdrawn'].includes(profileForm.versionStatus)" class="err">
            该状态会阻止文档进入最终检索结果。
          </p>
          <p v-if="profileError" class="err">{{ profileError }}</p>
          <div class="modal-actions">
            <button class="btn ghost" type="button" @click="closeProfile">取消</button>
            <button class="btn" type="submit" :disabled="profileModal.saving">
              {{ profileModal.saving ? "保存中…" : "保存治理档案" }}
            </button>
          </div>
        </form>
      </div>
    </div>

    <div v-if="reviewModal.show" class="modal-bg" @click.self="closeReview">
      <div class="modal review-modal">
        <div class="modal-head">
          <span>{{ reviewActionLabel(reviewModal.status) }}</span>
          <button class="icon-btn" type="button" @click="closeReview">×</button>
        </div>
        <form class="profile-form" @submit.prevent="submitReview">
          <p><strong>{{ reviewModal.issue?.title }}</strong></p>
          <p class="hint">{{ reviewModal.issue?.summary }}</p>
          <label class="field">
            <span class="field-label">
              审核说明 <span v-if="['resolved', 'ignored'].includes(reviewModal.status)">必填</span>
            </span>
            <textarea v-model.trim="reviewModal.note" class="input review-textarea" maxlength="2000"></textarea>
          </label>
          <p v-if="reviewError" class="err">{{ reviewError }}</p>
          <div class="modal-actions">
            <button class="btn ghost" type="button" @click="closeReview">取消</button>
            <button class="btn" type="submit" :disabled="reviewModal.saving">提交</button>
          </div>
        </form>
      </div>
    </div>
    <div v-if="toastMsg" class="toast">{{ toastMsg }}</div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from "vue";
import { govDocuments, govIssueStats, govIssues, govProfile, govReview, govScan, saveGovProfile } from "../api.js";

const lifecycleStages = [
  { title: "责任归属", desc: "责任人与适用范围" },
  { title: "有效窗口", desc: "生效及失效时间" },
  { title: "定期复审", desc: "周期性校验内容" },
  { title: "版本退出", desc: "替代、撤回可追溯" },
];
const issueTypes = ["metadata_missing", "not_yet_effective", "expired", "expiring", "review_due"];
const pageSize = 12;
const activeTab = ref("documents");
const refreshing = ref(false);
const stats = ref({ documents: 0, governedDocuments: 0, metadataCoverage: 0, byStatus: {}, unresolved: 0 });
const docs = ref({ total: 0, list: [] });
const docsLoading = ref(false);
const docKeyword = ref("");
const docPage = ref(1);
const selectedDocIds = ref([]);
const issues = ref({ total: 0, list: [] });
const issuesLoading = ref(false);
const issueFilters = reactive({ keyword: "", status: "", type: "", severity: "" });
const lastScan = ref(null);
const profileModal = reactive({ show: false, saving: false, doc: null, effectiveState: "metadata_incomplete" });
const profileForm = reactive({
  owner: "",
  applicableRegion: "",
  effectiveAt: "",
  expiresAt: "",
  isPermanent: false,
  reviewIntervalDays: "",
  nextReviewAt: "",
  versionLabel: "",
  versionStatus: "",
});
const profileError = ref("");
const reviewModal = reactive({ show: false, saving: false, issue: null, status: "confirmed", note: "" });
const reviewError = ref("");
const toastMsg = ref("");
let toastTimer = null;

const coveragePercent = computed(() => {
  if (stats.value.metadataCoverage != null) {
    const n = Number(stats.value.metadataCoverage);
    if (n <= 1) return Math.round(Math.max(0, Math.min(1, n)) * 100);
    return Math.round(Math.max(0, Math.min(100, n)));
  }
  const d = stats.value.documents || 0;
  return d ? Math.round(((stats.value.governedDocuments || 0) / d) * 100) : 0;
});
const unresolvedCount = computed(
  () => stats.value.unresolved || (stats.value.byStatus?.open || 0) + (stats.value.byStatus?.confirmed || 0)
);
const docTotalPages = computed(() => Math.max(1, Math.ceil((docs.value.total || 0) / pageSize)));
const allCurrentDocsSelected = computed(
  () => docs.value.list?.length > 0 && docs.value.list.every((doc) => selectedDocIds.value.includes(doc.docId))
);

function toast(m) {
  toastMsg.value = m;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toastMsg.value = "";
  }, 2200);
}
function toLocal(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso).slice(0, 16);
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}
function documentStatusLabel(s) {
  return { pending: "待解析", parsed: "已解析", vectorized: "已向量化" }[s] || s || "—";
}
function effectiveStateLabel(s) {
  return {
    metadata_incomplete: "档案不完整",
    draft: "草稿",
    active: "现行有效",
    not_yet_effective: "尚未生效",
    expired: "已过期",
    superseded: "已被替代",
    withdrawn: "已撤回",
  }[s] || s || "—";
}
function effectiveStateBadge(s) {
  return {
    active: "badge-success",
    metadata_incomplete: "badge-warning",
    draft: "badge-neutral",
    not_yet_effective: "badge-info",
    expired: "badge-danger",
    superseded: "badge-neutral",
    withdrawn: "badge-danger",
  }[s] || "badge-neutral";
}
function versionStatusLabel(s) {
  return { draft: "草稿", active: "现行有效", superseded: "已被替代", withdrawn: "已撤回" }[s] || "未标注";
}
function missingFieldLabel(f) {
  return {
    owner: "责任人",
    applicableRegion: "适用区域",
    effectiveAt: "生效时间",
    expiryPolicy: "失效策略",
    reviewPolicy: "复审策略",
    versionLabel: "版本标识",
    versionStatus: "版本状态",
  }[f] || f;
}
function issueTypeLabel(t) {
  return {
    metadata_missing: "元数据缺失",
    not_yet_effective: "尚未生效",
    expired: "已过期",
    expiring: "即将到期",
    review_due: "到达复审日",
  }[t] || t;
}
function severityLabel(s) {
  return { critical: "严重", warning: "警告", info: "提示" }[s] || s;
}
function severityBadge(s) {
  return { critical: "badge-danger", warning: "badge-warning", info: "badge-info" }[s] || "badge-neutral";
}
function issueStatusLabel(s) {
  return { open: "待确认", confirmed: "已确认", resolved: "已解决", ignored: "已忽略" }[s] || s;
}
function issueStatusBadge(s) {
  return { open: "badge-warning", confirmed: "badge-info", resolved: "badge-success", ignored: "badge-neutral" }[s] || "badge-neutral";
}
function reviewActionLabel(s) {
  return { confirmed: "确认问题", resolved: "标记解决", ignored: "忽略问题", open: "重新打开" }[s] || "审核";
}

async function loadStats() {
  stats.value = { ...stats.value, ...(await govIssueStats()) };
}
async function loadDocuments() {
  docsLoading.value = true;
  try {
    docs.value = await govDocuments(docPage.value, pageSize, docKeyword.value);
  } finally {
    docsLoading.value = false;
  }
}
async function loadIssues() {
  issuesLoading.value = true;
  try {
    issues.value = await govIssues(issueFilters);
  } finally {
    issuesLoading.value = false;
  }
}
async function refreshAll() {
  refreshing.value = true;
  try {
    await Promise.all([loadStats(), loadDocuments(), loadIssues()]);
  } finally {
    refreshing.value = false;
  }
}
function searchDocuments() {
  docPage.value = 1;
  loadDocuments();
}
function changeDocPage(p) {
  if (p < 1 || p > docTotalPages.value) return;
  docPage.value = p;
  loadDocuments();
}
function toggleCurrentDocs(c) {
  const ids = (docs.value.list || []).map((d) => d.docId);
  if (c) selectedDocIds.value = [...new Set([...selectedDocIds.value, ...ids])];
  else selectedDocIds.value = selectedDocIds.value.filter((id) => !ids.includes(id));
}
async function runScan(selectedOnly = false) {
  const ids = selectedOnly ? selectedDocIds.value : [];
  try {
    lastScan.value = await govScan(ids);
    await refreshAll();
    toast("扫描完成");
  } catch (e) {
    toast(e.message || "扫描失败");
  }
}
async function openProfile(doc) {
  profileModal.doc = doc;
  profileModal.show = true;
  profileError.value = "";
  const meta = (await govProfile(doc.docId)) || {};
  profileForm.owner = meta.owner || "";
  profileForm.applicableRegion = meta.applicableRegion || meta.applicable_region || "";
  profileForm.effectiveAt = toLocal(meta.effectiveAt || meta.effective_at);
  profileForm.expiresAt = toLocal(meta.expiresAt || meta.expires_at);
  profileForm.isPermanent = Boolean(meta.isPermanent || meta.is_permanent);
  profileForm.reviewIntervalDays = meta.reviewIntervalDays || meta.review_interval_days || "";
  profileForm.nextReviewAt = toLocal(meta.nextReviewAt || meta.next_review_at);
  profileForm.versionLabel = meta.versionLabel || meta.version_label || "";
  profileForm.versionStatus = meta.versionStatus || meta.version_status || "";
  profileModal.effectiveState = doc.effectiveState || "metadata_incomplete";
}
function closeProfile() {
  profileModal.show = false;
}
function onPermanentChange() {
  if (profileForm.isPermanent) profileForm.expiresAt = "";
}
async function submitProfile() {
  profileError.value = "";
  profileModal.saving = true;
  try {
    await saveGovProfile(profileModal.doc.docId, { ...profileForm });
    closeProfile();
    await refreshAll();
    toast("档案已保存");
  } catch (e) {
    profileError.value = e.message || "保存失败";
  } finally {
    profileModal.saving = false;
  }
}
function openReview(issue, status) {
  reviewModal.show = true;
  reviewModal.issue = issue;
  reviewModal.status = status;
  reviewModal.note = "";
  reviewError.value = "";
}
function closeReview() {
  reviewModal.show = false;
}
async function submitReview() {
  reviewError.value = "";
  if (["resolved", "ignored"].includes(reviewModal.status) && !reviewModal.note) {
    reviewError.value = "解决或忽略需要填写说明";
    return;
  }
  reviewModal.saving = true;
  try {
    await govReview(reviewModal.issue.id, reviewModal.status, reviewModal.note);
    closeReview();
    await refreshAll();
    toast("已更新");
  } catch (e) {
    reviewError.value = e.message || "提交失败";
  } finally {
    reviewModal.saving = false;
  }
}

onMounted(refreshAll);
</script>
