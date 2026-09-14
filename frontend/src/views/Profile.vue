<template>
  <div class="page admin-page">
    <div class="page-inner">
    <header class="page-head">
      <div>
        <h2>账号</h2>
        <p class="hint">查看身份并修改密码，至少 6 位。</p>
      </div>
    </header>
    <section class="card profile-card">
      <p><strong>{{ info.username || auth.username }}</strong>
        <span class="muted"> · {{ roleLabel }}</span>
        <span class="badge" :class="info.status === 'inactive' ? 'badge-neutral' : 'badge-success'">
          {{ info.status === "inactive" ? "禁用" : "可用" }}
        </span>
      </p>
      <form class="dept-form" @submit.prevent="saveDept">
        <div class="field">
          <label class="field-label">科室</label>
          <input class="input" v-model="dept" placeholder="空=不限科室（公开文献均可见）" />
        </div>
        <button class="btn ghost" type="submit" :disabled="deptBusy">保存科室</button>
      </form>
      <p class="hint">科室用于知识库 ACL：文献限定某科室时，仅该科室医师与管理员可见。</p>
      <form @submit.prevent="onSubmit">
        <div class="field">
          <label class="field-label">原密码</label>
          <input class="input" v-model="oldPassword" type="password" autocomplete="current-password" />
        </div>
        <div class="field">
          <label class="field-label">新密码</label>
          <input class="input" v-model="newPassword" type="password" autocomplete="new-password" />
        </div>
        <div class="field">
          <label class="field-label">确认新密码</label>
          <input class="input" v-model="confirmPassword" type="password" autocomplete="new-password" />
        </div>
        <button class="btn" type="submit" :disabled="busy">🔑 修改密码</button>
      </form>
      <p v-if="ok" class="hint">密码已更新，下次请用新密码登录。</p>
      <p v-if="deptOk" class="hint">{{ deptOk }}</p>
      <p v-if="error" class="err">{{ error }}</p>
    </section>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from "vue";
import { changePassword, me, patchMe } from "../api.js";
import { useAuthStore } from "../stores/auth.js";

const auth = useAuthStore();
const info = ref({ username: "", role: "", status: "active", dept: "" });
const dept = ref("");
const oldPassword = ref("");
const newPassword = ref("");
const confirmPassword = ref("");
const busy = ref(false);
const deptBusy = ref(false);
const error = ref("");
const ok = ref(false);
const deptOk = ref("");

const roleLabel = computed(() =>
  (info.value.role || auth.role) === "admin" ? "管理员" : "医师"
);

onMounted(async () => {
  try {
    const data = await me();
    info.value = data;
    dept.value = data.dept || "";
    auth.setSession({
      token: auth.token,
      id: data.id,
      username: data.username,
      role: data.role,
      dept: data.dept || "",
    });
  } catch (e) {
    error.value = e.message || String(e);
  }
});

async function saveDept() {
  error.value = "";
  deptOk.value = "";
  deptBusy.value = true;
  try {
    const data = await patchMe(dept.value);
    info.value = data;
    auth.setSession({
      token: auth.token,
      id: data.id,
      username: data.username,
      role: data.role,
      dept: data.dept || "",
    });
    deptOk.value = data.dept ? `科室已设为 ${data.dept}` : "科室已清空（可见公开文献）";
  } catch (e) {
    error.value = e.message || String(e);
  } finally {
    deptBusy.value = false;
  }
}

async function onSubmit() {
  error.value = "";
  ok.value = false;
  if (newPassword.value.length < 6) {
    error.value = "新密码至少 6 位";
    return;
  }
  if (newPassword.value !== confirmPassword.value) {
    error.value = "两次输入的新密码不一致";
    return;
  }
  busy.value = true;
  try {
    await changePassword(oldPassword.value, newPassword.value);
    oldPassword.value = "";
    newPassword.value = "";
    confirmPassword.value = "";
    ok.value = true;
  } catch (e) {
    error.value = e.message || String(e);
  } finally {
    busy.value = false;
  }
}
</script>
