<template>
  <div class="login-page">
    <div class="login-bg">
      <div class="blob b1"></div>
      <div class="blob b2"></div>
    </div>
    <form class="login-card" @submit.prevent="onSubmit">
      <div class="login-head">
        <div class="brand-seal">草</div>
        <h1>草<span>本</span>通</h1>
      </div>
      <p class="login-sub">中医知识图谱问答。问方、问药、问证，多轮追问仍接上文。</p>
      <div class="field">
        <label class="field-label">用户名</label>
        <input class="input" v-model="username" autocomplete="username" placeholder="请输入用户名" />
      </div>
      <div class="field">
        <label class="field-label">密码</label>
        <input
          class="input"
          v-model="password"
          type="password"
          autocomplete="current-password"
          placeholder="请输入密码"
        />
      </div>
      <button class="btn" type="submit" :disabled="busy">{{ busy ? "登录中…" : "🌿 入室" }}</button>
      <div v-if="error" class="err">{{ error }}</div>
      <p class="hint">由医师管理员开方配帐，不开放自行注册。</p>
      <div class="login-feats">
        <span class="badge badge-primary">知识图谱</span>
        <span class="badge badge-info">多轮问诊</span>
        <span class="badge badge-success">方药检索</span>
      </div>
    </form>
  </div>
</template>

<script setup>
import { ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import { login as loginApi } from "../api.js";
import { useAuthStore } from "../stores/auth.js";

const auth = useAuthStore();
const router = useRouter();
const route = useRoute();
const username = ref("");
const password = ref("");
const busy = ref(false);
const error = ref("");

async function onSubmit() {
  error.value = "";
  busy.value = true;
  try {
    const data = await loginApi(username.value.trim(), password.value);
    auth.setSession(data);
    const next = typeof route.query.redirect === "string" ? route.query.redirect : "/chat";
    router.replace(next.startsWith("/") ? next : "/chat");
  } catch (e) {
    error.value = e.message || "登录失败";
  } finally {
    busy.value = false;
  }
}
</script>
