<template>
  <div class="login-page">
    <form class="login-card" @submit.prevent="onSubmit">
      <div class="login-brand">
        <div class="brand-seal">草</div>
        <h1>草本通</h1>
        <p class="login-sub">使用医师管理员配发的账号登录。不开放自行注册。</p>
      </div>
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
      <button class="btn" type="submit" :disabled="busy">{{ busy ? "登录中…" : "登录" }}</button>
      <div v-if="error" class="err">{{ error }}</div>
      <p class="hint">忘记密码时联系管理员重置，不要在公共设备保存会话。</p>
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
