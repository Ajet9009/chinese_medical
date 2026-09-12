<template>
  <div class="app-frame">
    <header class="auth-bar">
      <div class="brand-lockup">
        <div class="brand-seal">草</div>
        <div class="auth-brand">草<span>本</span>通<small>中医知识图谱</small></div>
      </div>
      <nav class="auth-nav">
        <router-link class="nav-pill" to="/chat">💬 问答</router-link>
        <router-link class="nav-pill" to="/documents">📚 知识库</router-link>
        <router-link v-if="auth.isAdmin" class="nav-pill" to="/knowledge-governance">🧭 知识治理</router-link>
        <router-link v-if="auth.isAdmin" class="nav-pill" to="/admin">⚙️ 管理</router-link>
      </nav>
      <div class="auth-user">
        <router-link class="profile-link" to="/profile" title="账号与密码">
          <div class="avatar">{{ initial }}</div>
          <span>{{ auth.username }} <span class="muted">· {{ roleLabel }}</span></span>
        </router-link>
        <button class="btn ghost sm" type="button" @click="logout">🚪 退出</button>
      </div>
    </header>
    <div class="app-body">
      <router-view />
    </div>
  </div>
</template>

<script setup>
import { computed } from "vue";
import { useRouter } from "vue-router";
import { useAuthStore } from "../stores/auth.js";

const auth = useAuthStore();
const router = useRouter();
const initial = computed(() => (auth.username || "用").slice(0, 1));
const roleLabel = computed(() => (auth.role === "admin" ? "管理员" : "医师"));

function logout() {
  auth.clear();
  router.push("/login");
}
</script>
