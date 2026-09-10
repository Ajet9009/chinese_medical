import { createRouter, createWebHistory } from "vue-router";
import { useAuthStore } from "./stores/auth.js";
import Login from "./views/Login.vue";
import AppLayout from "./views/AppLayout.vue";
import Chat from "./views/Chat.vue";
import Admin from "./views/Admin.vue";
import Profile from "./views/Profile.vue";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/login", name: "login", component: Login, meta: { public: true } },
    {
      path: "/",
      component: AppLayout,
      children: [
        { path: "", redirect: "/chat" },
        { path: "chat", name: "chat", component: Chat },
        { path: "profile", name: "profile", component: Profile },
        { path: "admin", name: "admin", component: Admin, meta: { admin: true } },
      ],
    },
  ],
});

router.beforeEach((to) => {
  const auth = useAuthStore();
  if (to.meta.public) {
    if (auth.isLoggedIn && to.path === "/login") return { path: "/chat" };
    return true;
  }
  if (!auth.isLoggedIn) return { path: "/login", query: { redirect: to.fullPath } };
  if (to.meta.admin && !auth.isAdmin) return { path: "/chat" };
  return true;
});

export default router;
