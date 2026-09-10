import { computed, ref } from "vue";
import { defineStore } from "pinia";

const TOKEN_KEY = "cm_token";
const USER_KEY = "cm_user";

function loadUser() {
  try {
    return JSON.parse(localStorage.getItem(USER_KEY) || "null") || {};
  } catch {
    return {};
  }
}

export const useAuthStore = defineStore("auth", () => {
  const token = ref(localStorage.getItem(TOKEN_KEY) || "");
  const userId = ref(loadUser().id || "");
  const username = ref(loadUser().username || "");
  const role = ref(loadUser().role || "");

  const isAdmin = computed(() => role.value === "admin");
  const isLoggedIn = computed(() => Boolean(token.value));

  function setSession({ token: nextToken, id, username: name, role: nextRole }) {
    token.value = nextToken || "";
    userId.value = id || "";
    username.value = name || "";
    role.value = nextRole || "";
    if (token.value) localStorage.setItem(TOKEN_KEY, token.value);
    else localStorage.removeItem(TOKEN_KEY);
    localStorage.setItem(
      USER_KEY,
      JSON.stringify({ id: userId.value, username: username.value, role: role.value })
    );
  }

  function clear() {
    setSession({ token: "", id: "", username: "", role: "" });
    localStorage.removeItem(USER_KEY);
  }

  return { token, userId, username, role, isAdmin, isLoggedIn, setSession, clear };
});
