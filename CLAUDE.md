# 草本通 — Claude Code 入口

Claude Code 自动注入的是仓库根目录 **`CLAUDE.md`**（或 `.claude/CLAUDE.md`），**不会**读 `.claude.md`。

项目约定的唯一正文：

@AGENTS.md

路径专项在 `.claude/rules/`（带 `paths`，按正在编辑的文件懒加载）。本机路径用 `CLAUDE.local.md`。
