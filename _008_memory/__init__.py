#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""企业级 Agent 记忆系统。

分层存储：
  - 短期记忆：Redis 滑动窗口（最近 N 轮对话）
  - 长期记忆：LangChain VectorStore（三元组事实）
  - 用户画像：结构化偏好（Postgres/Mongo 抽象）

核心：Token 预算管理 + 实体遗忘（信息冲突解决）。
"""
