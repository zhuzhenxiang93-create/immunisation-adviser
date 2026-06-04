# Live Demo Script
## IMAC Immunisation Guidelines Adviser Agent
### Duration: 2 minutes 30 seconds

---

## Pre-Demo Checklist（演示前必做）

- [ ] 启动系统：`conda run -n immunisation-adviser python start.py`
- [ ] 浏览器打开 `http://127.0.0.1:8000`，**不要提前登录**（登录要现场演示）
- [ ] 另一个标签页打开 `http://127.0.0.1:8000/dashboard.html`
- [ ] **提前跑一次热身查询**（第一次因加载 3633 个 chunk 需要 ~8s，之后约 5s）
- [ ] 备用截图准备好：登录界面、查询结果、PII 拦截、dashboard

---

## Demo Flow（6 个场景，共 2 分 30 秒）

| # | 场景 | 时长 | 展示功能 |
|---|---|---|---|
| 1 | 登录 | 0:00–0:20 | Authentication，JWT，多用户 |
| 2 | 标准查询 + 引用 | 0:20–1:10 | 核心 RAG，Source transparency，Confidence |
| 3 | 分类徽章 + 历史记录 | 1:10–1:30 | 6维分类，查询历史，Auditability |
| 4 | Not found 升级 | 1:30–1:55 | Accuracy over recall，Clinical safety |
| 5 | PII 拦截 | 1:55–2:15 | Privacy by design |
| 6 | Dashboard 统计 | 2:15–2:30 | Call insights，Reporting |

---

## 场景 1 — 登录（0:00–0:20）

**操作：** 展示登录弹窗，输入账号密码

**说：**
> "The system requires authentication. Each advisor has their own account — queries, history, and audit logs are tracked per user. This supports accountability and means IMAC can review any interaction if a clinical concern is raised."

**要点：**
- 登录成功后 JWT token 自动保存
- 展示多用户意味着每个顾问的历史是独立的

---

## 场景 2 — 标准查询 + 引用（0:20–1:10）

**输入查询：**
> When should the MMR vaccine be given to a 12-month-old child in New Zealand?

**等待时（~4–5 秒）说：**
> "Notice the progress indicator — the system is searching the knowledge base, then analysing sections, then generating the cited response. The whole pipeline from query to answer in under five seconds."

**答案出来后，依次指出：**

1. **置信度徽章（HIGH，绿色）**
   > "The confidence badge — high means the answer is explicitly stated in the retrieved sections. Not inferred, not extrapolated. Directly quoted."

2. **引用卡片里的 excerpt（原文引用）**
   > "Every citation includes a verbatim excerpt from the source. The advisor can verify the answer without leaving the screen."

3. **来源 + 章节 + URL**
   > "NZ Immunisation Handbook — the exact chapter, section number, and a clickable URL. Clear provenance for every claim."

4. **答案最后一行**
   > "And every single response ends with: 'Final clinical decisions remain with the qualified advisor.' This is a hard rule in the system prompt — the LLM cannot override it."

---

## 场景 3 — 分类徽章 + 历史记录（1:10–1:30）

**仍在刚才的查询结果页，指向分类徽章：**

> "The system automatically classifies every query across six dimensions: vaccine type, query type, clinical scenario, caller type, patient age group, and urgency. Rule-based, zero latency, zero extra API cost."

**指向左侧历史记录列表：**
> "Every query is saved to the history panel on the left — the advisor can click any previous query to review the full response and citations. This is the audit trail: every question asked, every source retrieved, every answer generated."

---

## 场景 4 — Not Found / 升级（1:30–1:55）

**输入查询：**
> A patient developed a firm lump at the hepatitis B injection site that has persisted for 3 months. What is the recommended management?

**等待时说：**
> "This is a specific adverse event management question. Let's see how the system handles a gap in the approved guidance."

**答案出来后（not_found）：**
> "The system cannot find a direct answer — and it says so clearly. It does not fabricate a plausible-sounding clinical response."

**指向 not_found 徽章和升级语：**
> "'I could not find a clear answer in the approved guidance. Please consult the relevant handbook section or escalate to a senior advisor.'"

> "This is accuracy over recall. In a clinical setting, a confident wrong answer is far more dangerous than an honest 'I don't know.' The system is designed to fail safely."

---

## 场景 5 — PII 拦截（1:55–2:15）

**输入查询：**
> My patient's phone number is 021-555-1234. They have an egg allergy — can they receive the influenza vaccine?

**在输入后立刻说（不等结果）：**
> "This query contains a NZ phone number — patient PII. Watch what happens."

**错误出现（HTTP 400）后：**
> "Blocked instantly — before reaching the retrieval engine, before reaching the LLM, before anything is logged. The system identifies the PII type and rejects the query."

> "The advisor is told to resubmit without the identifier. The clinical question — egg allergy and influenza — can still be answered. Only the personal detail is excluded."

> "Privacy by design. Not a policy document — a technical control."

---

## 场景 6 — Dashboard 统计（2:15–2:30）

**切换到 Dashboard 标签页（`http://127.0.0.1:8000/dashboard.html`）：**

> "Finally — the analytics dashboard. This aggregates all queries across all users: which vaccines generate the most questions, which query types appear most often, confidence distribution, and daily call volume."

**快速指向图表：**
- 柱状图：Top vaccine types
- 柱状图：What advisors get asked most（query types）
- 置信度分布

> "For IMAC management, this is the reporting layer — identifying gaps in guidance coverage, peak query periods, and which topics need more advisor training. This data is what makes the system valuable beyond just answering individual queries."

---

## 过渡到评估结果（2:30）

> "So in two and a half minutes we've demonstrated: authenticated access, cited clinical guidance, query classification, audit history, safe escalation, PII protection, and aggregated analytics. Let me now show you how we've measured this systematically against 35 labelled questions."

*(切换到 Slide 6 — Evaluation Results)*

---

## 每个场景覆盖的评分项

| 评分维度 | 场景1 登录 | 场景2 查询 | 场景3 分类/历史 | 场景4 Not found | 场景5 PII | 场景6 Dashboard |
|---|---|---|---|---|---|---|
| Value to use case | | ✅ 核心问题解决 | | | | ✅ 报告价值 |
| Source transparency | | ✅ Excerpt + URL | | | | |
| Confidence indicator | | ✅ HIGH | | ✅ NOT_FOUND | | |
| Query classification | | | ✅ 6维分类 | | | |
| Auditability | ✅ 多用户 | | ✅ 历史记录 | ✅ 记录升级 | ✅ PII 不存储 | |
| Accuracy over recall | | | | ✅ 安全升级 | | |
| Privacy by design | ✅ JWT | | | | ✅ 硬性拦截 | |
| Clinical safety first | | ✅ 结尾语 | | ✅ 固定升级语 | | |
| Reporting / stretch goal | | | | | | ✅ 统计图表 |

---

## 备用截图（网络故障应急）

准备以下截图：
1. 登录弹窗 + 成功后主界面
2. MMR 查询的 high confidence 回答（引用展开）
3. Not found 回答（升级语可见）
4. PII 拦截错误信息
5. Dashboard 完整截图

**话术：**
> "We've prepared screenshots of each scenario — the system was tested this morning and all features are confirmed working."
