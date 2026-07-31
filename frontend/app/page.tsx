"use client";

/**
 * app/page.tsx
 * ------------
 * Живой трейс мультиагентного графа поверх SSE от api_server.py (/chat).
 *
 * Классификация шага в узел графа (nodeKeyFromLabel) сделана эвристикой
 * по подстрокам, т.к. я не видел agents.py / critic.py / generate.py —
 * только supervisor.py, main.py, graph.py, state.py. Судя по main.py
 * ("--- Путь агентов ---": "supervisor→data(forced) → data(sql) →
 * supervisor→finish → generate → critic:ok") эвристика должна покрыть
 * все реальные форматы. Если какой-то узел не подсвечивается — пришли
 * мне точные строки step из agents.py/critic.py, поправлю за 1 минуту.
 */

import { useRef, useState } from "react";

type NodeKey =
  | "supervisor"
  | "retriever"
  | "web"
  | "data"
  | "code"
  | "generate"
  | "critic";

type NodeState = "idle" | "active" | "done" | "approved" | "revise";

type StepEvent = {
  label: string;
  steps_so_far: string[];
  sql_result?: string | null;
  code_result?: string | null;
  plan?: string | null;
  approved?: boolean | null;
  critic_reason?: string | null;
  revisions: number;
};

type FinalEvent = {
  answer: string;
  steps: string[];
  revisions: number;
  approved?: boolean | null;
};

const NODES: Record<NodeKey, { x: number; y: number; label: string }> = {
  supervisor: { x: 320, y: 70, label: "SUPERVISOR" },
  retriever: { x: 90, y: 190, label: "RETRIEVER" },
  web: { x: 250, y: 190, label: "WEB" },
  data: { x: 390, y: 190, label: "DATA · SQL" },
  code: { x: 550, y: 190, label: "CODE" },
  generate: { x: 320, y: 300, label: "GENERATE" },
  critic: { x: 320, y: 400, label: "CRITIC" },
};

const SPECIALISTS: NodeKey[] = ["retriever", "web", "data", "code"];

function extractSupervisorTarget(label: string): NodeKey | null {
  // "supervisor→data(forced)" -> "data" | "supervisor→finish" -> "generate"
  const match = label.match(/supervisor.*?(retriever|web|data|code|finish)/i);
  if (!match) return null;
  const token = match[1].toLowerCase();
  if (token === "finish") return "generate";
  return token as NodeKey;
}

function nodeKeyFromLabel(label: string): NodeKey | null {
  const l = label.toLowerCase();
  if (l.includes("critic")) return "critic";
  if (l.includes("generate")) return "generate";
  if (l.includes("retriever")) return "retriever";
  if (/\bweb\b/.test(l)) return "web";
  if (l.includes("supervisor")) return "supervisor";
  if (l.includes("code")) return "code";
  if (l.includes("data") || l.includes("sql")) return "data";
  return null;
}

export default function Home() {
  const [question, setQuestion] = useState("");
  const [log, setLog] = useState<string[]>([]);
  const [nodeStates, setNodeStates] = useState<Record<NodeKey, NodeState>>({
    supervisor: "idle",
    retriever: "idle",
    web: "idle",
    data: "idle",
    code: "idle",
    generate: "idle",
    critic: "idle",
  });
  const [activeEdges, setActiveEdges] = useState<Set<string>>(new Set());
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [revisions, setRevisions] = useState(0);
  const abortRef = useRef<AbortController | null>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  function markEdge(a: NodeKey, b: NodeKey) {
    setActiveEdges((prev) => new Set(prev).add([a, b].sort().join("|")));
  }

  function setNode(key: NodeKey, state: NodeState) {
    setNodeStates((prev) => ({ ...prev, [key]: state }));
  }

  function resetGraph() {
    setNodeStates({
      supervisor: "idle",
      retriever: "idle",
      web: "idle",
      data: "idle",
      code: "idle",
      generate: "idle",
      critic: "idle",
    });
    setActiveEdges(new Set());
    setRevisions(0);
  }

  function applyStep(evt: StepEvent) {
    const key = nodeKeyFromLabel(evt.label);
    if (!key) return;

    // предыдущий активный узел -> done, новый -> active
    setNodeStates((prev) => {
      const next: Record<NodeKey, NodeState> = { ...prev };
      (Object.keys(next) as NodeKey[]).forEach((k) => {
        if (next[k] === "active") next[k] = "done";
      });
      next[key] = "active";
      return next;
    });

    if (key === "supervisor") {
      const target = extractSupervisorTarget(evt.label);
      if (target) markEdge("supervisor", target);
    }
    if (SPECIALISTS.includes(key)) {
      markEdge("supervisor", key);
    }
    if (key === "generate") {
      markEdge("supervisor", "generate");
      markEdge("generate", "critic");
    }
    if (key === "critic") {
      markEdge("generate", "critic");
      if (evt.approved === true) {
        setNode("critic", "approved");
      } else if (evt.approved === false) {
        setNode("critic", "revise");
        markEdge("critic", "supervisor");
      }
    }

    setRevisions(evt.revisions ?? 0);
    setLog((prev) => [...prev, evt.label]);
  }

  async function askQuestion() {
    if (!question.trim() || loading) return;

    setLog([]);
    setAnswer("");
    setError(null);
    resetGraph();
    setLoading(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
        signal: controller.signal,
      });
      if (!res.body) throw new Error("Стрим не поддерживается ответом сервера");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const rawEvents = buffer.split("\n\n");
        buffer = rawEvents.pop() || "";

        for (const raw of rawEvents) {
          if (!raw.trim()) continue;
          const lines = raw.split("\n");
          const eventLine = lines.find((l) => l.startsWith("event:"));
          const dataLine = lines.find((l) => l.startsWith("data:"));
          if (!eventLine || !dataLine) continue;

          const eventName = eventLine.replace("event:", "").trim();
          const data = JSON.parse(dataLine.replace("data:", "").trim());

          if (eventName === "step") applyStep(data as StepEvent);
          else if (eventName === "final") setAnswer((data as FinalEvent).answer);
          else if (eventName === "error") setError(data.message);
        }
      }
    } catch (e: any) {
      if (e.name !== "AbortError") setError(e.message || "Ошибка запроса");
    } finally {
      setLoading(false);
    }
  }

  const edgeList: [NodeKey, NodeKey, boolean][] = [
    ...SPECIALISTS.map((s): [NodeKey, NodeKey, boolean] => ["supervisor", s, false]),
    ["supervisor", "generate", false],
    ["generate", "critic", false],
    ["critic", "supervisor", true],
  ];

  return (
    <main style={{ maxWidth: 760, margin: "0 auto", padding: "2.5rem 1.25rem 4rem" }}>
      <header style={{ marginBottom: "1.75rem" }}>
        <div style={{ color: "var(--muted)", fontSize: 12, letterSpacing: 1 }}>
          LANGGRAPH · SUPERVISOR + 4 SPECIALISTS + CRITIC
        </div>
        <h1 style={{ fontSize: 22, margin: "0.25rem 0 0", fontWeight: 700 }}>
          Multi-Agent AI Analyst
        </h1>
      </header>

      {/* ---------- граф ---------- */}
      <div
        style={{
          background: "var(--panel)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
          padding: "1rem",
          marginBottom: "1.25rem",
        }}
      >
        <svg viewBox="0 0 640 460" width="100%" style={{ display: "block" }}>
          {edgeList.map(([a, b, isLoop]) => {
            const active = activeEdges.has([a, b].sort().join("|"));
            const na = NODES[a];
            const nb = NODES[b];
            if (isLoop) {
              // дуга справа от supervisor->critic для петли ревизии
              return (
                <path
                  key={`${a}-${b}`}
                  className="edge"
                  data-active={active}
                  data-loop="true"
                  d={`M ${nb.x + 55} ${nb.y - 10} C 560 340, 560 100, ${na.x + 55} ${na.y + 5}`}
                />
              );
            }
            return (
              <line
                key={`${a}-${b}`}
                className="edge"
                data-active={active}
                x1={na.x}
                y1={na.y}
                x2={nb.x}
                y2={nb.y}
              />
            );
          })}

          {(Object.keys(NODES) as NodeKey[]).map((key) => {
            const n = NODES[key];
            const state = nodeStates[key];
            const w = key === "data" ? 84 : 78;
            return (
              <g key={key} className="node" data-state={state}>
                <rect
                  className="node-shape"
                  x={n.x - w / 2}
                  y={n.y - 18}
                  width={w}
                  height={36}
                  rx={8}
                />
                <text
                  className="node-label"
                  x={n.x}
                  y={n.y + 4}
                  textAnchor="middle"
                >
                  {n.label}
                </text>
              </g>
            );
          })}
        </svg>
        {revisions > 0 && (
          <div style={{ fontSize: 11, color: "var(--warning)", marginTop: 4 }}>
            revisions: {revisions}
          </div>
        )}
      </div>

      {/* ---------- ввод ---------- */}
      <div style={{ display: "flex", gap: 8 }}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && askQuestion()}
          placeholder="Сколько клиентов ушло в Q3?"
          style={{
            flex: 1,
            padding: "0.65rem 0.9rem",
            borderRadius: "var(--radius)",
            border: "1px solid var(--border)",
            background: "var(--panel-2)",
            color: "var(--text)",
            outline: "none",
          }}
        />
        <button
          onClick={askQuestion}
          disabled={loading}
          style={{
            padding: "0.65rem 1.3rem",
            borderRadius: "var(--radius)",
            border: "1px solid var(--accent)",
            background: loading ? "var(--panel-2)" : "transparent",
            color: "var(--accent)",
            cursor: loading ? "default" : "pointer",
            fontWeight: 600,
          }}
        >
          {loading ? "..." : "run →"}
        </button>
      </div>

      {error && (
        <p style={{ color: "var(--error)", marginTop: "1rem", fontSize: 13 }}>
          error: {error}
        </p>
      )}

      {/* ---------- лог шагов, терминальный стиль ---------- */}
      {log.length > 0 && (
        <div
          style={{
            marginTop: "1.25rem",
            background: "#000",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius)",
            padding: "0.85rem 1rem",
            fontSize: 12,
            lineHeight: 1.7,
            color: "#9dd6ff",
            maxHeight: 220,
            overflowY: "auto",
          }}
        >
          {log.map((l, i) => (
            <div key={i}>
              <span style={{ color: "var(--muted)" }}>{String(i + 1).padStart(2, "0")} $ </span>
              {l}
            </div>
          ))}
        </div>
      )}

      {/* ---------- ответ ---------- */}
      {answer && (
        <div
          style={{
            marginTop: "1.25rem",
            padding: "1rem 1.1rem",
            borderRadius: "var(--radius)",
            background: "#0d2b20",
            border: "1px solid var(--success)",
            lineHeight: 1.55,
            fontSize: 14,
          }}
        >
          <div style={{ color: "var(--success)", fontSize: 11, marginBottom: 4 }}>
            ANSWER
          </div>
          {answer}
        </div>
      )}
    </main>
  );
}