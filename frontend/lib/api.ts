import type {
  ChatEvent,
  GapResult,
  Health,
  NodeSearchResult,
  Stats,
  Subgraph,
  WhatIfResult,
} from "./types";

const BASE = "/api";

export async function getHealth(): Promise<Health> {
  const res = await fetch(`${BASE}/health`, { cache: "no-store" });
  return res.json();
}

export async function getStats(): Promise<Stats> {
  const res = await fetch(`${BASE}/stats`, { cache: "no-store" });
  return res.json();
}

export async function searchNodes(
  query: string,
  limit = 8,
): Promise<NodeSearchResult[]> {
  const res = await fetch(
    `${BASE}/graph/search?query=${encodeURIComponent(query)}&limit=${limit}`,
  );
  return res.json();
}

export async function getSubgraph(
  nodeId: string,
  maxDepth = 2,
  maxNodes = 80,
): Promise<Subgraph> {
  const res = await fetch(
    `${BASE}/graph/subgraph?node_id=${encodeURIComponent(nodeId)}&max_depth=${maxDepth}&max_nodes=${maxNodes}`,
  );
  return res.json();
}

export async function getGap(candidate: string, job: string): Promise<GapResult> {
  const res = await fetch(`${BASE}/gap`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ candidate, job }),
  });
  return res.json();
}

export async function getWhatIf(
  candidate: string,
  acquiredSkills: string[],
): Promise<WhatIfResult> {
  const res = await fetch(`${BASE}/gap/whatif`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ candidate, acquired_skills: acquiredSkills }),
  });
  return res.json();
}

export async function buildKnowledgeBase(force = false): Promise<{
  status: string;
  stats?: unknown;
  message?: string;
}> {
  const res = await fetch(`${BASE}/admin/build`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ force }),
  });
  return res.json();
}

/** 通过 fetch + ReadableStream 消费后端 POST /api/chat 的 SSE 流 */
export async function streamChat(
  query: string,
  onEvent: (evt: ChatEvent) => void,
  signal?: AbortSignal,
  topK = 5,
): Promise<void> {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      for (const line of raw.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") return;
        try {
          onEvent(JSON.parse(payload) as ChatEvent);
        } catch {
          // 忽略非法帧
        }
      }
    }
  }
}