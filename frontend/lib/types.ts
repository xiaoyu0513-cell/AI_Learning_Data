// 后端 API 类型定义

export type RouteDecision = {
  strategy: string;
  confidence: number;
  complexity: number;
  relationship: number;
  reasoning_required: boolean;
  entity_count: number;
  reasoning: string;
};

export type Citation = {
  index: number;
  source: string;
  node_id: string;
  job_name: string;
  strategy: string;
  score: number;
};

export type ChatEvent =
  | { type: "status"; content: string }
  | { type: "route_decision"; content: RouteDecision }
  | { type: "citations"; content: Citation[] }
  | { type: "token"; content: string }
  | { type: "done" }
  | { type: "error"; content: string };

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  streaming?: boolean;
  error?: boolean;
};

export type GraphNode = {
  id: string;
  label: string;
  type: string;
  properties: Record<string, unknown>;
};

export type GraphEdge = {
  source: string;
  target: string;
  type: string;
};

export type Subgraph = {
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export type NodeSearchResult = {
  name: string;
  label: string;
  id: number;
};

export type Health = {
  status: string;
  ready: boolean;
  milvus_rows: number;
};

export type Stats = {
  graph: Record<string, number>;
  milvus: { row_count: number };
  routing: Record<string, unknown>;
};

export type LearningPath = {
  skill: string;
  prerequisites: string[];
};

export type GapResult = {
  candidate: string;
  job: string;
  match_rate: number;
  matched_skills: string[];
  missing_skills: string[];
  learning_paths: LearningPath[];
  recommendation: string;
  error?: string;
};

export type WhatIfOpportunity = {
  job_id: string;
  job_name: string;
  match_rate: number;
  newly_matched: string[];
};

export type WhatIfResult = {
  candidate: string;
  acquired_skills: string[];
  new_opportunities: WhatIfOpportunity[];
  error?: string;
};