"use client";

import { useEffect, useRef, useState } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  MarkerType,
  MiniMap,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  Briefcase,
  Building2,
  Loader2,
  RefreshCw,
  Search,
  X,
} from "lucide-react";
import { getSubgraph, searchNodes } from "@/lib/api";
import type { NodeSearchResult, Subgraph } from "@/lib/types";

type KgData = {
  label: string;
  type: string;
  properties: Record<string, unknown>;
};

const TYPE_CONFIG: Record<
  string,
  { color: string; shape: "rect" | "pill" | "circle" | "tag"; icon?: typeof Briefcase }
> = {
  Job: { color: "#60a5fa", shape: "rect", icon: Briefcase },
  Skill: { color: "#10b981", shape: "pill" },
  Candidate: { color: "#a855f7", shape: "circle" },
  Company: { color: "#22d3ee", shape: "rect", icon: Building2 },
  JobCategory: { color: "#f59e0b", shape: "tag" },
  SeniorityLevel: { color: "#06b6d4", shape: "pill" },
  Responsibility: { color: "#94a3b8", shape: "rect" },
};

function colorOf(type: string) {
  return TYPE_CONFIG[type]?.color ?? "#94a3b8";
}

const initials = (s: string) => s.trim().slice(0, 2);

function skillColor(p: Record<string, unknown>) {
  const v = String(p["熟练度"] ?? p["level"] ?? p["Level"] ?? "").toLowerCase();
  if (/(精通|expert|advanced|高级|资深)/.test(v)) return "#10b981";
  if (/(熟练|intermediate|中级|proficient)/.test(v)) return "#f59e0b";
  return "#94a3b8";
}

function JobSalary(p: Record<string, unknown>) {
  const v = p["薪资"] ?? p["salary"] ?? p["salary_range"];
  if (!v) return null;
  return (
    <div className="mt-0.5 font-mono text-[10px] text-muted">{String(v)}</div>
  );
}

function KgNode({ data, selected }: NodeProps) {
  const d = data as unknown as KgData;
  const t = d.type;
  const cfg = TYPE_CONFIG[t] ?? TYPE_CONFIG.Responsibility;

  if (t === "Candidate") {
    return (
      <div className="group relative flex flex-col items-center">
        <Handle
          type="target"
          position={Position.Left}
          style={{ background: cfg.color, width: 7, height: 7 }}
        />
        <div
          className="flex h-12 w-12 items-center justify-center rounded-full border-2 bg-gradient-to-br from-accent-600 to-accent-2 text-sm font-semibold text-white transition-all"
          style={{
            borderColor: selected ? "var(--color-accent-400)" : "var(--color-accent-500)",
            boxShadow: selected
              ? "0 0 0 4px color-mix(in oklab, var(--color-accent-500) 35%, transparent), 0 0 28px color-mix(in oklab, var(--color-accent-500) 55%, transparent)"
              : "0 0 20px color-mix(in oklab, var(--color-accent-500) 35%, transparent)",
          }}
        >
          {initials(d.label)}
        </div>
        <span className="mt-1.5 max-w-[120px] truncate text-[10px] text-muted">
          {d.label}
        </span>
        <Handle
          type="source"
          position={Position.Right}
          style={{ background: cfg.color, width: 7, height: 7 }}
        />
      </div>
    );
  }

  const isPill = cfg.shape === "pill" || cfg.shape === "tag";
  const color = t === "Skill" ? skillColor(d.properties) : cfg.color;
  const Icon = cfg.icon;

  return (
    <div
      className={`px-3 py-2 text-xs shadow-md transition-shadow ${
        isPill ? "rounded-full" : "rounded-xl"
      }`}
      style={{
        border: `1px solid ${selected ? "var(--color-accent-400)" : `${color}55`}`,
        background: selected
          ? `linear-gradient(135deg, ${color}44, ${color}11)`
          : `linear-gradient(135deg, ${color}2e, oklch(0.19 0.015 265 / 0.9))`,
        boxShadow: selected
          ? `0 0 0 3px color-mix(in oklab, var(--color-accent-500) 25%, transparent)`
          : undefined,
      }}
    >
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: color, width: 7, height: 7 }}
      />
      <div className="flex items-center gap-1.5">
        {Icon && <Icon className="h-3.5 w-3.5 shrink-0" style={{ color }} />}
        <span className="max-w-[160px] truncate font-medium text-ink">
          {d.label}
        </span>
      </div>
      {t === "Job" && <JobSalary {...d.properties} />}
      <Handle
        type="source"
        position={Position.Right}
        style={{ background: color, width: 7, height: 7 }}
      />
    </div>
  );
}

const nodeTypes = { kg: KgNode };

/** 确定性分层布局：以度数最大的节点为根做 BFS 分层，按层从左到右排列（无需外部 worker） */
function layoutGraph(sub: Subgraph): { nodes: Node[]; edges: Edge[] } {
  const adj = new Map<string, string[]>();
  for (const n of sub.nodes) adj.set(n.id, []);
  for (const e of sub.edges) {
    adj.get(e.source)?.push(e.target);
    adj.get(e.target)?.push(e.source);
  }

  let root = sub.nodes[0]?.id ?? "";
  let maxDeg = -1;
  for (const n of sub.nodes) {
    const d = adj.get(n.id)?.length ?? 0;
    if (d > maxDeg) {
      maxDeg = d;
      root = n.id;
    }
  }

  const depth = new Map<string, number>();
  const queue: string[] = [root];
  depth.set(root, 0);
  while (queue.length) {
    const cur = queue.shift()!;
    const d = depth.get(cur)!;
    for (const nxt of adj.get(cur) ?? []) {
      if (!depth.has(nxt)) {
        depth.set(nxt, d + 1);
        queue.push(nxt);
      }
    }
  }
  for (const n of sub.nodes) if (!depth.has(n.id)) depth.set(n.id, 0);

  const layers = new Map<number, string[]>();
  for (const n of sub.nodes) {
    const d = depth.get(n.id)!;
    if (!layers.has(d)) layers.set(d, []);
    layers.get(d)!.push(n.id);
  }

  const LAYER_GAP_X = 240;
  const NODE_GAP_Y = 72;
  const pos = new Map<string, { x: number; y: number }>();
  for (const [d, ids] of layers) {
    const totalH = (ids.length - 1) * NODE_GAP_Y;
    ids.forEach((id, i) => {
      pos.set(id, { x: d * LAYER_GAP_X, y: i * NODE_GAP_Y - totalH / 2 });
    });
  }

  const nodes: Node[] = sub.nodes.map((n) => {
    const p = pos.get(n.id) ?? { x: 0, y: 0 };
    return {
      id: n.id,
      position: p,
      data: {
        label: n.label,
        type: n.type,
        properties: n.properties ?? {},
      } satisfies KgData,
      type: "kg",
    };
  });

  const edges: Edge[] = sub.edges.map((e, i) => ({
    id: `e${i}`,
    source: e.source,
    target: e.target,
    label: e.type,
    type: "default",
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: "rgba(255,255,255,0.28)",
      width: 14,
      height: 14,
    },
    style: { stroke: "rgba(255,255,255,0.16)", strokeWidth: 1.4 },
    labelStyle: {
      fill: "rgba(190,190,215,0.85)",
      fontSize: 8,
      fontFamily: "JetBrains Mono, ui-monospace, monospace",
    },
    labelBgStyle: { fill: "oklch(0.16 0.012 265)", fillOpacity: 0.92 },
    labelBgPadding: [4, 1] as [number, number],
    labelBgBorderRadius: 3,
  }));

  return { nodes, edges };
}

export default function GraphExplorer() {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [depth, setDepth] = useState(2);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<NodeSearchResult[]>([]);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<{
    id: string;
    data: KgData;
  } | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const load = async (nodeId: string) => {
    setLoading(true);
    setError("");
    setQuery("");
    setResults([]);
    setSelected(null);
    try {
      const sub = await getSubgraph(nodeId, depth, 80);
      if (sub.nodes.length === 0) {
        setError(`未找到节点「${nodeId}」`);
        setNodes([]);
        setEdges([]);
      } else {
        const laid = layoutGraph(sub);
        setNodes(laid.nodes);
        setEdges(laid.edges);
      }
    } catch {
      setError("图谱数据加载失败，请确认后端已就绪");
    } finally {
      setLoading(false);
    }
  };

  // 初始以候选人「张三」为中心，展示 候选人→技能→岗位 的匹配网络
  useEffect(() => {
    load("张三");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onQuery = (q: string) => {
    setQuery(q);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (!q.trim()) {
      setResults([]);
      return;
    }
    debounceRef.current = setTimeout(async () => {
      try {
        setResults(await searchNodes(q));
      } catch {
        setResults([]);
      }
    }, 250);
  };

  const selCfg = selected ? TYPE_CONFIG[selected.data.type] : null;

  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 items-center gap-2 border-b border-line bg-surface px-4 py-3">
        <div className="relative w-64">
          <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-faint" />
          <input
            value={query}
            onChange={(e) => onQuery(e.target.value)}
            placeholder="搜索节点…（岗位/技能/候选人）"
            className="w-full rounded-lg border border-line bg-panel py-1.5 pl-8 pr-8 text-xs outline-none transition-colors focus:border-accent-400/50"
          />
          {query && (
            <button
              onClick={() => {
                setQuery("");
                setResults([]);
              }}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-faint hover:text-ink"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
          {results.length > 0 && (
            <div className="absolute left-0 top-full z-20 mt-1 w-full overflow-hidden rounded-lg border border-line bg-panel shadow-xl">
              {results.map((r) => (
                <button
                  key={r.id}
                  onClick={() => load(r.name)}
                  className="flex w-full items-center justify-between px-3 py-2 text-left text-xs transition-colors hover:bg-white/5"
                >
                  <span className="truncate text-ink">{r.name}</span>
                  <span
                    className="ml-2 shrink-0 text-[10px]"
                    style={{ color: colorOf(r.label) }}
                  >
                    {r.label}
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="ml-2 flex items-center gap-1 rounded-lg border border-line bg-panel p-0.5">
          {[1, 2, 3].map((d) => (
            <button
              key={d}
              onClick={() => setDepth(d)}
              className={`rounded-md px-2.5 py-1 text-xs transition-colors ${
                depth === d ? "bg-accent-600 text-white" : "text-muted hover:text-ink"
              }`}
            >
              {d} 跳
            </button>
          ))}
        </div>

        <button
          onClick={() => load("张三")}
          className="ml-1 flex items-center gap-1.5 rounded-lg border border-line bg-panel px-2.5 py-1.5 text-xs text-muted transition-colors hover:text-ink"
        >
          <RefreshCw className="h-3.5 w-3.5" /> 重置
        </button>

        <span className="ml-auto flex items-center gap-2 text-[11px] text-faint">
          <span className="font-mono">{nodes.length} 节点</span>·
          <span className="font-mono">{edges.length} 关系</span>
        </span>
      </div>

      <div className="flex shrink-0 flex-wrap gap-x-3 gap-y-1 border-b border-line bg-surface px-4 py-2">
        {Object.entries(TYPE_CONFIG).map(([t, cfg]) => (
          <span key={t} className="flex items-center gap-1 text-[10px] text-muted">
            <span className="h-2 w-2 rounded-full" style={{ background: cfg.color }} /> {t}
          </span>
        ))}
      </div>

      <div className="graph-dark relative min-h-0 flex-1">
        {loading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-canvas/60">
            <Loader2 className="h-6 w-6 animate-spin text-accent-400" />
          </div>
        )}
        {error ? (
          <div className="flex h-full items-center justify-center px-6 text-center text-sm text-muted">
            {error}
          </div>
        ) : (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.2 }}
            minZoom={0.2}
            maxZoom={2}
            onNodeClick={(_, node) =>
              setSelected({ id: node.id, data: node.data as unknown as KgData })
            }
            onNodeDoubleClick={(_, node) => load(node.id)}
            proOptions={{ hideAttribution: true }}
          >
            <Background
              variant={BackgroundVariant.Dots}
              gap={24}
              size={1}
              color="rgba(255,255,255,0.07)"
            />
            <Controls showInteractive={false} />
            <MiniMap
              pannable
              zoomable
              nodeColor={(n) => colorOf((n.data as unknown as KgData).type)}
            />
          </ReactFlow>
        )}

        {/* 节点详情抽屉（从右侧滑入） */}
        <div
          className={`absolute inset-y-0 right-0 z-20 flex w-80 flex-col border-l border-line bg-surface/95 shadow-2xl backdrop-blur-xl transition-transform duration-300 ${
            selected ? "translate-x-0" : "translate-x-full"
          }`}
        >
          {selected && selCfg && (
            <>
              <div className="flex h-12 shrink-0 items-center gap-2 border-b border-line px-4">
                <span
                  className="rounded-md px-2 py-0.5 text-[11px]"
                  style={{ background: `${selCfg.color}22`, color: selCfg.color }}
                >
                  {selected.data.type}
                </span>
                <span className="truncate text-sm font-medium text-ink">
                  {selected.data.label}
                </span>
                <button
                  onClick={() => setSelected(null)}
                  className="ml-auto flex h-7 w-7 items-center justify-center rounded-lg text-faint transition-colors hover:bg-white/5 hover:text-ink"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
              <div className="min-h-0 flex-1 overflow-y-auto p-4">
                <p className="text-[11px] uppercase tracking-wide text-faint">节点 ID</p>
                <p className="mt-0.5 font-mono text-xs text-muted">{selected.id}</p>
                <p className="mt-4 text-[11px] uppercase tracking-wide text-faint">
                  属性
                </p>
                <div className="mt-2 space-y-1.5">
                  {Object.entries(selected.data.properties ?? {}).length === 0 && (
                    <p className="text-xs text-faint">无附加属性</p>
                  )}
                  {Object.entries(selected.data.properties ?? {}).map(([k, v]) => (
                    <div key={k} className="flex items-start gap-2 text-xs">
                      <span className="shrink-0 text-faint">{k}</span>
                      <span className="ml-auto break-all text-right text-muted">
                        {String(v)}
                      </span>
                    </div>
                  ))}
                </div>
                <p className="mt-4 rounded-lg border border-line bg-panel/60 px-3 py-2 text-[11px] leading-relaxed text-faint">
                  双击节点可将其设为中心，展开邻域；单击选中查看详情。
                </p>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}