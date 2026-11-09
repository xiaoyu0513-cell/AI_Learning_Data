"use client";

import {
  Activity,
  Briefcase,
  CircleDot,
  Database,
  FileText,
  Layers,
  Network,
  Target,
} from "lucide-react";
import { useWorkbench } from "@/lib/store/workbench";
import type { Citation } from "@/lib/types";

const strategyLabel: Record<string, string> = {
  graph_rag: "图 RAG 检索",
  traditional: "向量 + BM25",
  combined: "混合检索（图 + 向量）",
};

const strategyIcon: Record<string, typeof Network> = {
  graph_rag: Network,
  traditional: Database,
  combined: Layers,
};

function SectionTitle({ icon: Icon, text }: { icon: typeof Target; text: string }) {
  return (
    <div className="flex items-center gap-1.5 text-xs font-medium text-muted">
      <Icon className="h-3.5 w-3.5 text-accent-400" />
      {text}
    </div>
  );
}

function Confidence({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const r = 26;
  const c = 2 * Math.PI * r;
  const offset = c * (1 - pct / 100);
  return (
    <div className="flex items-center gap-3">
      <div className="relative h-16 w-16 shrink-0">
        <svg className="h-16 w-16 -rotate-90" viewBox="0 0 64 64">
          <circle
            cx="32"
            cy="32"
            r={r}
            fill="none"
            stroke="oklch(1 0 0 / 0.08)"
            strokeWidth="5"
          />
          <circle
            cx="32"
            cy="32"
            r={r}
            fill="none"
            stroke="url(#conf-grad)"
            strokeWidth="5"
            strokeLinecap="round"
            strokeDasharray={c}
            strokeDashoffset={offset}
            className="transition-all duration-700 ease-out"
          />
          <defs>
            <linearGradient id="conf-grad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#8b5cf6" />
              <stop offset="100%" stopColor="#60a5fa" />
            </linearGradient>
          </defs>
        </svg>
        <span className="absolute inset-0 flex items-center justify-center text-sm font-semibold text-ink">
          {pct}%
        </span>
      </div>
      <div className="text-xs leading-relaxed text-muted">
        置信度
        <div className="text-[10px] text-faint">路由可信度</div>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-white/5 px-2 py-1">
      <div className="text-[10px] text-faint">{label}</div>
      <div className="font-mono text-xs text-ink">{value}</div>
    </div>
  );
}

function CitationCard({ c }: { c: Citation }) {
  const pct = Math.round((c.score ?? 0) * 100);
  const Icon = strategyIcon[c.strategy] ?? FileText;
  return (
    <div className="card p-3">
      <div className="flex items-center gap-2">
        <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded bg-accent-500/15 font-mono text-[10px] text-accent-300">
          {c.index}
        </span>
        <span className="truncate text-xs font-medium text-ink">
          {c.job_name || c.node_id || "—"}
        </span>
        <Icon className="ml-auto h-3.5 w-3.5 shrink-0 text-faint" />
      </div>
      <div className="mt-2 flex items-center gap-2">
        <span className="shrink-0 text-[10px] text-faint">相关性</span>
        <div className="h-1 flex-1 overflow-hidden rounded-full bg-white/10">
          <div
            className="h-full rounded-full bg-gradient-to-r from-accent-500 to-cyan-400"
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className="shrink-0 font-mono text-[10px] text-ink">{pct}%</span>
      </div>
      <div className="mt-2 flex items-center gap-1.5 text-[10px] text-muted">
        <span className="rounded bg-white/5 px-1.5 py-0.5">
          {strategyLabel[c.strategy] ?? c.strategy}
        </span>
        <span className="rounded bg-white/5 px-1.5 py-0.5">{c.source}</span>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-line py-10 text-center">
      <Activity className="h-6 w-6 text-faint" />
      <p className="text-xs text-faint">在左侧发起一次问答，检索过程将在此呈现</p>
    </div>
  );
}

export default function RightPanel() {
  const obs = useWorkbench((s) => s.observability);
  const { route, citations, trace, status, query } = obs;

  return (
    <aside className="flex h-full flex-col border-l border-line bg-surface">
      <div className="flex h-10 shrink-0 items-center gap-2 border-b border-line px-4">
        <Activity className="h-4 w-4 text-accent-400" />
        <span className="text-sm font-medium">检索可观测</span>
        <span className="ml-auto flex items-center gap-1.5 text-[11px] text-muted">
          <CircleDot
            className={`h-3 w-3 ${status === "空闲" ? "text-faint" : "text-emerald-400"}`}
          />
          {status}
        </span>
      </div>

      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
        {query && (
          <section>
            <SectionTitle icon={Target} text="当前查询" />
            <p className="card mt-2 px-3 py-2 text-xs text-muted">{query}</p>
          </section>
        )}

        {route ? (
          <section>
            <SectionTitle icon={Network} text="路由决策" />
            <div className="card mt-2 space-y-3 p-3">
              <Confidence value={route.confidence} />
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted">策略</span>
                <span className="rounded-md bg-accent-500/15 px-2 py-0.5 text-xs text-accent-300">
                  {strategyLabel[route.strategy] ?? route.strategy}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <Metric label="复杂度" value={route.complexity.toFixed(2)} />
                <Metric label="关系密集度" value={route.relationship.toFixed(2)} />
                <Metric label="实体数" value={String(route.entity_count)} />
                <Metric label="需推理" value={route.reasoning_required ? "是" : "否"} />
              </div>
              <p className="border-t border-line pt-2 text-xs leading-relaxed text-muted">
                {route.reasoning}
              </p>
            </div>
          </section>
        ) : (
          <EmptyState />
        )}

        {citations.length > 0 && (
          <section>
            <SectionTitle icon={Briefcase} text={`引用来源 · ${citations.length}`} />
            <div className="mt-2 space-y-2">
              {citations.map((c) => (
                <CitationCard key={c.index} c={c} />
              ))}
            </div>
          </section>
        )}

        {trace.length > 0 && (
          <section>
            <SectionTitle icon={FileText} text="思考过程" />
            <ol className="mt-2 space-y-0 border-l border-line pl-4">
              {trace.map((t) => (
                <li key={t.id} className="relative pb-3 pl-3">
                  <span className="absolute -left-[5px] top-1 h-2 w-2 rounded-full bg-accent-400" />
                  <p className="text-xs text-muted">{t.text}</p>
                  <span className="text-[10px] text-faint">{t.time}</span>
                </li>
              ))}
            </ol>
          </section>
        )}
      </div>
    </aside>
  );
}