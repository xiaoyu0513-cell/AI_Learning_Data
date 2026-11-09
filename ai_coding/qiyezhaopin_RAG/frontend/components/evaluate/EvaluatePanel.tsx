"use client";

import { useEffect, useState } from "react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

/* ===== 设计系统语义色（与 globals.css OKLCH token 对应） ===== */
const C = {
  violet: "#8b5cf6",
  blue: "#60a5fa",
  cyan: "#22d3ee",
  emerald: "#34d399",
  amber: "#f59e0b",
  rose: "#fb7185",
  grid: "rgba(255,255,255,0.05)",
  tick: "oklch(0.55 0 0)",
  muted: "#9a9ab3",
};

const kpis = [
  { label: "总评估问题数", value: 120, decimals: 0, suffix: "", delta: "+12 本周", up: true },
  { label: "总体准确率", value: 86.2, decimals: 1, suffix: "%", delta: "+2.4% 环比", up: true },
  { label: "平均响应时间", value: 1.2, decimals: 1, suffix: "s", delta: "-0.3s 环比", up: true },
  { label: "幻觉率", value: 3.1, decimals: 1, suffix: "%", delta: "目标 < 5%", up: true },
];

const domainData = [
  { domain: "AI基础设施", accuracy: 92 },
  { domain: "后端开发", accuracy: 88 },
  { domain: "算法研究", accuracy: 85 },
  { domain: "数据工程", accuracy: 81 },
  { domain: "前端开发", accuracy: 79 },
  { domain: "产品经理", accuracy: 76 },
];

const strategyData = [
  { name: "图 RAG", value: 38, color: C.violet },
  { name: "向量 + BM25", value: 34, color: C.blue },
  { name: "混合检索", value: 28, color: C.emerald },
];

const accTimeData = [
  { round: "R1", accuracy: 78, latency: 2.1 },
  { round: "R2", accuracy: 82, latency: 1.8 },
  { round: "R3", accuracy: 85, latency: 1.5 },
  { round: "R4", accuracy: 84, latency: 1.6 },
  { round: "R5", accuracy: 88, latency: 1.3 },
  { round: "R6", accuracy: 90, latency: 1.2 },
];

const ablation = [
  { variant: "仅向量检索 (Baseline)", acc: 72.4, latency: 1.1, mrr: 0.64, hit: 81, best: false },
  { variant: "+ BM25 混合", acc: 78.9, latency: 1.3, mrr: 0.71, hit: 86, best: false },
  { variant: "+ 图RAG 重排", acc: 84.6, latency: 1.5, mrr: 0.78, hit: 91, best: false },
  { variant: "+ 人岗先验（当前）", acc: 86.2, latency: 1.2, mrr: 0.81, hit: 94, best: true },
];

/* ===== 滚动计数器 ===== */
function useCountUp(target: number, duration = 800) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    if (typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setVal(target);
      return;
    }
    let raf = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const p = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setVal(target * eased);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, duration]);
  return val;
}

function KpiCard({ k }: { k: (typeof kpis)[number] }) {
  const v = useCountUp(k.value);
  const shown = v.toFixed(k.decimals);
  return (
    <div className="card card-hover p-5">
      <div className="text-xs text-muted">{k.label}</div>
      <div className="tnum mt-1.5 text-2xl font-semibold text-ink">
        {shown}
        <span className="ml-0.5 text-base text-muted">{k.suffix}</span>
      </div>
      <div
        className={`mt-1.5 inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] ${
          k.up ? "bg-emerald-500/10 text-emerald-300" : "bg-rose-500/10 text-rose-300"
        }`}
      >
        {k.delta}
      </div>
    </div>
  );
}

function ChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: any[];
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl border border-white/10 bg-zinc-900/90 p-3 shadow-2xl backdrop-blur-xl">
      {label != null && <div className="mb-1.5 text-xs font-medium text-muted">{label}</div>}
      {payload.map((p) => (
        <div key={p.dataKey} className="flex items-center gap-2 py-0.5 text-xs">
          <span
            className="h-2 w-2 rounded-full"
            style={{ background: p.color || p.payload?.color || C.violet }}
          />
          <span className="text-muted">{p.name}:</span>
          <span className="tnum font-medium text-ink">{p.value}</span>
        </div>
      ))}
    </div>
  );
}

function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`card card-hover p-5 ${className}`}>{children}</div>;
}

export default function EvaluatePanel() {
  return (
    <div className="h-full space-y-5 overflow-y-auto p-6">
      <div>
        <h2 className="text-lg font-semibold">检索回答评估看板</h2>
        <p className="mt-0.5 text-xs text-faint">
          示例数据（离线快照）· LLM Judge + Red Team 汇聚结果
        </p>
      </div>

      {/* KPI */}
      <div className="grid grid-cols-2 gap-5 lg:grid-cols-4">
        {kpis.map((k) => (
          <KpiCard key={k.label} k={k} />
        ))}
      </div>

      {/* 图表区 */}
      <div className="grid gap-5 lg:grid-cols-2">
        <Card>
          <h3 className="mb-4 text-sm font-medium">准确率按领域分布</h3>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={domainData} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
                <defs>
                  <linearGradient id="barGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#6d28d9" />
                    <stop offset="100%" stopColor="#60a5fa" />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={C.grid} vertical={false} />
                <XAxis
                  dataKey="domain"
                  tick={{ fill: C.tick, fontSize: 12 }}
                  axisLine={false}
                  tickLine={false}
                  interval={0}
                  angle={-12}
                  textAnchor="end"
                  height={48}
                />
                <YAxis
                  domain={[0, 100]}
                  tick={{ fill: C.tick, fontSize: 12 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip content={<ChartTooltip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
                <Bar dataKey="accuracy" name="准确率 %" fill="url(#barGrad)" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card>
          <h3 className="mb-4 text-sm font-medium">检索策略占比</h3>
          <div className="h-40">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={strategyData}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={55}
                  outerRadius={80}
                  paddingAngle={3}
                  stroke="none"
                >
                  {strategyData.map((s) => (
                    <Cell key={s.name} fill={s.color} />
                  ))}
                </Pie>
                <Tooltip content={<ChartTooltip />} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          {/* 图例（下方，圆点 + 文字） */}
          <div className="mt-2 flex flex-wrap justify-center gap-x-5 gap-y-2">
            {strategyData.map((s) => (
              <div key={s.name} className="flex items-center gap-1.5 text-xs text-muted">
                <span className="h-2 w-2 rounded-full" style={{ background: s.color }} />
                <span>{s.name}</span>
                <span className="tnum text-faint">{s.value}%</span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* 双轴折线/面积图 */}
      <Card>
        <h3 className="mb-4 text-sm font-medium">准确率 vs 响应时间（迭代轮次）</h3>
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={accTimeData} margin={{ top: 4, right: 0, left: -16, bottom: 0 }}>
              <defs>
                <linearGradient id="accFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke={C.grid} vertical={false} />
              <XAxis
                dataKey="round"
                tick={{ fill: C.tick, fontSize: 12 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                yAxisId="l"
                domain={[60, 100]}
                tick={{ fill: C.tick, fontSize: 12 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                yAxisId="r"
                orientation="right"
                domain={[0, 2.5]}
                tick={{ fill: C.tick, fontSize: 12 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip content={<ChartTooltip />} />
              <Area
                yAxisId="l"
                type="monotone"
                dataKey="accuracy"
                name="准确率 %"
                stroke={C.violet}
                fill="url(#accFill)"
                strokeWidth={2}
                dot={{ r: 3, fill: C.violet }}
              />
              <Line
                yAxisId="r"
                type="monotone"
                dataKey="latency"
                name="响应时间 s"
                stroke={C.emerald}
                strokeWidth={2}
                dot={{ r: 3, fill: C.emerald }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </Card>

      {/* 消融实验对比表 */}
      <Card>
        <h3 className="mb-4 text-sm font-medium">消融实验对比</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-line text-left text-xs text-faint">
                <th className="pb-2.5 pr-4 font-medium">配置</th>
                <th className="pb-2.5 pr-4 font-medium">准确率</th>
                <th className="pb-2.5 pr-4 font-medium">响应时间</th>
                <th className="pb-2.5 pr-4 font-medium">MRR</th>
                <th className="pb-2.5 font-medium">命中率</th>
              </tr>
            </thead>
            <tbody>
              {ablation.map((r) => (
                <tr
                  key={r.variant}
                  className={`border-b border-line/50 transition-colors last:border-0 ${
                    r.best ? "bg-emerald-500/5" : "hover:bg-white/[0.03]"
                  }`}
                >
                  <td className="py-2.5 pr-4">
                    <span
                      className={
                        r.best ? "font-medium text-emerald-300" : "text-ink"
                      }
                    >
                      {r.variant}
                    </span>
                    {r.best && (
                      <span className="ml-2 rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] text-emerald-300">
                        最优
                      </span>
                    )}
                  </td>
                  <td className="tnum py-2.5 pr-4 text-muted">{r.acc.toFixed(1)}%</td>
                  <td className="tnum py-2.5 pr-4 text-muted">{r.latency.toFixed(1)}s</td>
                  <td className="tnum py-2.5 pr-4 text-muted">{r.mrr.toFixed(2)}</td>
                  <td className="tnum py-2.5 text-muted">{r.hit}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}