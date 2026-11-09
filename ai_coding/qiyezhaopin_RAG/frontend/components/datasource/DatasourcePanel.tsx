"use client";

import { useCallback, useEffect, useState } from "react";
import {
  CheckCircle2,
  Database,
  FileText,
  FileType2,
  Layers,
  Loader2,
  Pencil,
  Plus,
  RefreshCw,
  Trash2,
  UploadCloud,
} from "lucide-react";
import { buildKnowledgeBase, getStats } from "@/lib/api";
import type { Stats } from "@/lib/types";

const sourceMeta: {
  key: string;
  label: string;
  desc: string;
  color: string;
}[] = [
  { key: "Job", label: "岗位 Job", desc: "招聘岗位、技能要求与职责", color: "#60a5fa" },
  { key: "Candidate", label: "候选人 Candidate", desc: "候选人画像与技能栈", color: "#f43f5e" },
  { key: "Skill", label: "技能 Skill", desc: "技能节点与前置依赖", color: "#10b981" },
  { key: "Company", label: "公司 Company", desc: "雇主与行业归属", color: "#22d3ee" },
  { key: "JobCategory", label: "岗位分类", desc: "六大岗位类别", color: "#a855f7" },
  { key: "SeniorityLevel", label: "职级", desc: "初级 / 中级 / 高级", color: "#06b6d4" },
  { key: "Responsibility", label: "职责描述", desc: "岗位职责文本 chunk", color: "#94a3b8" },
];

const formats = [
  { ext: "MD", cls: "text-blue-400 border-blue-400/30 bg-blue-500/10" },
  { ext: "PDF", cls: "text-rose-400 border-rose-400/30 bg-rose-500/10" },
  { ext: "CSV", cls: "text-emerald-400 border-emerald-400/30 bg-emerald-500/10" },
  { ext: "DOCX", cls: "text-blue-400 border-blue-400/30 bg-blue-500/10" },
  { ext: "XLSX", cls: "text-emerald-400 border-emerald-400/30 bg-emerald-500/10" },
];

const timeline = [
  { id: 1, type: "add" as const, text: "新增 32 个岗位节点", time: "今天 14:22" },
  { id: 2, type: "edit" as const, text: "更新 5 条技能依赖关系", time: "昨天 09:10" },
  { id: 3, type: "add" as const, text: "导入候选人画像 12 份", time: "10-02 18:45" },
  { id: 4, type: "delete" as const, text: "清理 3 个重复职责 chunk", time: "10-01 11:30" },
];

const timelineIcon = {
  add: { Icon: Plus, cls: "text-emerald-400 bg-emerald-500/15" },
  edit: { Icon: Pencil, cls: "text-amber-400 bg-amber-500/15" },
  delete: { Icon: Trash2, cls: "text-rose-400 bg-rose-500/15" },
};

export default function DatasourcePanel() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [building, setBuilding] = useState(false);
  const [message, setMessage] = useState("");
  const [dragging, setDragging] = useState(false);

  const load = useCallback(async () => {
    try {
      setStats(await getStats());
    } catch {
      // 后端未就绪
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const build = async () => {
    setBuilding(true);
    setMessage("");
    try {
      const res = await buildKnowledgeBase(false);
      setMessage(res.status === "ok" ? "知识库构建完成" : res.message ?? "构建失败");
      await load();
    } catch {
      setMessage("构建请求失败，请确认后端已就绪");
    } finally {
      setBuilding(false);
    }
  };

  const totalNodes = stats
    ? Object.values(stats.graph).reduce((a, b) => a + b, 0)
    : 0;
  const vectors = stats?.milvus?.row_count ?? 0;

  return (
    <div className="h-full space-y-5 overflow-y-auto p-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">知识库数据源</h2>
          <p className="mt-0.5 text-xs text-faint">
            Neo4j 图数据 + Milvus 向量索引 · 实时统计来自 /api/stats
          </p>
        </div>
        <button
          onClick={build}
          disabled={building}
          className="flex items-center gap-2 rounded-lg bg-accent-600 px-3.5 py-2 text-sm text-white shadow-lg shadow-accent-500/25 transition-colors hover:bg-accent-500 disabled:opacity-60"
        >
          {building ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <RefreshCw className="h-4 w-4" />
          )}
          {building ? "构建中…" : "重建知识库"}
        </button>
      </div>

      {message && <div className="card px-4 py-2.5 text-xs text-muted">{message}</div>}

      {/* 上传区域 */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
        }}
        className={`flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed px-6 py-10 text-center transition-all ${
          dragging
            ? "border-accent-400/60 bg-accent-500/5"
            : "border-white/15 bg-white/[0.015]"
        }`}
      >
        <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-accent-500/15 text-accent-300">
          <UploadCloud className="h-6 w-6" />
        </div>
        <div>
          <p className="text-sm font-medium text-ink">拖拽文件到此处上传</p>
          <p className="mt-1 text-xs text-faint">
            或将 JSON / Markdown / PDF / CSV 等格式文件放入数据目录后点「重建知识库」
          </p>
        </div>
        <div className="mt-1 flex flex-wrap justify-center gap-1.5">
          {formats.map((f) => (
            <span
              key={f.ext}
              className={`rounded-md border px-2 py-0.5 text-[10px] font-medium ${f.cls}`}
            >
              {f.ext}
            </span>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-5 lg:grid-cols-4">
        <StatCard icon={Database} label="图节点总数" value={totalNodes} color="text-accent-400" />
        <StatCard icon={Layers} label="向量条数" value={vectors} color="text-emerald-400" />
        <StatCard
          icon={Database}
          label="岗位数"
          value={stats?.graph?.Job ?? 0}
          color="text-blue-400"
        />
        <StatCard
          icon={Layers}
          label="技能数"
          value={stats?.graph?.Skill ?? 0}
          color="text-amber-400"
        />
      </div>

      <div className="grid gap-5 lg:grid-cols-3">
        {/* 数据源概览 */}
        <div className="card overflow-hidden lg:col-span-2">
          <div className="flex items-center gap-2 border-b border-line px-4 py-3">
            <FileText className="h-4 w-4 text-accent-400" />
            <span className="text-sm font-medium">数据源概览</span>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-line bg-panel text-left text-xs text-muted">
                <th className="px-4 py-2.5 font-medium">数据源</th>
                <th className="px-4 py-2.5 text-right font-medium">节点数</th>
                <th className="px-4 py-2.5 font-medium">状态</th>
              </tr>
            </thead>
            <tbody>
              {sourceMeta.map((s) => {
                const count = stats?.graph?.[s.key] ?? 0;
                return (
                  <tr
                    key={s.key}
                    className="border-b border-line transition-colors last:border-0 hover:bg-white/[0.03]"
                  >
                    <td className="px-4 py-2.5">
                      <div className="flex items-center gap-2.5">
                        <span
                          className="rounded-md px-1.5 py-0.5 text-[10px] font-medium"
                          style={{ background: `${s.color}22`, color: s.color }}
                        >
                          {s.key === "JobCategory" ? "CAT" : s.key}
                        </span>
                        <div className="leading-tight">
                          <div className="text-ink">{s.label}</div>
                          <div className="text-[11px] text-faint">{s.desc}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-muted">{count}</td>
                    <td className="px-4 py-2.5">
                      <span className="inline-flex items-center gap-1.5 text-xs text-emerald-400">
                        <CheckCircle2 className="h-3.5 w-3.5" />
                        已索引
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* 增量更新日志 */}
        <div className="card p-4">
          <div className="flex items-center gap-2 border-b border-line pb-3">
            <FileType2 className="h-4 w-4 text-accent-400" />
            <span className="text-sm font-medium">增量更新日志</span>
          </div>
          <ol className="mt-3 space-y-0 border-l border-line pl-4">
            {timeline.map((t) => {
              const meta = timelineIcon[t.type];
              const Icon = meta.Icon;
              return (
                <li key={t.id} className="relative pb-4 pl-4 last:pb-0">
                  <span
                    className={`absolute -left-[13px] top-0.5 flex h-6 w-6 items-center justify-center rounded-full ${meta.cls}`}
                  >
                    <Icon className="h-3 w-3" />
                  </span>
                  <p className="text-xs leading-snug text-muted">{t.text}</p>
                  <span className="mt-0.5 block text-[10px] text-faint">{t.time}</span>
                </li>
              );
            })}
          </ol>
        </div>
      </div>
    </div>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
  color,
}: {
  icon: typeof Database;
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div className="card card-hover p-5">
      <div className="flex items-center gap-2 text-xs text-muted">
        <Icon className="h-3.5 w-3.5" />
        {label}
      </div>
      <div className={`mt-1.5 text-2xl font-semibold tabular-nums ${color}`}>{value}</div>
    </div>
  );
}