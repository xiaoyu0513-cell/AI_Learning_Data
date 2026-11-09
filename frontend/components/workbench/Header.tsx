"use client";

import { Database, Network, PanelRightClose, PanelRightOpen, Search } from "lucide-react";
import { useSystemStatus } from "@/lib/hooks";
import { useWorkbench } from "@/lib/store/workbench";
import ThemeSwitcher from "@/components/workbench/ThemeSwitcher";

export default function Header() {
  const { health, stats, ready } = useSystemStatus();
  const rightOpen = useWorkbench((s) => s.rightOpen);
  const toggleRight = useWorkbench((s) => s.toggleRight);
  const jobs = stats?.graph?.Job ?? 0;
  const rows = stats?.milvus?.row_count ?? health?.milvus_rows ?? 0;

  return (
    <header className="flex h-14 shrink-0 items-center gap-3 border-b border-line bg-surface/70 px-5 backdrop-blur-xl">
      <div className="flex items-center gap-2.5">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-accent-400 via-accent-500 to-accent-2 shadow-lg shadow-accent-500/25">
          <Network className="h-4 w-4 text-white" />
        </div>
        <div className="flex flex-col leading-tight">
          <span className="text-gradient text-sm font-semibold">招聘知识图谱</span>
          <span className="text-[11px] text-faint">RAG 工作台</span>
        </div>
      </div>

      <div className="mx-2 flex max-w-md flex-1 items-center gap-2 rounded-lg border border-line bg-panel px-3 py-1.5 text-muted">
        <Search className="h-4 w-4" />
        <span className="text-xs">全局搜索 / 输入命令…</span>
        <kbd className="ml-auto rounded border border-line px-1.5 text-[10px]">
          Ctrl K
        </kbd>
      </div>

      <div className="ml-auto flex items-center gap-4">
        <ThemeSwitcher />
        <button
          onClick={toggleRight}
          title={rightOpen ? "收起检索面板" : "展开检索面板"}
          className={`flex h-8 w-8 items-center justify-center rounded-lg border border-line text-muted transition-colors hover:bg-white/5 hover:text-ink ${
            rightOpen ? "bg-accent-500/10 text-accent-300" : ""
          }`}
        >
          {rightOpen ? (
            <PanelRightClose className="h-4 w-4" />
          ) : (
            <PanelRightOpen className="h-4 w-4" />
          )}
        </button>
        <span className="flex items-center gap-1.5 text-xs text-muted">
          <Database
            className={`h-3.5 w-3.5 ${ready ? "text-emerald-400" : "text-faint"}`}
          />
          索引 v2.3 · {rows} 向量
        </span>
        <span className="hidden items-center gap-1.5 text-xs text-muted sm:flex">
          <Database className="h-3.5 w-3.5 text-blue-400" />
          {jobs} 岗位
        </span>
        <span
          className={`flex items-center gap-1.5 text-xs ${ready ? "text-emerald-400" : "text-amber-400"}`}
        >
          <span
            className={`h-2 w-2 animate-pulse rounded-full ${ready ? "bg-emerald-400" : "bg-amber-400"}`}
          />
          {ready ? "系统就绪" : "连接中…"}
        </span>
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-amber-400 to-rose-500 text-xs font-semibold text-white">
          管
        </div>
      </div>
    </header>
  );
}