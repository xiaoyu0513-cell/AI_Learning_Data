"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  FolderOpen,
  MessageSquare,
  Pin,
  Search,
  Settings,
  Target,
  Workflow,
} from "lucide-react";

const nav = [
  { href: "/chat", label: "问答", desc: "RAG 对话", icon: MessageSquare },
  { href: "/graph", label: "图谱", desc: "知识图谱", icon: Workflow },
  { href: "/gap", label: "人岗分析", desc: "Gap 匹配", icon: Target },
  { href: "/evaluate", label: "评估", desc: "质量看板", icon: BarChart3 },
  { href: "/datasource", label: "数据源", desc: "知识库管理", icon: FolderOpen },
];

const pinned = [
  "RAG算法工程师 · 人岗匹配",
  "Go+K8s 基础设施岗位",
  "候选人 张三 技能盘点",
];

export default function LeftNav() {
  const pathname = usePathname();

  return (
    <aside className="flex h-full flex-col border-r border-line bg-surface">
      <div className="p-3">
        <div className="flex items-center gap-2 rounded-lg border border-line bg-panel/70 px-2.5 py-2 text-muted transition-colors focus-within:border-accent-500/50">
          <Search className="h-4 w-4" />
          <span className="flex-1 text-xs">搜索会话…</span>
          <kbd className="rounded border border-line px-1 text-[10px]">Ctrl K</kbd>
        </div>
      </div>

      <nav className="space-y-1.5 px-2.5">
        {nav.map((n) => {
          const active = pathname.startsWith(n.href);
          const Icon = n.icon;
          return (
            <Link
              key={n.href}
              href={n.href}
              className={`group relative flex items-center gap-3 rounded-xl px-3 py-2.5 transition-colors ${
                active
                  ? "bg-accent-500/12 text-ink"
                  : "text-muted hover:bg-white/5 hover:text-ink"
              }`}
            >
              {active && (
                <span className="absolute left-0 top-1/2 h-5 w-1 -translate-y-1/2 rounded-r-full bg-gradient-to-b from-accent-400 to-accent-500" />
              )}
              <Icon
                className={`h-[18px] w-[18px] ${
                  active ? "text-accent-300" : "text-faint group-hover:text-muted"
                }`}
              />
              <div className="flex flex-col">
                <span className="text-sm">{n.label}</span>
                <span className="text-[11px] text-faint">{n.desc}</span>
              </div>
            </Link>
          );
        })}
      </nav>

      <div className="mt-5 min-h-0 flex-1 overflow-y-auto px-2">
        <div className="mb-2 flex items-center gap-1.5 px-2.5 text-[11px] font-medium uppercase tracking-wide text-faint">
          <Pin className="h-3 w-3" /> 固定会话
        </div>
        <div className="space-y-1">
          {pinned.map((p) => (
            <button
              key={p}
              className="w-full rounded-lg px-2.5 py-1.5 text-left text-xs text-muted transition-colors hover:bg-white/5 hover:text-ink"
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      <div className="border-t border-line p-2">
        <button className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-muted transition-colors hover:bg-white/5 hover:text-ink">
          <Settings className="h-4 w-4" />
          <span className="text-sm">设置</span>
        </button>
      </div>
    </aside>
  );
}