"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BarChart3, FolderOpen, MessageSquare, Target, Workflow } from "lucide-react";

const tabs = [
  { href: "/chat", label: "对话", icon: MessageSquare },
  { href: "/graph", label: "图谱", icon: Workflow },
  { href: "/gap", label: "人岗分析", icon: Target },
  { href: "/evaluate", label: "评估", icon: BarChart3 },
  { href: "/datasource", label: "数据源", icon: FolderOpen },
];

export default function CenterTabs() {
  const pathname = usePathname();

  return (
    <div className="flex h-12 shrink-0 items-center gap-1.5 border-b border-line px-4">
      {tabs.map((t) => {
        const active = pathname.startsWith(t.href);
        const Icon = t.icon;
        return (
          <Link
            key={t.href}
            href={t.href}
            className={`relative flex items-center gap-1.5 rounded-lg px-3.5 py-2 text-sm transition-colors ${
              active
                ? "bg-accent-500/12 font-medium text-ink"
                : "text-muted hover:bg-white/5 hover:text-ink"
            }`}
          >
            <Icon className={`h-4 w-4 ${active ? "text-accent-300" : ""}`} />
            {t.label}
            {active && (
              <span className="absolute inset-x-3 -bottom-[13px] h-0.5 rounded-full bg-gradient-to-r from-accent-400 to-accent-500" />
            )}
          </Link>
        );
      })}
    </div>
  );
}