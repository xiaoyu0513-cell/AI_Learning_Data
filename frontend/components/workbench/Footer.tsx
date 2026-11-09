"use client";

import { useSystemStatus } from "@/lib/hooks";

export default function Footer() {
  const { ready } = useSystemStatus(30000);

  return (
    <footer className="flex h-8 shrink-0 items-center gap-4 border-t border-line bg-surface px-4 text-[11px] text-faint">
      <span className="flex items-center gap-1.5">
        <span
          className={`h-1.5 w-1.5 rounded-full ${ready ? "bg-emerald-400" : "bg-amber-400"}`}
        />
        系统状态 {ready ? "正常" : "离线"}
      </span>
      <span>索引版本 v2.3</span>
      <span>最后更新 2026-10-04</span>
      <span className="ml-auto">Neo4j + Milvus · BGE-small-zh-v1.5 · DeepSeek</span>
    </footer>
  );
}