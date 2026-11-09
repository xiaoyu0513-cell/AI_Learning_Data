"use client";

import { useEffect, useState } from "react";
import { getHealth, getStats } from "./api";
import type { Health, Stats } from "./types";

/** 周期性获取后端健康状态与索引统计（供 Header/Footer 复用） */
export function useSystemStatus(intervalMs = 20000) {
  const [health, setHealth] = useState<Health | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const [h, s] = await Promise.all([getHealth(), getStats()]);
        if (!alive) return;
        setHealth(h);
        setStats(s);
      } catch {
        // 后端尚未就绪
      }
    };
    load();
    const t = setInterval(load, intervalMs);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [intervalMs]);

  return { health, stats, ready: health?.ready ?? false };
}