"use client";

import { useEffect, useState } from "react";

type Summary = {
  recall: number | null; mrr: number | null; hit: number | null;
  abstention: number | null; constraint_satisfaction: number | null;
  filter_exact_match: number | null; p50_ms: number; p95_ms: number;
  error_count: number; fallback_count: number;
};
type Report = {
  status: string; message?: string; mode?: string; created_at?: string;
  case_count?: number; k?: number; note?: string; index_version?: string;
  results?: Record<string, { status: string; reason?: string; summary?: Summary }>;
};
const percent = (value: number | null | undefined) =>
  value == null ? "—" : (value * 100).toFixed(1) + "%";

export default function EvaluatePanel() {
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/evaluation/latest", { cache: "no-store", signal: controller.signal })
      .then(async (res) => {
        if (!res.ok) throw new Error("评测结果读取失败");
        setReport(await res.json() as Report);
      })
      .catch((err: Error) => { if (err.name !== "AbortError") setError(err.message); });
    return () => controller.abort();
  }, [refresh]);
  return (
    <div className="h-full space-y-5 overflow-y-auto p-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">检索评测</h2>
        <button className="card px-3 py-2 text-xs" onClick={() => { setError(""); setRefresh(v => v + 1); }}>刷新</button>
      </div>
      {error && <p role="alert" className="text-sm text-rose-300">{error}</p>}
      {!report && !error && <p className="text-sm text-muted">正在读取评测结果…</p>}
      {report?.status === "not_run" && <p className="text-sm text-muted">尚未运行评测。</p>}
      {report?.results && <>
        <p className="text-xs text-muted">
          {report.mode === "offline" ? "离线 BM25 基线" : "完整检索对比"} · {report.case_count} 道题 · K={report.k} · {report.created_at}
        </p>
        <p className="text-xs text-muted">{report.note}</p>
        <div className="card overflow-x-auto p-4">
          <table className="w-full whitespace-nowrap text-left text-sm">
            <thead><tr className="border-b border-line text-xs text-faint">
              {["检索方案", "Recall@K", "MRR", "Hit@K", "无答案拒答率", "条件满足率", "P95 / ms", "错误 / 降级"].map(h =>
                <th className="p-3 font-medium" key={h}>{h}</th>)}
            </tr></thead>
            <tbody>{Object.entries(report.results).map(([name, result]) => (
              <tr className="border-b border-line/50" key={name}>
                <td className="p-3">{name}{result.status !== "ok" && <span className="ml-2 text-xs text-amber-300">{result.reason || result.status}</span>}</td>
                <td className="p-3">{percent(result.summary?.recall)}</td>
                <td className="p-3">{result.summary?.mrr?.toFixed(3) ?? "—"}</td>
                <td className="p-3">{percent(result.summary?.hit)}</td>
                <td className="p-3">{percent(result.summary?.abstention)}</td>
                <td className="p-3">{percent(result.summary?.constraint_satisfaction)}</td>
                <td className="p-3">{result.summary?.p95_ms.toFixed(1) ?? "—"}</td>
                <td className="p-3">{result.summary ? result.summary.error_count + " / " + result.summary.fallback_count : "—"}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
        <p className="text-xs text-faint">这些指标评价岗位检索，不代表答案准确率或幻觉率。未运行的方案不会显示估算成绩。</p>
      </>}
    </div>
  );
}
