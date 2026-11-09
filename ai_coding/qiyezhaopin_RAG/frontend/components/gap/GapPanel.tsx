"use client";

import { useMemo, useState } from "react";
import {
  ArrowRight,
  CheckCircle2,
  ChevronDown,
  GitCompareArrows,
  GraduationCap,
  Loader2,
  Sparkles,
  Target,
  UserRound,
  XCircle,
} from "lucide-react";
import { getGap, getWhatIf } from "@/lib/api";
import type { GapResult, WhatIfResult } from "@/lib/types";

type CandidateMeta = {
  id: string;
  name: string;
  years: number;
  education: string;
  skills: string[];
};

type JobMeta = {
  id: string;
  name: string;
  company?: string;
};

const candidates: CandidateMeta[] = [
  { id: "cand_001", name: "张三", years: 3, education: "本科", skills: ["Python", "FastAPI", "MySQL", "Redis", "Docker"] },
  { id: "cand_002", name: "李四", years: 5, education: "硕士", skills: ["Go", "Kubernetes", "Docker", "Linux", "Python"] },
  { id: "cand_003", name: "王五", years: 0, education: "本科", skills: ["Python", "RAG", "LangChain", "PyTorch"] },
  { id: "cand_004", name: "赵六", years: 2, education: "本科", skills: ["Java", "Spring Boot", "MySQL"] },
  { id: "cand_005", name: "孙七", years: 4, education: "本科", skills: ["Python", "PyTorch", "NLP", "TensorFlow"] },
  { id: "cand_006", name: "周八", years: 0, education: "硕士", skills: ["Python", "知识图谱", "Neo4j", "RAG"] },
];

const jobs: JobMeta[] = [
  { id: "job_001", name: "Python后端开发工程师（腾讯·深圳）", company: "腾讯" },
  { id: "job_002", name: "Python后端开发工程师（阿里巴巴·杭州）", company: "阿里巴巴" },
  { id: "job_003", name: "Java后端开发工程师", company: "美团" },
  { id: "job_004", name: "后端开发工程师（应届）", company: "字节跳动" },
  { id: "job_005", name: "高级Python后端开发工程师（京东·北京）", company: "京东" },
  { id: "job_006", name: "Go后端开发工程师", company: "蚂蚁集团" },
  { id: "job_010", name: "AI平台后端工程师", company: "字节跳动" },
  { id: "job_011", name: "大模型基础设施工程师", company: "美团" },
  { id: "job_012", name: "云原生AI基础设施工程师", company: "阿里巴巴" },
  { id: "job_013", name: "AI训推平台工程师", company: "华为" },
  { id: "job_014", name: "AI基础设施运维工程师", company: "科大讯飞" },
  { id: "job_020", name: "RAG算法工程师（应届）", company: "百度" },
  { id: "job_021", name: "大模型应用开发工程师（应届）", company: "智谱AI" },
  { id: "job_022", name: "NLP算法工程师", company: "科大讯飞" },
  { id: "job_023", name: "知识图谱工程师（应届）", company: "百度" },
  { id: "job_024", name: "大模型训练工程师", company: "智谱AI" },
  { id: "job_030", name: "数据工程师", company: "字节跳动" },
  { id: "job_040", name: "前端开发工程师", company: "腾讯" },
  { id: "job_050", name: "云原生平台工程师", company: "阿里巴巴" },
  { id: "job_051", name: "SRE工程师", company: "华为" },
];

function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`card p-5 ${className}`}>{children}</div>;
}

function Chip({ active, onClick, children }: { active?: boolean; onClick?: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`rounded-full border px-2.5 py-1 text-xs transition-colors ${
        active
          ? "border-accent-500/50 bg-accent-500/15 text-accent-300"
          : "border-line bg-surface text-muted hover:text-ink"
      }`}
    >
      {children}
    </button>
  );
}

export default function GapPanel() {
  const [candidate, setCandidate] = useState("张三");
  const [job, setJob] = useState("RAG算法工程师（应届）");
  const [loading, setLoading] = useState(false);
  const [gap, setGap] = useState<GapResult | null>(null);
  const [err, setErr] = useState("");

  const [selected, setSelected] = useState<string[]>([]);
  const [whatifLoading, setWhatifLoading] = useState(false);
  const [whatif, setWhatif] = useState<WhatIfResult | null>(null);
  const [whatifErr, setWhatifErr] = useState("");

  const candidateMeta = useMemo(
    () => candidates.find((c) => c.name === candidate),
    [candidate],
  );
  const jobMeta = useMemo(() => jobs.find((j) => j.name === job), [job]);

  const analyze = async () => {
    setLoading(true);
    setErr("");
    setGap(null);
    setWhatif(null);
    setSelected([]);
    try {
      const res = await getGap(candidate, job);
      if (res.error) {
        setErr(res.error);
      } else {
        setGap(res);
      }
    } catch {
      setErr("分析请求失败，请确认后端已就绪");
    } finally {
      setLoading(false);
    }
  };

  const runWhatIf = async () => {
    if (selected.length === 0) return;
    setWhatifLoading(true);
    setWhatifErr("");
    setWhatif(null);
    try {
      const res = await getWhatIf(candidate, selected);
      if (res.error) {
        setWhatifErr(res.error);
      } else {
        setWhatif(res);
      }
    } catch {
      setWhatifErr("推演请求失败，请确认后端已就绪");
    } finally {
      setWhatifLoading(false);
    }
  };

  const toggleSkill = (skill: string) => {
    setSelected((prev) =>
      prev.includes(skill) ? prev.filter((s) => s !== skill) : [...prev, skill],
    );
  };

  const matchPct = gap ? Math.round(gap.match_rate * 100) : 0;

  return (
    <div className="h-full space-y-5 overflow-y-auto p-6">
      <div>
        <h2 className="text-lg font-semibold">人岗 Gap 分析</h2>
        <p className="mt-0.5 text-xs text-faint">
          技能匹配率 + 缺口识别 + 学习路径 + What-if 推演 · 对接 /api/gap 与 /api/gap/whatif
        </p>
      </div>

      {/* 选择器 */}
      <Card>
        <div className="grid gap-3 lg:grid-cols-[1fr_1fr_auto]">
          <Field label="候选人">
            <select
              value={candidate}
              onChange={(e) => {
                setCandidate(e.target.value);
                setGap(null);
                setWhatif(null);
              }}
              className="w-full appearance-none rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink outline-none transition-colors focus:border-accent-500/50"
            >
              {candidates.map((c) => (
                <option key={c.id} value={c.name}>
                  {c.name} · {c.years}年 · {c.education}
                </option>
              ))}
            </select>
            <ChevronDown className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
          </Field>

          <Field label="目标岗位">
            <select
              value={job}
              onChange={(e) => {
                setJob(e.target.value);
                setGap(null);
                setWhatif(null);
              }}
              className="w-full appearance-none rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink outline-none transition-colors focus:border-accent-500/50"
            >
              {jobs.map((j) => (
                <option key={j.id} value={j.name}>
                  {j.name}
                </option>
              ))}
            </select>
            <ChevronDown className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
          </Field>

          <div className="flex items-end">
            <button
              onClick={analyze}
              disabled={loading}
              className="flex items-center justify-center gap-2 rounded-lg bg-accent-600 px-4 py-2 text-sm text-white shadow-lg shadow-accent-500/25 transition-colors hover:bg-accent-500 disabled:opacity-60"
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <GitCompareArrows className="h-4 w-4" />}
              {loading ? "分析中…" : "对比分析"}
            </button>
          </div>
        </div>

        {candidateMeta && (
          <div className="mt-3 flex flex-wrap items-center gap-1.5 border-t border-line pt-3 text-xs text-muted">
            <span className="flex items-center gap-1 text-faint">
              <UserRound className="h-3.5 w-3.5" /> {candidateMeta.name}的已会技能：
            </span>
            {candidateMeta.skills.map((s) => (
              <span key={s} className="rounded-full border border-line bg-surface px-2 py-0.5 text-muted">
                {s}
              </span>
            ))}
          </div>
        )}
      </Card>

      {err && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-300">
          {err}
        </div>
      )}

      {/* 分析结果 */}
      {gap && jobMeta && (
        <div className="space-y-4">
          <div className="grid gap-4 lg:grid-cols-[220px_1fr]">
            {/* 匹配率 */}
            <Card>
              <div className="text-xs text-muted">匹配率</div>
              <div className="mt-1 text-4xl font-semibold text-accent-400">{matchPct}%</div>
              <div className="mt-3 h-2 overflow-hidden rounded-full bg-surface">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-accent-500 to-emerald-400 transition-all"
                  style={{ width: `${matchPct}%` }}
                />
              </div>
              <div className="mt-3 flex items-center gap-1.5 text-xs text-muted">
                <Target className="h-3.5 w-3.5" /> {jobMeta.name}
              </div>
            </Card>

            {/* 技能匹配明细 */}
            <Card>
              <div className="mb-3 flex items-center justify-between">
                <h3 className="text-sm font-medium">技能对照</h3>
                <div className="flex items-center gap-2 text-[11px] text-faint">
                  <span className="flex items-center gap-1">
                    <CheckCircle2 className="h-3 w-3 text-emerald-400" />
                    已匹配 {gap.matched_skills.length}
                  </span>
                  <span className="flex items-center gap-1">
                    <XCircle className="h-3 w-3 text-rose-400" />
                    缺失 {gap.missing_skills.length}
                  </span>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex flex-wrap gap-1.5">
                  {gap.matched_skills.map((s) => (
                    <Chip key={s} active>
                      {s}
                    </Chip>
                  ))}
                  {gap.matched_skills.length === 0 && (
                    <span className="text-xs text-faint">无已匹配技能</span>
                  )}
                </div>
                {gap.missing_skills.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 border-t border-line pt-2">
                    {gap.missing_skills.map((s) => {
                      const isSel = selected.includes(s);
                      return (
                        <button
                          key={s}
                          onClick={() => toggleSkill(s)}
                          className={`flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs transition-colors ${
                            isSel
                              ? "border-rose-400/60 bg-rose-500/20 text-rose-300"
                              : "border-rose-500/30 bg-rose-500/10 text-rose-300/80 hover:text-rose-200"
                          }`}
                        >
                          {s}
                          {isSel && <span className="text-rose-200">✓</span>}
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
              <p className="mt-3 text-xs text-faint">点击缺失技能可选中，用于下方 What-if 推演</p>
            </Card>
          </div>

          {/* 学习路径 */}
          {gap.learning_paths.length > 0 && (
            <Card>
              <h3 className="mb-3 flex items-center gap-1.5 text-sm font-medium">
                <GraduationCap className="h-4 w-4" /> 建议学习路径
              </h3>
              <div className="grid gap-2 lg:grid-cols-2">
                {gap.learning_paths.map((lp) => (
                  <div
                    key={lp.skill}
                    className="flex items-center gap-2 rounded-lg border border-line bg-surface px-3 py-2"
                  >
                    <span className="text-sm text-ink">{lp.skill}</span>
                    {lp.prerequisites.length > 0 && (
                      <>
                        <ArrowRight className="h-3.5 w-3.5 text-faint" />
                        <div className="flex flex-wrap items-center gap-1">
                          {lp.prerequisites.map((p, i) => (
                            <span key={p} className="flex items-center gap-1 text-xs text-muted">
                              {i > 0 && <span className="text-faint">→</span>}
                              {p}
                            </span>
                          ))}
                        </div>
                      </>
                    )}
                  </div>
                ))}
              </div>
            </Card>
          )}

          <div className="card border-accent-500/20 bg-accent-500/5 px-4 py-3 text-sm text-accent-300">
            {gap.recommendation}
          </div>
        </div>
      )}

      {/* What-if 推演 */}
      {gap && (
        <Card>
          <div className="mb-3 flex items-center justify-between">
            <h3 className="flex items-center gap-1.5 text-sm font-medium">
              <Sparkles className="h-4 w-4" /> What-if 推演
            </h3>
            <button
              onClick={runWhatIf}
              disabled={selected.length === 0 || whatifLoading}
              className="flex items-center gap-2 rounded-lg bg-emerald-600 px-3 py-1.5 text-xs text-white transition-colors hover:bg-emerald-500 disabled:opacity-50"
            >
              {whatifLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
              补齐已选技能并推演
            </button>
          </div>
          <p className="text-xs text-faint">
            已选 {selected.length} 项技能：{selected.length ? selected.join("、") : "未选择"}
          </p>

          {whatifErr && (
            <div className="mt-3 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-300">
              {whatifErr}
            </div>
          )}

          {whatif && (
            <div className="mt-3 space-y-2">
              {whatif.new_opportunities.length === 0 ? (
                <p className="text-xs text-muted">补齐这些技能后暂无匹配度 ≥ 50% 的新岗位</p>
              ) : (
                whatif.new_opportunities.map((op) => (
                  <div
                    key={op.job_id}
                    className="flex items-center justify-between gap-3 rounded-lg border border-line bg-surface px-3 py-2"
                  >
                    <div className="min-w-0">
                      <div className="truncate text-sm text-ink">{op.job_name}</div>
                      <div className="mt-0.5 flex flex-wrap gap-1">
                        {op.newly_matched.map((s) => (
                          <span
                            key={s}
                            className="rounded-full bg-emerald-500/15 px-1.5 py-0.5 text-[10px] text-emerald-300"
                          >
                            + {s}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div className="shrink-0 text-right">
                      <div className="text-lg font-semibold text-emerald-400">
                        {Math.round(op.match_rate * 100)}%
                      </div>
                      <div className="text-[10px] text-faint">匹配率</div>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
        </Card>
      )}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-1.5 text-xs text-muted">{label}</div>
      <div className="relative">{children}</div>
    </div>
  );
}