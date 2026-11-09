import type { Metadata } from "next";
import "./globals.css";
import ThemeProvider from "@/components/theme/ThemeProvider";

export const metadata: Metadata = {
  title: "招聘知识图谱 RAG 工作台",
  description: "企业级招聘知识图谱检索增强问答系统（RAG + 图检索 + 人岗匹配）",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN" className="h-full antialiased">
      <body className="h-full overflow-hidden bg-canvas font-sans text-ink">
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}