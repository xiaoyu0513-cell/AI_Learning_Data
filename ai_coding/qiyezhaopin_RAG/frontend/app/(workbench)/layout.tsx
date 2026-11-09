"use client";

import { useEffect } from "react";
import {
  Group,
  Panel,
  Separator,
  useGroupRef,
} from "react-resizable-panels";
import Header from "@/components/workbench/Header";
import Footer from "@/components/workbench/Footer";
import LeftNav from "@/components/workbench/LeftNav";
import RightPanel from "@/components/workbench/RightPanel";
import CenterTabs from "@/components/workbench/CenterTabs";
import { useWorkbench } from "@/lib/store/workbench";

const STORAGE_KEY = "recruitment-workbench-layout";

export default function WorkbenchLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const groupRef = useGroupRef();
  const rightOpen = useWorkbench((s) => s.rightOpen);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (raw) groupRef.current?.setLayout(JSON.parse(raw));
    } catch {
      // 忽略损坏的布局缓存
    }
  }, [groupRef]);

  return (
    <div className="relative z-10 flex h-full flex-col">
      <Header />
      <Group
        id="recruitment-workbench"
        orientation="horizontal"
        groupRef={groupRef}
        onLayoutChanged={(layout, meta) => {
          if (!meta.isUserInteraction) return;
          try {
            if (typeof window !== "undefined") {
              localStorage.setItem(STORAGE_KEY, JSON.stringify(layout));
            }
          } catch {
            // 忽略存储失败
          }
        }}
        className="min-h-0 flex-1"
      >
        <Panel
          id="nav"
          defaultSize={16}
          minSize={12}
          maxSize={24}
          collapsible
          collapsedSize={0}
        >
          <LeftNav />
        </Panel>
        <Separator className="bg-line transition-colors hover:bg-accent-500/70" />
        <Panel id="center" defaultSize={56} minSize={34}>
          <div className="flex h-full min-w-0 flex-col">
            <CenterTabs />
            <main className="min-h-0 flex-1 overflow-hidden">{children}</main>
          </div>
        </Panel>
        {rightOpen && (
          <>
            <Separator className="bg-line transition-colors hover:bg-accent-500/70" />
            <Panel
              id="inspector"
              defaultSize={28}
              minSize={20}
              collapsible
              collapsedSize={0}
            >
              <RightPanel />
            </Panel>
          </>
        )}
      </Group>
      <Footer />
    </div>
  );
}