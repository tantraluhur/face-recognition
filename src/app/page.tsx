"use client";

import dynamic from "next/dynamic";

const FaceComparison = dynamic(
  () => import("@/components/FaceComparison"),
  { ssr: false }
);

export default function Home() {
  return (
    <main className="flex flex-1 flex-col bg-zinc-50 dark:bg-zinc-950">
      <FaceComparison />
    </main>
  );
}
