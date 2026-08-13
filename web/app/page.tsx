import type { Metadata } from "next";
import { Arena } from "./Arena";

export const metadata: Metadata = {
  title: "Tank Trouble AI Arena",
  description: "P27b、实时规划和自博弈的本地可视化对战场。",
};

export default function Home() {
  return <Arena />;
}
