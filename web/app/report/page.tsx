import type { Metadata } from "next";
import { Report } from "./Report";

export const metadata: Metadata = {
  title: "Tank Trouble AI — Report",
  description: "方法与结果：自己设计的观测、联赛 PPO、MPC 老师与实时搜索。",
};

export default function ReportPage() {
  return <Report />;
}
