import { Header } from "@/components/layout/Header";
import { Sidebar } from "@/components/layout/Sidebar";
import { Hero } from "@/components/dashboard/Hero";
import { AIAdvisor } from "@/components/dashboard/AIAdvisor";
import { QuestionGrid } from "@/components/dashboard/QuestionGrid";
import { KPIGrid } from "@/components/dashboard/KPIGrid";

export default function Dashboard() {
  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar />

      <main className="flex flex-1 flex-col">
        <Header />

        <div className="mx-auto w-full max-w-7xl px-8 py-8">
            <Hero />
            <AIAdvisor />
            <KPIGrid />
            <QuestionGrid />
        </div>
      </main>
    </div>
  );
}