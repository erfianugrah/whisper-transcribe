import { useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { ThemeToggle } from "@/components/theme-toggle";
import { HistoryTab } from "@/features/history";
import { LiveTab } from "@/features/live";
import { QueueTab } from "@/features/queue";
import { TranscribeTab } from "@/features/transcribe";
import { getStatus } from "@/lib/api";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/")({ component: Home });

function StatusBar() {
	const { data, isError } = useQuery({
		queryKey: ["status"],
		queryFn: getStatus,
		refetchInterval: 5000,
	});
	const dot = isError
		? "bg-destructive"
		: data?.busy
			? "bg-primary"
			: "bg-emerald-500";
	return (
		<div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px] text-muted-foreground">
			<span className="flex items-center gap-1.5 text-foreground">
				<span className={cn("inline-block size-2 rounded-full", dot)} />
				{isError ? "offline" : data?.busy ? "busy" : "ready"}
			</span>
			{data && (
				<>
					<span className="text-border">|</span>
					<span>{data.gpu}</span>
					<span className="text-border">|</span>
					<span>
						{data.device}/{data.compute_type}
					</span>
					<span>batch {data.default_batch_size}</span>
					<span>diarize {data.diarization_available ? "on" : "off"}</span>
				</>
			)}
		</div>
	);
}

const TABS = [
	{ id: "transcribe", label: "Transcribe" },
	{ id: "queue", label: "Queue" },
	{ id: "history", label: "History" },
	{ id: "live", label: "Live" },
] as const;
type TabId = (typeof TABS)[number]["id"];

function initialTab(): TabId {
	const h = window.location.hash.replace("#", "");
	return TABS.some((t) => t.id === h) ? (h as TabId) : "transcribe";
}

function Home() {
	const [tab, setTabState] = useState<TabId>(initialTab);
	const setTab = (id: TabId) => {
		setTabState(id);
		// Reflect in the URL hash so a refresh restores the same tab.
		window.history.replaceState(null, "", `#${id}`);
	};
	return (
		<div className="mx-auto max-w-[1600px] px-6 py-5">
			<header className="mb-4">
				<div className="flex items-center justify-between gap-4">
					<h1 className="text-base font-bold tracking-tight">
						whisper<span className="text-primary">·</span>transcribe
					</h1>
					<div className="flex items-center gap-3">
						<ThemeToggle />
						<a
							href="/classic/"
							className="font-mono text-[11px] text-muted-foreground underline-offset-2 hover:underline"
						>
							classic UI →
						</a>
					</div>
				</div>
				<div className="mt-2">
					<StatusBar />
				</div>
			</header>

			<nav className="flex items-center gap-1 border-b">
				{TABS.map((t) => (
					<button
						key={t.id}
						type="button"
						onClick={() => setTab(t.id)}
						className={cn(
							"-mb-px border-b-2 px-3 py-2 text-sm font-medium transition-colors",
							tab === t.id
								? "border-primary text-foreground"
								: "border-transparent text-muted-foreground hover:text-foreground",
						)}
					>
						{t.label}
					</button>
				))}
			</nav>

			<main className="py-5">
				{tab === "transcribe" && <TranscribeTab />}
				{tab === "queue" && <QueueTab />}
				{tab === "history" && <HistoryTab />}
				{tab === "live" && <LiveTab />}
			</main>
		</div>
	);
}
