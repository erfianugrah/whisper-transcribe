import { useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { HistoryTab } from "@/features/history";
import { LiveTab } from "@/features/live";
import { QueueTab } from "@/features/queue";
import { TranscribeTab } from "@/features/transcribe";
import { getStatus } from "@/lib/api";

export const Route = createFileRoute("/")({ component: Home });

function StatusBar() {
	const { data, isError } = useQuery({
		queryKey: ["status"],
		queryFn: getStatus,
		refetchInterval: 5000,
	});
	return (
		<div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-xs text-muted-foreground">
			<span className="flex items-center gap-1.5">
				<span
					className={
						isError
							? "inline-block size-2 rounded-full bg-destructive"
							: data?.busy
								? "inline-block size-2 rounded-full bg-primary"
								: "inline-block size-2 rounded-full bg-emerald-500"
					}
				/>
				{isError ? "offline" : data?.busy ? "busy" : "ready"}
			</span>
			{data && (
				<>
					<span>{data.gpu}</span>
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

function Home() {
	return (
		<div className="mx-auto max-w-5xl px-4 py-6">
			<header className="mb-5 border-b pb-4">
				<div className="flex items-baseline justify-between">
					<h1 className="text-lg font-bold tracking-tight">
						whisper<span className="text-primary">·</span>transcribe
					</h1>
					<a
						href="/"
						className="font-mono text-xs text-muted-foreground underline-offset-2 hover:underline"
					>
						classic UI →
					</a>
				</div>
				<div className="mt-2">
					<StatusBar />
				</div>
			</header>

			<Tabs defaultValue="transcribe">
				<TabsList>
					<TabsTrigger value="transcribe">Transcribe</TabsTrigger>
					<TabsTrigger value="queue">Queue</TabsTrigger>
					<TabsTrigger value="history">History</TabsTrigger>
					<TabsTrigger value="live">Live</TabsTrigger>
				</TabsList>
				<TabsContent value="transcribe" className="mt-4">
					<TranscribeTab />
				</TabsContent>
				<TabsContent value="queue" className="mt-4">
					<QueueTab />
				</TabsContent>
				<TabsContent value="history" className="mt-4">
					<HistoryTab />
				</TabsContent>
				<TabsContent value="live" className="mt-4">
					<LiveTab />
				</TabsContent>
			</Tabs>
		</div>
	);
}
