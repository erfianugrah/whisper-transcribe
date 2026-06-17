import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import {
	FORMATS,
	getJob,
	getMedia,
	LANGUAGES,
	MODELS,
	submitJob,
	type TranscribeOptions,
	uploadFile,
	ytDownload,
} from "@/lib/api";

type Source = "youtube" | "server" | "upload";

const TERMINAL = new Set(["done", "failed", "cancelled"]);

function Field({
	label,
	children,
}: {
	label: string;
	children: React.ReactNode;
}) {
	return (
		<div className="flex flex-col gap-1">
			<Label className="text-xs text-muted-foreground">{label}</Label>
			{children}
		</div>
	);
}

export function TranscribeTab() {
	const qc = useQueryClient();
	const [source, setSource] = useState<Source>("youtube");
	const [url, setUrl] = useState("");
	const [serverPath, setServerPath] = useState("");
	const [file, setFile] = useState<File | null>(null);

	const [model, setModel] = useState("turbo");
	const [format, setFormat] = useState("srt");
	const [language, setLanguage] = useState("Auto-detect");
	const [translate, setTranslate] = useState<"auto" | "true" | "false">("auto");
	const [diarize, setDiarize] = useState(false);
	const [hotwords, setHotwords] = useState("");

	const [jobId, setJobId] = useState<string | null>(null);

	const media = useQuery({
		queryKey: ["media"],
		queryFn: () => getMedia(),
		enabled: source === "server",
	});

	const job = useQuery({
		queryKey: ["job", jobId],
		queryFn: () => getJob(jobId!),
		enabled: !!jobId,
		refetchInterval: (q) =>
			q.state.data && TERMINAL.has(q.state.data.status) ? false : 1500,
	});

	const submit = useMutation({
		mutationFn: async () => {
			let file_path = "";
			let cleanup = false;
			if (source === "youtube") {
				if (!url.trim()) throw new Error("Enter a URL");
				toast.info("Downloading audio…");
				const dl = await ytDownload(url.trim());
				if (!dl.filename) throw new Error("Download returned no file");
				file_path = dl.filename;
				cleanup = true;
			} else if (source === "server") {
				if (!serverPath) throw new Error("Pick a server file");
				file_path = serverPath;
			} else {
				if (!file) throw new Error("Choose a file to upload");
				toast.info("Uploading…");
				const up = await uploadFile(file);
				file_path = up.file_path;
				cleanup = true;
			}
			const opts: TranscribeOptions = {
				file_path,
				model,
				format,
				language,
				translate,
				diarize,
				hotwords: hotwords.trim() || undefined,
				cleanup,
			};
			return submitJob(opts);
		},
		onSuccess: (r) => {
			setJobId(r.job_id);
			qc.invalidateQueries({ queryKey: ["queue"] });
			toast.success(
				`Queued ${r.job_id.slice(0, 8)} (position ${r.position ?? "?"})`,
			);
		},
		onError: (e: Error) => toast.error(e.message),
	});

	const j = job.data;
	const result = j?.result;

	return (
		<div className="grid gap-6 lg:grid-cols-[minmax(0,360px)_1fr]">
			{/* ── Controls ── */}
			<div className="flex flex-col gap-4">
				<div className="flex gap-1">
					{(["youtube", "server", "upload"] as const).map((s) => (
						<Button
							key={s}
							variant={source === s ? "default" : "outline"}
							size="sm"
							className="flex-1"
							onClick={() => setSource(s)}
						>
							{s === "youtube" ? "URL" : s === "server" ? "Server" : "Upload"}
						</Button>
					))}
				</div>

				{source === "youtube" && (
					<Field label="Video / audio URL">
						<Input
							placeholder="https://youtube.com/watch?v=…"
							value={url}
							onChange={(e) => setUrl(e.target.value)}
						/>
					</Field>
				)}
				{source === "server" && (
					<Field label={`Server file (${media.data?.files.length ?? 0} found)`}>
						<Select value={serverPath} onValueChange={setServerPath}>
							<SelectTrigger>
								<SelectValue placeholder="Select a media file…" />
							</SelectTrigger>
							<SelectContent>
								{media.data?.files.map((f) => (
									<SelectItem key={f.path} value={f.path}>
										{f.name}
									</SelectItem>
								))}
							</SelectContent>
						</Select>
					</Field>
				)}
				{source === "upload" && (
					<Field label="Upload file">
						<Input
							type="file"
							accept="audio/*,video/*"
							onChange={(e) => setFile(e.target.files?.[0] ?? null)}
						/>
					</Field>
				)}

				<div className="grid grid-cols-2 gap-3">
					<Field label="Model">
						<Select value={model} onValueChange={setModel}>
							<SelectTrigger>
								<SelectValue />
							</SelectTrigger>
							<SelectContent>
								{MODELS.map((m) => (
									<SelectItem key={m} value={m}>
										{m}
									</SelectItem>
								))}
							</SelectContent>
						</Select>
					</Field>
					<Field label="Format">
						<Select value={format} onValueChange={setFormat}>
							<SelectTrigger>
								<SelectValue />
							</SelectTrigger>
							<SelectContent>
								{FORMATS.map((f) => (
									<SelectItem key={f} value={f}>
										{f}
									</SelectItem>
								))}
							</SelectContent>
						</Select>
					</Field>
					<Field label="Language">
						<Select value={language} onValueChange={setLanguage}>
							<SelectTrigger>
								<SelectValue />
							</SelectTrigger>
							<SelectContent>
								{LANGUAGES.map((l) => (
									<SelectItem key={l} value={l}>
										{l}
									</SelectItem>
								))}
							</SelectContent>
						</Select>
					</Field>
					<Field label="Translate → EN">
						<Select
							value={translate}
							onValueChange={(v) => setTranslate(v as typeof translate)}
						>
							<SelectTrigger>
								<SelectValue />
							</SelectTrigger>
							<SelectContent>
								<SelectItem value="auto">auto</SelectItem>
								<SelectItem value="true">always</SelectItem>
								<SelectItem value="false">never</SelectItem>
							</SelectContent>
						</Select>
					</Field>
				</div>

				<label className="flex items-center gap-2 text-sm">
					<input
						type="checkbox"
						checked={diarize}
						onChange={(e) => setDiarize(e.target.checked)}
					/>
					Speaker diarization
				</label>

				<Field label="Hotwords (comma-separated)">
					<Input
						value={hotwords}
						onChange={(e) => setHotwords(e.target.value)}
						placeholder="optional"
					/>
				</Field>

				<Button onClick={() => submit.mutate()} disabled={submit.isPending}>
					{submit.isPending ? "Submitting…" : "Transcribe"}
				</Button>
			</div>

			{/* ── Result ── */}
			<div className="min-w-0">
				{!jobId && (
					<div className="flex h-full min-h-40 items-center justify-center rounded border border-dashed text-sm text-muted-foreground">
						Submit a job to see the transcript here.
					</div>
				)}
				{jobId && (
					<div className="flex flex-col gap-3">
						<div className="flex items-center gap-3 font-mono text-xs">
							<Badge
								variant={
									j && j.status === "failed" ? "destructive" : "secondary"
								}
							>
								{j?.status ?? "loading"}
							</Badge>
							<span className="text-muted-foreground">{jobId.slice(0, 8)}</span>
							{j?.status === "queued" && (
								<span>position {j.position ?? "?"}</span>
							)}
							{j?.status === "running" && (
								<span className="text-primary">transcribing…</span>
							)}
						</div>

						{j?.status === "failed" && (
							<pre className="overflow-auto rounded border border-destructive/40 bg-destructive/5 p-3 text-xs text-destructive">
								{j.error || "unknown error"}
							</pre>
						)}

						{result?.transcript && (
							<>
								<div className="flex gap-2">
									<Button
										size="sm"
										variant="outline"
										onClick={() => {
											navigator.clipboard.writeText(result.transcript!);
											toast.success("Copied transcript");
										}}
									>
										Copy
									</Button>
									<Button
										size="sm"
										variant="outline"
										onClick={() => {
											const blob = new Blob([result.transcript!], {
												type: "text/plain",
											});
											const a = document.createElement("a");
											a.href = URL.createObjectURL(blob);
											a.download = `transcript-${jobId.slice(0, 8)}.txt`;
											a.click();
											URL.revokeObjectURL(a.href);
										}}
									>
										Download .txt
									</Button>
									{result.task && (
										<Badge variant="outline">{result.task}</Badge>
									)}
								</div>
								<Textarea
									readOnly
									value={result.transcript}
									className="min-h-80 font-mono text-xs leading-relaxed"
								/>
							</>
						)}
					</div>
				)}
			</div>
		</div>
	);
}
