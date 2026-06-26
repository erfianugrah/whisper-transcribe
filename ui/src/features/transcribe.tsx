import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
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

// ── Transcription options ────────────────────────────────────────────────
// Everything the worker accepts lives in one object so adding a knob is a
// single key here + one <Field> in the form below — no new useState wiring.
interface Opts {
	model: string;
	format: string;
	language: string;
	translate: "auto" | "true" | "false";
	diarize: boolean;
	minSpeakers: string;
	maxSpeakers: string;
	batchSize: string;
	hotwords: string;
	initialPrompt: string;
	suppressNumerals: boolean;
}

const DEFAULT_OPTS: Opts = {
	model: "turbo",
	format: "srt",
	language: "Auto-detect",
	translate: "auto",
	diarize: false,
	minSpeakers: "",
	maxSpeakers: "",
	batchSize: "",
	hotwords: "",
	initialPrompt: "",
	suppressNumerals: false,
};

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

function SelectField({
	label,
	value,
	onChange,
	options,
}: {
	label: string;
	value: string;
	onChange: (v: string) => void;
	options: readonly string[];
}) {
	return (
		<Field label={label}>
			<Select value={value} onValueChange={onChange}>
				<SelectTrigger>
					<SelectValue />
				</SelectTrigger>
				<SelectContent>
					{options.map((o) => (
						<SelectItem key={o} value={o}>
							{o}
						</SelectItem>
					))}
				</SelectContent>
			</Select>
		</Field>
	);
}

// Whole-label SPEAKER_xx → friendly-name replace. Works on plain transcript,
// srt/vtt ("[SPEAKER_xx]") and json ('"speaker": "SPEAKER_xx"') alike.
function applyRenames(text: string, renames: Record<string, string>): string {
	let out = text;
	for (const [label, name] of Object.entries(renames)) {
		if (name.trim()) out = out.replaceAll(label, name.trim());
	}
	return out;
}

function download(name: string, content: string, mime = "text/plain") {
	const blob = new Blob([content], { type: mime });
	const a = document.createElement("a");
	a.href = URL.createObjectURL(blob);
	a.download = name;
	a.click();
	URL.revokeObjectURL(a.href);
}

export function TranscribeTab() {
	const qc = useQueryClient();
	const [source, setSource] = useState<Source>("youtube");
	const [url, setUrl] = useState("");
	const [serverPath, setServerPath] = useState("");
	const [file, setFile] = useState<File | null>(null);

	const [opts, setOpts] = useState<Opts>(DEFAULT_OPTS);
	const set = <K extends keyof Opts>(key: K, value: Opts[K]) =>
		setOpts((o) => ({ ...o, [key]: value }));

	const [jobId, setJobId] = useState<string | null>(null);
	// Client-side SPEAKER_xx → friendly-name remap applied to the result.
	const [renames, setRenames] = useState<Record<string, string>>({});
	const [filter, setFilter] = useState("");

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
			const payload: TranscribeOptions = {
				file_path,
				model: opts.model,
				format: opts.format,
				language: opts.language,
				translate: opts.translate,
				diarize: opts.diarize,
				min_speakers:
					opts.diarize && opts.minSpeakers
						? Number(opts.minSpeakers)
						: undefined,
				max_speakers:
					opts.diarize && opts.maxSpeakers
						? Number(opts.maxSpeakers)
						: undefined,
				batch_size: opts.batchSize ? Number(opts.batchSize) : undefined,
				hotwords: opts.hotwords.trim() || undefined,
				initial_prompt: opts.initialPrompt.trim() || undefined,
				suppress_numerals: opts.suppressNumerals || undefined,
				cleanup,
			};
			return submitJob(payload);
		},
		onSuccess: (r) => {
			setJobId(r.job_id);
			setRenames({});
			setFilter("");
			qc.invalidateQueries({ queryKey: ["queue"] });
			toast.success(
				`Queued ${r.job_id.slice(0, 8)} (position ${r.position ?? "?"})`,
			);
		},
		onError: (e: Error) => toast.error(e.message),
	});

	const j = job.data;
	const result = j?.result;

	// Distinct SPEAKER_xx labels present in the diarized transcript.
	const speakers = useMemo(() => {
		if (!result?.transcript) return [] as string[];
		const found = new Set(result.transcript.match(/SPEAKER_\d+/g) ?? []);
		return [...found].sort();
	}, [result?.transcript]);

	// Renamed transcript for display / copy / .txt download.
	const displayed = useMemo(
		() => (result?.transcript ? applyRenames(result.transcript, renames) : ""),
		[result?.transcript, renames],
	);

	// Lines filtered by the in-transcript search box.
	const filteredLines = useMemo(() => {
		if (!filter.trim()) return null;
		const q = filter.toLowerCase();
		return displayed.split("\n").filter((l) => l.toLowerCase().includes(q));
	}, [displayed, filter]);

	const fmt = result?.format || opts.format;

	return (
		<div className="grid gap-4 lg:grid-cols-[minmax(0,360px)_1fr] lg:items-start">
			{/* ── Controls ── */}
			<div className="flex flex-col gap-3 rounded border bg-card p-3">
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
						<div className="flex gap-2">
							<Select value={serverPath} onValueChange={setServerPath}>
								<SelectTrigger className="flex-1">
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
							<Button
								variant="outline"
								size="icon"
								title="Rescan /media"
								disabled={media.isFetching}
								onClick={async () => {
									await getMedia(true);
									qc.invalidateQueries({ queryKey: ["media"] });
								}}
							>
								↻
							</Button>
						</div>
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
					<SelectField
						label="Model"
						value={opts.model}
						onChange={(v) => set("model", v)}
						options={MODELS}
					/>
					<SelectField
						label="Format"
						value={opts.format}
						onChange={(v) => set("format", v)}
						options={FORMATS}
					/>
					<SelectField
						label="Language"
						value={opts.language}
						onChange={(v) => set("language", v)}
						options={LANGUAGES}
					/>
					<Field label="Translate → EN">
						<Select
							value={opts.translate}
							onValueChange={(v) => set("translate", v as Opts["translate"])}
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

				<Field label="Batch size (blank = auto / VRAM-derived)">
					<Input
						type="number"
						min={1}
						max={64}
						value={opts.batchSize}
						onChange={(e) => set("batchSize", e.target.value)}
						placeholder="auto"
					/>
				</Field>

				<label className="flex items-center gap-2 text-sm">
					<input
						type="checkbox"
						checked={opts.diarize}
						onChange={(e) => set("diarize", e.target.checked)}
					/>
					Speaker diarization
				</label>

				{opts.diarize && (
					<div className="grid grid-cols-2 gap-3">
						<Field label="Min speakers">
							<Input
								type="number"
								min={1}
								value={opts.minSpeakers}
								onChange={(e) => set("minSpeakers", e.target.value)}
								placeholder="auto"
							/>
						</Field>
						<Field label="Max speakers">
							<Input
								type="number"
								min={1}
								value={opts.maxSpeakers}
								onChange={(e) => set("maxSpeakers", e.target.value)}
								placeholder="auto"
							/>
						</Field>
					</div>
				)}

				<label className="flex items-center gap-2 text-sm">
					<input
						type="checkbox"
						checked={opts.suppressNumerals}
						onChange={(e) => set("suppressNumerals", e.target.checked)}
					/>
					Suppress numerals (spell out numbers)
				</label>

				<Field label="Hotwords (comma-separated)">
					<Input
						value={opts.hotwords}
						onChange={(e) => set("hotwords", e.target.value)}
						placeholder="optional"
					/>
				</Field>

				<Field label="Initial prompt (context / spelling hints)">
					<Textarea
						value={opts.initialPrompt}
						onChange={(e) => set("initialPrompt", e.target.value)}
						placeholder="optional — biases the decoder toward this vocabulary/style"
						className="min-h-16 text-xs"
					/>
				</Field>

				<Button onClick={() => submit.mutate()} disabled={submit.isPending}>
					{submit.isPending ? "Submitting…" : "Transcribe"}
				</Button>
			</div>

			{/* ── Result ── */}
			<div className="min-w-0 rounded border bg-card p-3 lg:min-h-[calc(100vh-12rem)]">
				{!jobId && (
					<div className="flex h-full min-h-80 items-center justify-center text-sm text-muted-foreground">
						Submit a job to see the transcript here.
					</div>
				)}
				{jobId && (
					<div className="flex h-full flex-col gap-3">
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
								<div className="flex flex-wrap gap-2">
									<Button
										size="sm"
										variant="outline"
										onClick={() => {
											navigator.clipboard.writeText(displayed);
											toast.success("Copied transcript");
										}}
									>
										Copy
									</Button>
									<Button
										size="sm"
										variant="outline"
										onClick={() =>
											download(`transcript-${jobId.slice(0, 8)}.txt`, displayed)
										}
									>
										Download .txt
									</Button>
									{result.subtitle_content && fmt !== "txt" && (
										<Button
											size="sm"
											onClick={() =>
												download(
													`transcript-${jobId.slice(0, 8)}.${fmt}`,
													applyRenames(result.subtitle_content ?? "", renames),
													fmt === "json" ? "application/json" : "text/plain",
												)
											}
										>
											Download .{fmt}
										</Button>
									)}
									{result.task && (
										<Badge variant="outline">{result.task}</Badge>
									)}
								</div>

								{speakers.length > 0 && (
									<div className="flex flex-col gap-2 rounded border bg-muted/30 p-2">
										<Label className="text-xs text-muted-foreground">
											Rename speakers (applies to copy + all downloads)
										</Label>
										<div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
											{speakers.map((s) => (
												<Input
													key={s}
													value={renames[s] ?? ""}
													placeholder={s}
													onChange={(e) =>
														setRenames((r) => ({ ...r, [s]: e.target.value }))
													}
												/>
											))}
										</div>
									</div>
								)}

								<Input
									value={filter}
									onChange={(e) => setFilter(e.target.value)}
									placeholder="Search transcript…"
									className="text-xs"
								/>

								{filteredLines ? (
									<div className="min-h-80 flex-1 overflow-auto rounded border bg-background p-3 font-mono text-xs leading-relaxed">
										<div className="mb-2 text-[10px] text-muted-foreground">
											{filteredLines.length} matching line
											{filteredLines.length === 1 ? "" : "s"}
										</div>
										{filteredLines.length === 0 ? (
											<span className="text-muted-foreground">no matches</span>
										) : (
											filteredLines.map((l, i) => (
												// biome-ignore lint/suspicious/noArrayIndexKey: read-only display list, lines can duplicate
												<p key={i} className="whitespace-pre-wrap">
													{l}
												</p>
											))
										)}
									</div>
								) : (
									<Textarea
										readOnly
										value={displayed}
										className="min-h-80 flex-1 font-mono text-xs leading-relaxed"
									/>
								)}
							</>
						)}
					</div>
				)}
			</div>
		</div>
	);
}
