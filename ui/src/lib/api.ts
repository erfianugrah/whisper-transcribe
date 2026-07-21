import { z } from "zod";

// API is same-origin: the SPA is served at /ui, endpoints live at /api/*.
const API = "";

async function jget<T>(schema: z.ZodType<T>, path: string): Promise<T> {
	const res = await fetch(`${API}${path}`);
	if (!res.ok) throw new Error(`${path} → ${res.status}`);
	return schema.parse(await res.json());
}

async function jpost<T>(
	schema: z.ZodType<T>,
	path: string,
	body: unknown,
): Promise<T> {
	const res = await fetch(`${API}${path}`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(body),
	});
	const json = await res.json().catch(() => ({}));
	if (!res.ok) {
		throw new Error(
			(json as { error?: string }).error || `${path} → ${res.status}`,
		);
	}
	return schema.parse(json);
}

// ── /api/status ──────────────────────────────────────────────────────────────
export const StatusSchema = z.object({
	status: z.string(),
	busy: z.boolean(),
	gpu: z.string(),
	device: z.string(),
	compute_type: z.string(),
	diarization_available: z.boolean(),
	default_batch_size: z.number(),
	vision: z
		.object({
			available: z.boolean(),
			model: z.string(),
			fps_interval: z.number().nullish(),
			max_frames: z.number().nullish(),
		})
		.partial()
		.nullish(),
});
export type Status = z.infer<typeof StatusSchema>;
export const getStatus = () => jget(StatusSchema, "/api/status");

// ── /api/queue ───────────────────────────────────────────────────────────────
const JobLike = z
	.object({
		id: z.string().nullish(),
		status: z.string().nullish(),
		consumer: z.string().nullish(),
		submitted_at: z.union([z.number(), z.string()]).nullish(),
		started_at: z.union([z.number(), z.string()]).nullish(),
		completed_at: z.union([z.number(), z.string()]).nullish(),
		position: z.number().nullish(),
		error: z.string().nullish(),
	})
	.passthrough();

export const QueueSchema = z.object({
	depth: z.number(),
	active: z.array(JobLike),
	recent: z.array(JobLike),
	available: z.boolean(),
});
export type Queue = z.infer<typeof QueueSchema>;
export const getQueue = () => jget(QueueSchema, "/api/queue");

// ── /api/history ─────────────────────────────────────────────────────────────
export const HistorySchema = z.array(
	z
		.object({
			timestamp: z.string().nullish(),
			filename: z.string().nullish(),
			duration_str: z.string().nullish(),
			language: z.string().nullish(),
			speakers: z.union([z.number(), z.string()]).nullish(),
			speed: z.string().nullish(),
			segments: z.union([z.number(), z.string()]).nullish(),
		})
		.passthrough(),
);
export type History = z.infer<typeof HistorySchema>;
export const getHistory = () => jget(HistorySchema, "/api/history");

// ── /api/media ───────────────────────────────────────────────────────────────
export const MediaSchema = z.object({
	files: z.array(
		z.object({
			name: z.string(),
			path: z.string(),
			possible_duplicate_of: z.string().optional(),
		}),
	),
});
export const getMedia = (refresh = false) =>
	jget(MediaSchema, `/api/media${refresh ? "?refresh=1" : ""}`);

// ── /api/jobs ────────────────────────────────────────────────────────────────
export const SubmitSchema = z.object({
	job_id: z.string(),
	status: z.string(),
	submitted_at: z.union([z.number(), z.string()]).nullish(),
	position: z.number().nullish(),
});

export interface TranscribeOptions {
	file_path: string;
	model: string;
	format: string;
	language: string;
	translate: "auto" | "true" | "false";
	diarize: boolean;
	min_speakers?: number;
	max_speakers?: number;
	batch_size?: number;
	hotwords?: string;
	initial_prompt?: string;
	suppress_numerals?: boolean;
	cleanup?: boolean;
	consumer?: string;
}

export const submitJob = (opts: TranscribeOptions) =>
	jpost(SubmitSchema, "/api/jobs", {
		...opts,
		translate: opts.translate === "auto" ? "auto" : opts.translate === "true",
		consumer: opts.consumer || "web-ui",
	});

export const JobSchema = z
	.object({
		status: z.string(),
		position: z.number().nullish(),
		submitted_at: z.union([z.number(), z.string()]).nullish(),
		started_at: z.union([z.number(), z.string()]).nullish(),
		completed_at: z.union([z.number(), z.string()]).nullish(),
		error: z.string().nullish(),
		permanent: z.boolean().nullish(),
		result: z
			.object({
				status: z.string().nullish(),
				transcript: z.string().nullish(),
				subtitle_file: z.string().nullable().nullish(),
				subtitle_content: z.string().nullish(),
				subtitle_name: z.string().nullish(),
				format: z.string().nullish(),
				task: z.string().nullish(),
			})
			.passthrough()
			.nullish(),
	})
	.passthrough();
export type Job = z.infer<typeof JobSchema>;
export const getJob = (id: string) => jget(JobSchema, `/api/jobs/${id}`);

export async function cancelJob(id: string): Promise<void> {
	const res = await fetch(`${API}/api/jobs/${id}`, { method: "DELETE" });
	if (!res.ok) {
		const j = await res.json().catch(() => ({}));
		throw new Error(
			(j as { error?: string }).error || `cancel → ${res.status}`,
		);
	}
}

// ── /api/yt-download ─────────────────────────────────────────────────────────
export const YtDownloadSchema = z
	.object({
		filename: z.string().nullish(),
		title: z.string().nullish(),
		duration: z.union([z.number(), z.string()]).nullish(),
	})
	.passthrough();

export const ytDownload = (url: string, keep_video = false) =>
	jpost(YtDownloadSchema, "/api/yt-download", { url, keep_video });

// ── /api/upload ──────────────────────────────────────────────────────────────
export const UploadSchema = z.object({
	file_path: z.string(),
	filename: z.string(),
	size: z.number(),
});
export async function uploadFile(
	file: File,
): Promise<z.infer<typeof UploadSchema>> {
	const fd = new FormData();
	fd.append("file", file);
	const res = await fetch(`${API}/api/upload`, { method: "POST", body: fd });
	const json = await res.json().catch(() => ({}));
	if (!res.ok)
		throw new Error(
			(json as { error?: string }).error || `upload → ${res.status}`,
		);
	return UploadSchema.parse(json);
}

// ── /api/voiceprints ─────────────────────────────────────────────────────────
// Enrolled voice prints power server-side speaker naming: matched speakers
// are relabeled to real names in every transcript (this SPA included).
export const VoiceprintsSchema = z.object({
	voiceprints: z.array(z.object({ name: z.string(), count: z.number() })),
});
export type Voiceprints = z.infer<typeof VoiceprintsSchema>;
export const getVoiceprints = () => jget(VoiceprintsSchema, "/api/voiceprints");

export const AddVoiceprintSchema = z.object({
	name: z.string(),
	count: z.number(),
	dim: z.number().nullish(),
});
export const addVoiceprint = (body: {
	name: string;
	file_path: string;
	start?: number;
	end?: number;
}) => jpost(AddVoiceprintSchema, "/api/voiceprints", body);

export async function deleteVoiceprint(name: string): Promise<void> {
	const res = await fetch(
		`${API}/api/voiceprints/${encodeURIComponent(name)}`,
		{
			method: "DELETE",
		},
	);
	if (!res.ok) throw new Error(`delete voiceprint → ${res.status}`);
}

// ── /api/vocabulary ──────────────────────────────────────────────────────────
// Persistent hotword vocabulary auto-injected into every transcription job
// (alongside enrolled voice-print names). One term per line server-side.
export const VocabularySchema = z.object({
	terms: z.array(z.string()),
	file: z.string(),
	auto_hotwords: z.boolean(),
	max_terms: z.number(),
	voiceprint_names: z.array(z.string()),
});
export type Vocabulary = z.infer<typeof VocabularySchema>;
export const getVocabulary = () => jget(VocabularySchema, "/api/vocabulary");

export async function putVocabulary(terms: string[]): Promise<Vocabulary> {
	const res = await fetch(`${API}/api/vocabulary`, {
		method: "PUT",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ terms }),
	});
	const json = await res.json().catch(() => ({}));
	if (!res.ok)
		throw new Error(
			(json as { error?: string }).error || `vocabulary → ${res.status}`,
		);
	return VocabularySchema.parse(json);
}

// ── /api/live ────────────────────────────────────────────────────────────────
export const LiveHealthSchema = z
	.object({
		status: z.string(),
		model: z.string().nullish(),
		max_streams: z.number().nullish(),
		active_streams: z.number().nullish(),
	})
	.passthrough();
export const getLiveHealth = () => jget(LiveHealthSchema, "/api/live/health");

// ── /api/research ──────────────────────────────────────────────────────────────
// SSE streaming: yields text chunks as they arrive from the LLM.
export async function* streamResearch(opts: {
	question: string;
	context: string;
	model?: string;
}): AsyncGenerator<string> {
	const res = await fetch(`${API}/api/research`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(opts),
	});
	if (!res.ok || !res.body) throw new Error(`research → ${res.status}`);
	const reader = res.body.getReader();
	const dec = new TextDecoder();
	let buf = "";
	while (true) {
		const { done, value } = await reader.read();
		if (done) break;
		buf += dec.decode(value, { stream: true });
		const lines = buf.split("\n");
		buf = lines.pop() ?? "";
		for (const line of lines) {
			if (!line.startsWith("data:")) continue;
			// Strip the "data:" prefix + exactly ONE optional SSE framing space.
			// Do NOT trim: streamed tokens carry leading/trailing spaces that are
			// part of the content (e.g. " management") — trimming concatenates words.
			let payload = line.slice(5);
			if (payload.startsWith(" ")) payload = payload.slice(1);
			if (payload === "[DONE]") return;
			if (payload.startsWith("[ERROR]"))
				throw new Error(payload.slice(7).trim());
			// server escapes \n inside chunks so one SSE line = one delta
			yield payload.replace(/\\n/g, "\n");
		}
	}
}

export async function liveChunk(
	pcm: ArrayBuffer,
	context: string,
): Promise<{ segments: { text: string; start?: number; end?: number }[] }> {
	const qs = new URLSearchParams({ context }).toString();
	const res = await fetch(`${API}/api/live/transcribe-chunk?${qs}`, {
		method: "POST",
		headers: { "Content-Type": "application/octet-stream" },
		body: pcm,
	});
	if (!res.ok) throw new Error(`live chunk → ${res.status}`);
	return res.json();
}

export const LANGUAGES = [
	"Auto-detect",
	"en",
	"zh",
	"de",
	"es",
	"ru",
	"ko",
	"fr",
	"ja",
	"pt",
	"tr",
	"pl",
	"ca",
	"nl",
	"ar",
	"sv",
	"it",
	"id",
	"hi",
	"fi",
	"vi",
	"he",
	"uk",
	"el",
	"ms",
	"cs",
	"ro",
	"da",
	"hu",
	"ta",
	"no",
	"th",
	"ur",
	"hr",
	"bg",
	"lt",
	"la",
	"mi",
	"ml",
	"cy",
	"sk",
	"te",
	"fa",
] as const;

export const MODELS = [
	"tiny",
	"base",
	"small",
	"medium",
	"large",
	"turbo",
] as const;
export const FORMATS = ["txt", "srt", "vtt", "json"] as const;
