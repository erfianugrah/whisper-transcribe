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
			fps_interval: z.number().optional(),
			max_frames: z.number().optional(),
		})
		.partial()
		.optional(),
});
export type Status = z.infer<typeof StatusSchema>;
export const getStatus = () => jget(StatusSchema, "/api/status");

// ── /api/queue ───────────────────────────────────────────────────────────────
const JobLike = z
	.object({
		id: z.string().optional(),
		status: z.string().optional(),
		consumer: z.string().optional(),
		submitted_at: z.union([z.number(), z.string()]).optional(),
		started_at: z.union([z.number(), z.string()]).optional(),
		completed_at: z.union([z.number(), z.string()]).optional(),
		position: z.number().optional(),
		error: z.string().optional(),
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
			timestamp: z.string().optional(),
			filename: z.string().optional(),
			duration_str: z.string().optional(),
			language: z.string().optional(),
			speakers: z.union([z.number(), z.string()]).optional(),
			speed: z.string().optional(),
			segments: z.union([z.number(), z.string()]).optional(),
		})
		.passthrough(),
);
export type History = z.infer<typeof HistorySchema>;
export const getHistory = () => jget(HistorySchema, "/api/history");

// ── /api/media ───────────────────────────────────────────────────────────────
export const MediaSchema = z.object({
	files: z.array(z.object({ name: z.string(), path: z.string() })),
});
export const getMedia = (refresh = false) =>
	jget(MediaSchema, `/api/media${refresh ? "?refresh=1" : ""}`);

// ── /api/jobs ────────────────────────────────────────────────────────────────
export const SubmitSchema = z.object({
	job_id: z.string(),
	status: z.string(),
	submitted_at: z.union([z.number(), z.string()]).optional(),
	position: z.number().optional(),
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
		position: z.number().optional(),
		submitted_at: z.union([z.number(), z.string()]).optional(),
		started_at: z.union([z.number(), z.string()]).optional(),
		completed_at: z.union([z.number(), z.string()]).optional(),
		error: z.string().optional(),
		permanent: z.boolean().optional(),
		result: z
			.object({
				status: z.string().optional(),
				transcript: z.string().optional(),
				subtitle_file: z.string().nullable().optional(),
				task: z.string().optional(),
			})
			.passthrough()
			.optional(),
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
		filename: z.string().optional(),
		title: z.string().optional(),
		duration: z.union([z.number(), z.string()]).optional(),
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

// ── /api/live ────────────────────────────────────────────────────────────────
export const LiveHealthSchema = z
	.object({
		status: z.string(),
		model: z.string().optional(),
		max_streams: z.number().optional(),
		active_streams: z.number().optional(),
	})
	.passthrough();
export const getLiveHealth = () => jget(LiveHealthSchema, "/api/live/health");

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
