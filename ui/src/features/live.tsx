import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { getLiveHealth, liveChunk } from "@/lib/api";

const TARGET_SR = 16000;
const CHUNK_SAMPLES = TARGET_SR * 10; // ~10s per inference pass

// Linear-interpolation downsample of mono Float32 to 16 kHz Int16 PCM.
function downsampleToPcm16(input: Float32Array, inRate: number): Int16Array {
	if (inRate === TARGET_SR) {
		const out = new Int16Array(input.length);
		for (let i = 0; i < input.length; i++) {
			const s = Math.max(-1, Math.min(1, input[i]));
			out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
		}
		return out;
	}
	const ratio = inRate / TARGET_SR;
	const outLen = Math.floor(input.length / ratio);
	const out = new Int16Array(outLen);
	for (let i = 0; i < outLen; i++) {
		const s = Math.max(-1, Math.min(1, input[Math.floor(i * ratio)]));
		out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
	}
	return out;
}

export function LiveTab() {
	const [recording, setRecording] = useState(false);
	const [transcript, setTranscript] = useState("");
	const health = useQuery({
		queryKey: ["live-health"],
		queryFn: getLiveHealth,
		refetchInterval: 5000,
	});

	const ctxRef = useRef<AudioContext | null>(null);
	const streamRef = useRef<MediaStream | null>(null);
	const procRef = useRef<ScriptProcessorNode | null>(null);
	const bufRef = useRef<Int16Array[]>([]);
	const countRef = useRef(0);
	const transcriptRef = useRef("");
	const sendingRef = useRef(false);

	const flush = async () => {
		if (sendingRef.current || countRef.current === 0) return;
		sendingRef.current = true;
		const chunks = bufRef.current;
		bufRef.current = [];
		countRef.current = 0;
		const total = chunks.reduce((n, c) => n + c.length, 0);
		const merged = new Int16Array(total);
		let off = 0;
		for (const c of chunks) {
			merged.set(c, off);
			off += c.length;
		}
		try {
			const { segments } = await liveChunk(
				merged.buffer as ArrayBuffer,
				transcriptRef.current.slice(-300),
			);
			const text = segments
				.map((s) => s.text)
				.join(" ")
				.trim();
			if (text) {
				transcriptRef.current += (transcriptRef.current ? " " : "") + text;
				setTranscript(transcriptRef.current);
			}
		} catch (e) {
			toast.error((e as Error).message);
		} finally {
			sendingRef.current = false;
		}
	};

	const stop = () => {
		procRef.current?.disconnect();
		procRef.current = null;
		for (const t of streamRef.current?.getTracks() ?? []) t.stop();
		streamRef.current = null;
		ctxRef.current?.close();
		ctxRef.current = null;
		setRecording(false);
		flush();
	};

	const start = async () => {
		try {
			const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
			streamRef.current = stream;
			const ctx = new AudioContext();
			ctxRef.current = ctx;
			const src = ctx.createMediaStreamSource(stream);
			const proc = ctx.createScriptProcessor(4096, 1, 1);
			procRef.current = proc;
			proc.onaudioprocess = (ev) => {
				const pcm = downsampleToPcm16(
					ev.inputBuffer.getChannelData(0),
					ctx.sampleRate,
				);
				bufRef.current.push(pcm);
				countRef.current += pcm.length;
				if (countRef.current >= CHUNK_SAMPLES) flush();
			};
			src.connect(proc);
			proc.connect(ctx.destination);
			setRecording(true);
		} catch (e) {
			toast.error(`Mic access failed: ${(e as Error).message}`);
		}
	};

	// Cleanup on unmount.
	// biome-ignore lint/correctness/useExhaustiveDependencies: stop only tears
	// down refs; run once on unmount.
	useEffect(() => () => stop(), []);

	const unavailable = health.data?.status === "unavailable";

	return (
		<div className="flex flex-col gap-4">
			<div className="flex items-center gap-3 font-mono text-xs text-muted-foreground">
				{unavailable ? (
					<Badge variant="destructive">sidecar offline</Badge>
				) : (
					<Badge variant="secondary">
						{health.data?.model ?? "whisper-live"}
					</Badge>
				)}
				{health.data?.max_streams != null && (
					<span>
						streams {health.data.active_streams ?? 0}/{health.data.max_streams}
					</span>
				)}
			</div>

			<div className="flex gap-2">
				{!recording ? (
					<Button onClick={start} disabled={unavailable}>
						● Start mic
					</Button>
				) : (
					<Button variant="destructive" onClick={stop}>
						■ Stop
					</Button>
				)}
				<Button
					variant="outline"
					onClick={() => {
						transcriptRef.current = "";
						setTranscript("");
					}}
					disabled={!transcript}
				>
					Clear
				</Button>
			</div>

			<Textarea
				readOnly
				value={transcript}
				placeholder="Live transcript appears here (≈10s latency per pass)…"
				className="min-h-80 font-mono text-xs leading-relaxed"
			/>
			<p className="text-xs text-muted-foreground">
				Audio is captured at your device rate, downsampled to 16 kHz mono, and
				sent in ~10s windows to the whisper-live sidecar via /api/live.
			</p>
		</div>
	);
}
