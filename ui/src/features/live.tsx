import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { getLiveHealth, liveChunk } from "@/lib/api";

const TARGET_SR = 16000;
const CHUNK_SECONDS = 6;
const CHUNK_SAMPLES = TARGET_SR * CHUNK_SECONDS;
const SILENCE_PEAK = 0.002; // ~ -54 dBFS: below this a window is treated as silent
const WAVE_HEIGHT = 72;

// Linear-interpolation downsample of mono Float32 to 16 kHz Int16 PCM.
function downsampleToPcm16(input: Float32Array, inRate: number): Int16Array {
	const ratio = inRate / TARGET_SR;
	const outLen = Math.max(1, Math.floor(input.length / ratio));
	const out = new Int16Array(outLen);
	for (let i = 0; i < outLen; i++) {
		const s = Math.max(-1, Math.min(1, input[Math.floor(i * ratio)] ?? 0));
		out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
	}
	return out;
}

export function LiveTab() {
	const [recording, setRecording] = useState(false);
	const [transcript, setTranscript] = useState("");
	const [sending, setSending] = useState(false);
	const [level, setLevel] = useState(0); // 0..1 instantaneous mic peak
	const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
	const [deviceId, setDeviceId] = useState<string>("");
	const health = useQuery({
		queryKey: ["live-health"],
		queryFn: getLiveHealth,
		refetchInterval: 5000,
	});

	const ctxRef = useRef<AudioContext | null>(null);
	const streamRef = useRef<MediaStream | null>(null);
	const procRef = useRef<ScriptProcessorNode | null>(null);
	const sinkRef = useRef<GainNode | null>(null);
	const analyserRef = useRef<AnalyserNode | null>(null);
	const canvasRef = useRef<HTMLCanvasElement | null>(null);
	const rafRef = useRef<number | null>(null);
	const columnsRef = useRef<Array<[number, number]>>([]); // [min,max] per x-pixel
	const bufRef = useRef<Int16Array[]>([]);
	const countRef = useRef(0);
	const peakRef = useRef(0); // max |sample| in the current window
	const transcriptRef = useRef("");
	const sendingRef = useRef(false);

	const flush = async () => {
		if (sendingRef.current || countRef.current === 0) return;
		sendingRef.current = true;
		const chunks = bufRef.current;
		const windowPeak = peakRef.current;
		bufRef.current = [];
		countRef.current = 0;
		peakRef.current = 0;
		// Skip effectively-silent windows: VAD would discard them anyway and
		// it saves a GPU round-trip. The waveform still shows live input.
		if (windowPeak < SILENCE_PEAK) {
			sendingRef.current = false;
			return;
		}
		setSending(true);
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
			setSending(false);
		}
	};

	const stop = () => {
		if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
		rafRef.current = null;
		procRef.current?.disconnect();
		sinkRef.current?.disconnect();
		analyserRef.current?.disconnect();
		procRef.current = null;
		sinkRef.current = null;
		analyserRef.current = null;
		for (const t of streamRef.current?.getTracks() ?? []) t.stop();
		streamRef.current = null;
		ctxRef.current?.close().catch(() => {});
		ctxRef.current = null;
		setRecording(false);
		setLevel(0);
		flush(); // send the partial tail
	};

	const start = async () => {
		if (!navigator.mediaDevices?.getUserMedia) {
			toast.error("Microphone needs a secure context (https or localhost).");
			return;
		}
		try {
			const stream = await navigator.mediaDevices.getUserMedia({
				// Processing OFF: (1) lets multiple tabs open the SAME physical
				// device — Chromium's APM (echo/NS/AGC) doesn't cleanly share one
				// device across consumers, so a 2nd tab with it on gets silence;
				// (2) raw audio transcribes better than voice-call-filtered audio.
				audio: {
					channelCount: 1,
					echoCancellation: false,
					noiseSuppression: false,
					autoGainControl: false,
					...(deviceId ? { deviceId: { exact: deviceId } } : {}),
				},
			});
			streamRef.current = stream;
			// Labels are unlocked now that permission is granted; refresh the list.
			if (devices.every((d) => !d.label)) loadDevices();
			const ctx = new AudioContext();
			if (ctx.state === "suspended") await ctx.resume();
			ctxRef.current = ctx;
			const src = ctx.createMediaStreamSource(stream);

			const analyser = ctx.createAnalyser();
			analyser.fftSize = 1024;
			analyserRef.current = analyser;
			src.connect(analyser);

			const proc = ctx.createScriptProcessor(4096, 1, 1);
			procRef.current = proc;
			// Muted sink: ScriptProcessor only fires while connected to the graph,
			// but routing the mic to the real output would cause speaker feedback.
			const sink = ctx.createGain();
			sink.gain.value = 0;
			sinkRef.current = sink;
			proc.onaudioprocess = (ev) => {
				const input = ev.inputBuffer.getChannelData(0);
				let peak = 0;
				for (let i = 0; i < input.length; i++) {
					const a = Math.abs(input[i]);
					if (a > peak) peak = a;
				}
				setLevel(peak);
				if (peak > peakRef.current) peakRef.current = peak;
				const pcm = downsampleToPcm16(input, ctx.sampleRate);
				bufRef.current.push(pcm);
				countRef.current += pcm.length;
				if (countRef.current >= CHUNK_SAMPLES) flush();
			};
			src.connect(proc);
			proc.connect(sink);
			sink.connect(ctx.destination);

			columnsRef.current = [];
			setRecording(true);
		} catch (e) {
			toast.error(`Mic access failed: ${(e as Error).message}`);
			setRecording(false);
		}
	};

	// Enumerate audio inputs. Labels are only populated after mic permission has
	// been granted once, so we request a throwaway stream first to unlock them.
	const loadDevices = async () => {
		if (!navigator.mediaDevices?.enumerateDevices) return;
		try {
			const probe = await navigator.mediaDevices.getUserMedia({ audio: true });
			for (const t of probe.getTracks()) t.stop();
		} catch {
			// permission denied / no device — enumerate still lists deviceIds
		}
		const all = await navigator.mediaDevices.enumerateDevices();
		const mics = all.filter((d) => d.kind === "audioinput");
		setDevices(mics);
		setDeviceId((cur) => cur || mics[0]?.deviceId || "");
	};

	// biome-ignore lint/correctness/useExhaustiveDependencies: load device list once.
	useEffect(() => {
		loadDevices();
		const md = navigator.mediaDevices;
		if (!md?.addEventListener) return;
		md.addEventListener("devicechange", loadDevices);
		return () => md.removeEventListener("devicechange", loadDevices);
	}, []);

	// Audacity-style scrolling waveform: one min/max column per x-pixel, shifted
	// left each animation frame. Driven off recording state so it tears down clean.
	useEffect(() => {
		if (!recording) return;
		const canvas = canvasRef.current;
		const analyser = analyserRef.current;
		if (!canvas || !analyser) return;
		const dpr = window.devicePixelRatio || 1;
		const cssW = canvas.clientWidth || 600;
		canvas.width = Math.floor(cssW * dpr);
		canvas.height = Math.floor(WAVE_HEIGHT * dpr);
		const g = canvas.getContext("2d");
		if (!g) return;
		g.scale(dpr, dpr);
		const buf = new Float32Array(analyser.fftSize);
		const cols = columnsRef.current;

		const styles = getComputedStyle(document.documentElement);
		const wave = `oklch(${styles.getPropertyValue("--primary").trim() || "0.65 0.19 41"})`;
		const mid = "oklch(0.6 0 0 / 0.35)";

		const draw = () => {
			analyser.getFloatTimeDomainData(buf);
			let lo = 1;
			let hi = -1;
			for (let i = 0; i < buf.length; i++) {
				if (buf[i] < lo) lo = buf[i];
				if (buf[i] > hi) hi = buf[i];
			}
			cols.push([lo, hi]);
			while (cols.length > cssW) cols.shift();

			g.clearRect(0, 0, cssW, WAVE_HEIGHT);
			const cy = WAVE_HEIGHT / 2;
			g.strokeStyle = mid;
			g.beginPath();
			g.moveTo(0, cy);
			g.lineTo(cssW, cy);
			g.stroke();

			g.strokeStyle = wave;
			g.lineWidth = 1;
			g.beginPath();
			const x0 = cssW - cols.length;
			for (let i = 0; i < cols.length; i++) {
				const x = x0 + i + 0.5;
				const [mn, mx] = cols[i];
				g.moveTo(x, cy - mx * (cy - 1));
				g.lineTo(x, cy - mn * (cy - 1));
			}
			g.stroke();
			rafRef.current = requestAnimationFrame(draw);
		};
		rafRef.current = requestAnimationFrame(draw);
		return () => {
			if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
		};
	}, [recording]);

	// biome-ignore lint/correctness/useExhaustiveDependencies: tear down on unmount only.
	useEffect(() => () => stop(), []);

	const unavailable = health.data?.status === "unavailable";

	return (
		<div className="flex flex-col gap-4">
			<div className="flex flex-wrap items-center gap-3 font-mono text-xs text-muted-foreground">
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
				{recording && (
					<span className="flex items-center gap-1.5 text-foreground">
						<span className="inline-block size-2 animate-pulse rounded-full bg-destructive" />
						recording
						{sending && " · transcribing…"}
					</span>
				)}
			</div>

			{recording && (
				<div className="flex flex-col gap-2 rounded border bg-card p-2">
					<canvas
						ref={canvasRef}
						className="w-full"
						style={{ height: WAVE_HEIGHT }}
					/>
					<div className="flex items-center gap-2">
						<span className="w-8 font-mono text-[11px] text-muted-foreground">
							lvl
						</span>
						<div className="h-2 flex-1 overflow-hidden rounded-sm bg-secondary">
							<div
								className={
									level > 0.01
										? "h-full bg-emerald-500"
										: "h-full bg-muted-foreground/40"
								}
								style={{ width: `${Math.min(100, Math.round(level * 140))}%` }}
							/>
						</div>
						{level <= SILENCE_PEAK && (
							<span className="font-mono text-[11px] text-destructive">
								no signal
							</span>
						)}
					</div>
				</div>
			)}

			<div className="flex flex-wrap items-center gap-2">
				<Select
					value={deviceId}
					onValueChange={setDeviceId}
					disabled={recording}
				>
					<SelectTrigger className="w-72">
						<SelectValue placeholder="Default microphone" />
					</SelectTrigger>
					<SelectContent>
						{devices.length === 0 && (
							<SelectItem value="none" disabled>
								no input devices
							</SelectItem>
						)}
						{devices.map((d, i) => (
							<SelectItem key={d.deviceId || `mic-${i}`} value={d.deviceId}>
								{d.label || `Microphone ${i + 1}`}
							</SelectItem>
						))}
					</SelectContent>
				</Select>
				<Button
					variant="outline"
					size="icon"
					title="Refresh device list"
					onClick={loadDevices}
					disabled={recording}
				>
					↻
				</Button>
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
				placeholder={`Live transcript appears here. First result lands after ~${CHUNK_SECONDS}s of speech.`}
				className="min-h-80 font-mono text-xs leading-relaxed"
			/>
			<p className="text-xs text-muted-foreground">
				Audio is captured from your mic, downsampled to 16 kHz mono, and sent in{" "}
				{CHUNK_SECONDS}s windows to the whisper-live sidecar via /api/live. The
				waveform + level meter show live mic input — if they stay flat while you
				speak, the browser is capturing the wrong device or has no mic
				permission.
			</p>
		</div>
	);
}
