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
import { getLiveHealth } from "@/lib/api";

const TARGET_SR = 16000;
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

function streamUrl(): string {
	const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
	return `${proto}//${window.location.host}/api/live/stream`;
}

export function LiveTab() {
	const [recording, setRecording] = useState(false);
	const [committed, setCommitted] = useState("");
	const [partial, setPartial] = useState("");
	const [trackInfo, setTrackInfo] = useState("");
	const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
	const [deviceId, setDeviceId] = useState<string>("");
	// "mic" → getUserMedia (a selected input device); "system" → getDisplayMedia
	// (desktop / tab / OBS audio — the user picks a surface and ticks "share
	// audio"). Both feed the identical downsample → /ws-stream graph below.
	const [source, setSource] = useState<"mic" | "system">("mic");
	const health = useQuery({
		queryKey: ["live-health"],
		queryFn: getLiveHealth,
		refetchInterval: 5000,
	});

	const ctxRef = useRef<AudioContext | null>(null);
	const streamRef = useRef<MediaStream | null>(null);
	const procRef = useRef<ScriptProcessorNode | null>(null);
	const sinkRef = useRef<GainNode | null>(null);
	// Firefox GCs an unreferenced MediaStreamAudioSourceNode even while it's
	// connected, silencing the whole graph. Hold a ref to keep it alive.
	const srcRef = useRef<MediaStreamAudioSourceNode | null>(null);
	const analyserRef = useRef<AnalyserNode | null>(null);
	const canvasRef = useRef<HTMLCanvasElement | null>(null);
	const rafRef = useRef<number | null>(null);
	const columnsRef = useRef<Array<[number, number]>>([]);
	const wsRef = useRef<WebSocket | null>(null);
	const committedRef = useRef("");

	const stop = () => {
		if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
		rafRef.current = null;
		const ws = wsRef.current;
		if (ws && ws.readyState === WebSocket.OPEN) {
			try {
				ws.send("done");
			} catch {}
			ws.close();
		}
		wsRef.current = null;
		procRef.current?.disconnect();
		srcRef.current?.disconnect();
		srcRef.current = null;
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
		setTrackInfo("");
	};

	const loadDevices = async () => {
		if (!navigator.mediaDevices?.enumerateDevices) return;
		try {
			const probe = await navigator.mediaDevices.getUserMedia({ audio: true });
			for (const t of probe.getTracks()) t.stop();
		} catch {}
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

	const start = async () => {
		const md = navigator.mediaDevices;
		if (!md?.getUserMedia) {
			toast.error("Capture needs a secure context (https or localhost).");
			return;
		}
		try {
			let stream: MediaStream;
			if (source === "system") {
				if (!md.getDisplayMedia) {
					toast.error(
						"System-audio capture needs getDisplayMedia (Chrome/Edge).",
					);
					return;
				}
				// Chrome only exposes the "share audio" checkbox when video is also
				// requested. Pick a tab/window/screen, ENABLE "Share tab/system
				// audio", then we drop the video track and keep only audio. Raw
				// capture: no echo-cancel / noise-suppress / AGC on system audio.
				const disp = await md.getDisplayMedia({
					video: true,
					audio: {
						channelCount: 1,
						echoCancellation: false,
						noiseSuppression: false,
						autoGainControl: false,
					},
				});
				for (const v of disp.getVideoTracks()) v.stop();
				if (disp.getAudioTracks().length === 0) {
					for (const t of disp.getTracks()) t.stop();
					toast.error(
						'No audio shared — re-pick and tick "Share tab/system audio".',
					);
					return;
				}
				stream = disp;
			} else {
				// Known-working mic config. autoGainControl is load-bearing for
				// quiet USB mics.
				stream = await md.getUserMedia({
					audio: {
						channelCount: 1,
						echoCancellation: true,
						noiseSuppression: true,
						autoGainControl: true,
						...(deviceId ? { deviceId: { exact: deviceId } } : {}),
					},
				});
			}
			streamRef.current = stream;
			if (devices.every((d) => !d.label)) loadDevices();
			// A user closing the browser's "sharing" bar ends the track — stop
			// cleanly so the UI returns to idle.
			for (const t of stream.getAudioTracks()) {
				t.addEventListener("ended", () => stop());
			}
			const track = stream.getAudioTracks()[0];
			const s = track?.getSettings?.() ?? {};
			setTrackInfo(
				`${track?.label || "?"} · ${s.sampleRate ?? "?"}Hz · ${
					s.channelCount ?? "?"
				}ch · muted=${track?.muted}`,
			);

			// Open the streaming WS to the sidecar (via same-origin proxy).
			const ws = new WebSocket(streamUrl());
			ws.binaryType = "arraybuffer";
			wsRef.current = ws;
			ws.onmessage = (ev) => {
				let msg: {
					type?: string;
					text?: string;
					eou?: boolean;
					message?: string;
				};
				try {
					msg = JSON.parse(ev.data);
				} catch {
					return;
				}
				if (msg.type === "commit") {
					let c = committedRef.current;
					if (msg.text) {
						c += (c && !c.endsWith("\n") ? " " : "") + msg.text;
					}
					// End-of-utterance (trailing-silence pause) → line break so
					// separate utterances render on their own lines.
					if (msg.eou && c.trim()) c = `${c.replace(/\s+$/, "")}\n`;
					committedRef.current = c;
					setCommitted(c);
					setPartial("");
				} else if (msg.type === "partial") {
					setPartial(msg.text ?? "");
				} else if (msg.type === "error") {
					toast.error(msg.message || "stream error");
				}
			};
			ws.onerror = () => toast.error("Live stream connection failed");

			const ctx = new AudioContext();
			if (ctx.state === "suspended") await ctx.resume();
			ctxRef.current = ctx;
			const src = ctx.createMediaStreamSource(stream);
			srcRef.current = src;

			const analyser = ctx.createAnalyser();
			analyser.fftSize = 1024;
			analyserRef.current = analyser;
			src.connect(analyser);

			const proc = ctx.createScriptProcessor(4096, 1, 1);
			procRef.current = proc;
			const sink = ctx.createGain();
			sink.gain.value = 0;
			sinkRef.current = sink;
			proc.onaudioprocess = (ev) => {
				const pcm = downsampleToPcm16(
					ev.inputBuffer.getChannelData(0),
					ctx.sampleRate,
				);
				const sock = wsRef.current;
				if (sock && sock.readyState === WebSocket.OPEN) {
					sock.send(pcm.buffer as ArrayBuffer);
				}
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

	// Scrolling waveform driven off the analyser (visual feedback only).
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
	const hasText = committed || partial;

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
						streaming
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
					{trackInfo && (
						<div className="font-mono text-[10px] text-muted-foreground">
							track: {trackInfo}
						</div>
					)}
				</div>
			)}

			<div className="flex flex-wrap items-center gap-2">
				<div className="flex items-center gap-1">
					<Button
						variant={source === "mic" ? "default" : "outline"}
						size="sm"
						disabled={recording}
						onClick={() => setSource("mic")}
					>
						Mic
					</Button>
					<Button
						variant={source === "system" ? "default" : "outline"}
						size="sm"
						disabled={recording}
						onClick={() => setSource("system")}
						title="Capture desktop / tab / OBS audio (tick 'share audio' in the picker)"
					>
						System / OBS
					</Button>
				</div>
				<Select
					value={deviceId}
					onValueChange={setDeviceId}
					disabled={recording || source === "system"}
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
						{source === "system" ? "● Capture system audio" : "● Start mic"}
					</Button>
				) : (
					<Button variant="destructive" onClick={stop}>
						■ Stop
					</Button>
				)}
				<Button
					variant="outline"
					onClick={() => {
						const text = (committed + (partial ? ` ${partial}` : "")).trim();
						if (!text) return;
						navigator.clipboard.writeText(text);
						toast.success("Copied transcript");
					}}
					disabled={!hasText}
				>
					Copy
				</Button>
				<Button
					variant="outline"
					onClick={() => {
						const text = (committed + (partial ? ` ${partial}` : "")).trim();
						if (!text) return;
						const blob = new Blob([`${text}\n`], { type: "text/plain" });
						const a = document.createElement("a");
						a.href = URL.createObjectURL(blob);
						const ts = new Date()
							.toISOString()
							.replace(/[:.]/g, "-")
							.slice(0, 19);
						a.download = `live-transcript-${ts}.txt`;
						a.click();
						URL.revokeObjectURL(a.href);
					}}
					disabled={!hasText}
				>
					Export .txt
				</Button>
				<Button
					variant="outline"
					onClick={() => {
						committedRef.current = "";
						setCommitted("");
						setPartial("");
					}}
					disabled={!hasText}
				>
					Clear
				</Button>
			</div>

			<div className="min-h-80 overflow-auto rounded border bg-card p-3 font-mono text-xs leading-relaxed">
				{hasText ? (
					<p className="whitespace-pre-wrap">
						<span>{committed}</span>{" "}
						<span className="text-muted-foreground italic">{partial}</span>
					</p>
				) : (
					<span className="text-muted-foreground">
						Live transcript appears here. Words commit (solid) once stable;
						provisional words show dimmed.
					</span>
				)}
			</div>

			<p className="text-xs text-muted-foreground">
				{source === "system"
					? "System / OBS audio (desktop, tab, or window — tick 'share audio' in the picker) streams continuously to the whisper-live sidecar."
					: "Mic audio streams continuously to the whisper-live sidecar."}{" "}
				It runs LocalAgreement streaming: transcribes a growing buffer and
				commits only words confirmed across consecutive passes — low latency
				without the short-window hallucinations. For a headless / no-browser
				capture, use the standalone CLI in <code>live-tap/</code>.
			</p>
		</div>
	);
}
