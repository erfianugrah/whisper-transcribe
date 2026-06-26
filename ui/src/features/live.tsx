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
const SILENCE_RMS = 0.004; // below this = effectively no audio
const SILENCE_AFTER_MS = 3000; // warn after this long with no audible input
const MAX_RECONNECTS = 6;

// Common whisper language codes for the per-session pin selector.
const LANGUAGES: Array<[string, string]> = [
	["auto", "Auto-detect"],
	["en", "English"],
	["es", "Spanish"],
	["fr", "French"],
	["de", "German"],
	["it", "Italian"],
	["pt", "Portuguese"],
	["nl", "Dutch"],
	["ja", "Japanese"],
	["ko", "Korean"],
	["zh", "Chinese"],
	["ru", "Russian"],
	["ar", "Arabic"],
	["hi", "Hindi"],
];

type Source = "mic" | "system" | "both";

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

// Firefox/Gecko (incl. Zen) expose getDisplayMedia but cannot capture its
// audio — system/tab audio yields zero audio tracks there. Gate the System /
// Both modes on Chromium so the buttons explain rather than silently fail.
function systemAudioSupported(): boolean {
	if (typeof navigator === "undefined") return false;
	if (!navigator.mediaDevices?.getDisplayMedia) return false;
	return !/firefox/i.test(navigator.userAgent || "");
}

export function LiveTab() {
	const [recording, setRecording] = useState(false);
	const [committed, setCommitted] = useState("");
	const [partial, setPartial] = useState("");
	const [trackInfo, setTrackInfo] = useState("");
	const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
	const [deviceId, setDeviceId] = useState<string>("");
	// "mic" → getUserMedia; "system" → getDisplayMedia (desktop/tab/OBS);
	// "both" → mic + system mixed (a meeting: you + the other participants).
	const [source, setSource] = useState<Source>("mic");
	const [language, setLanguage] = useState("auto");
	const [translate, setTranslate] = useState(false);
	const [micGain, setMicGain] = useState(1.4); // boost a usually-quiet mic
	const [sysGain, setSysGain] = useState(1.0);
	const [level, setLevel] = useState(0); // input RMS 0..1 for the meter
	const [silent, setSilent] = useState(false);
	const sysOk = systemAudioSupported();

	const health = useQuery({
		queryKey: ["live-health"],
		queryFn: getLiveHealth,
		refetchInterval: 5000,
	});

	const ctxRef = useRef<AudioContext | null>(null);
	const streamsRef = useRef<MediaStream[]>([]);
	const procRef = useRef<ScriptProcessorNode | null>(null);
	const sinkRef = useRef<GainNode | null>(null);
	const mixerRef = useRef<GainNode | null>(null);
	const micGainRef = useRef<GainNode | null>(null);
	const sysGainRef = useRef<GainNode | null>(null);
	// Firefox GCs an unreferenced MediaStreamAudioSourceNode even while it's
	// connected, silencing the graph. Hold refs to keep them alive.
	const srcNodesRef = useRef<MediaStreamAudioSourceNode[]>([]);
	const analyserRef = useRef<AnalyserNode | null>(null);
	const canvasRef = useRef<HTMLCanvasElement | null>(null);
	const rafRef = useRef<number | null>(null);
	const columnsRef = useRef<Array<[number, number]>>([]);
	const wsRef = useRef<WebSocket | null>(null);
	const committedRef = useRef("");
	// Reconnect bookkeeping (parity with the live-tap CLI + voice bot).
	const wantStreamRef = useRef(false);
	const reconnectsRef = useRef(0);
	const reconnectTimerRef = useRef<number | null>(null);
	// Level meter / silence detection.
	const levelRef = useRef(0);
	const lastLoudRef = useRef(0);

	const stop = () => {
		wantStreamRef.current = false;
		if (reconnectTimerRef.current != null)
			clearTimeout(reconnectTimerRef.current);
		reconnectTimerRef.current = null;
		reconnectsRef.current = 0;
		if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
		rafRef.current = null;
		const ws = wsRef.current;
		if (ws && ws.readyState === WebSocket.OPEN) {
			try {
				ws.send("done");
			} catch {}
		}
		ws?.close();
		wsRef.current = null;
		procRef.current?.disconnect();
		for (const n of srcNodesRef.current) n.disconnect();
		srcNodesRef.current = [];
		micGainRef.current?.disconnect();
		sysGainRef.current?.disconnect();
		mixerRef.current?.disconnect();
		sinkRef.current?.disconnect();
		analyserRef.current?.disconnect();
		procRef.current = null;
		micGainRef.current = null;
		sysGainRef.current = null;
		mixerRef.current = null;
		sinkRef.current = null;
		analyserRef.current = null;
		for (const s of streamsRef.current) for (const t of s.getTracks()) t.stop();
		streamsRef.current = [];
		ctxRef.current?.close().catch(() => {});
		ctxRef.current = null;
		levelRef.current = 0;
		setLevel(0);
		setSilent(false);
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

	async function captureMic(): Promise<MediaStream> {
		// autoGainControl is load-bearing for quiet USB mics.
		return navigator.mediaDevices.getUserMedia({
			audio: {
				channelCount: 1,
				echoCancellation: true,
				noiseSuppression: true,
				autoGainControl: true,
				...(deviceId ? { deviceId: { exact: deviceId } } : {}),
			},
		});
	}

	async function captureSystem(): Promise<MediaStream> {
		const md = navigator.mediaDevices;
		// Chrome only exposes the "share audio" checkbox when video is also
		// requested. Pick a tab/window/screen, ENABLE "Share tab/system audio";
		// we drop the video track and keep only raw audio (no AGC/NS/echo on
		// system audio).
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
			throw new Error(
				'No audio shared — re-pick and tick "Share tab/system audio".',
			);
		}
		return disp;
	}

	function openWs() {
		const ws = new WebSocket(streamUrl());
		ws.binaryType = "arraybuffer";
		wsRef.current = ws;
		ws.onopen = () => {
			reconnectsRef.current = 0; // healthy connection
			const cfg: Record<string, unknown> = {};
			if (language && language !== "auto") cfg.language = language;
			if (translate) cfg.translate = true;
			try {
				ws.send(JSON.stringify(cfg)); // per-session handshake (first frame)
			} catch {}
		};
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
				if (msg.text) c += (c && !c.endsWith("\n") ? " " : "") + msg.text;
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
		ws.onerror = () => {}; // let onclose drive reconnect
		ws.onclose = () => {
			if (!wantStreamRef.current) return; // intentional stop
			reconnectsRef.current += 1;
			const n = reconnectsRef.current;
			if (n > MAX_RECONNECTS) {
				toast.error("Live stream lost — gave up reconnecting");
				stop();
				return;
			}
			const backoff = Math.min(8000, 500 * 2 ** n);
			toast.message(
				`Live stream dropped — reconnecting… (${n}/${MAX_RECONNECTS})`,
			);
			reconnectTimerRef.current = window.setTimeout(openWs, backoff);
		};
	}

	const start = async () => {
		const md = navigator.mediaDevices;
		if (!md?.getUserMedia) {
			toast.error("Capture needs a secure context (https or localhost).");
			return;
		}
		if ((source === "system" || source === "both") && !sysOk) {
			toast.error(
				"System audio needs a Chromium browser (Chrome/Edge). Firefox/Zen can't capture it.",
			);
			return;
		}
		try {
			const streams: MediaStream[] = [];
			let micStream: MediaStream | null = null;
			let sysStream: MediaStream | null = null;
			if (source === "mic" || source === "both") {
				micStream = await captureMic();
				streams.push(micStream);
			}
			if (source === "system" || source === "both") {
				sysStream = await captureSystem();
				streams.push(sysStream);
			}
			streamsRef.current = streams;
			if (devices.every((d) => !d.label)) loadDevices();
			// Closing the browser "sharing" bar (or unplugging the mic) ends a
			// track — stop cleanly so the UI returns to idle.
			for (const s of streams)
				for (const t of s.getAudioTracks())
					t.addEventListener("ended", () => stop());

			const labels: string[] = [];
			if (micStream)
				labels.push(`mic: ${micStream.getAudioTracks()[0]?.label || "?"}`);
			if (sysStream)
				labels.push(`sys: ${sysStream.getAudioTracks()[0]?.label || "?"}`);
			setTrackInfo(labels.join("  ·  "));

			const ctx = new AudioContext();
			if (ctx.state === "suspended") await ctx.resume();
			ctxRef.current = ctx;

			// Mixer: each source → its own gain → mixer → analyser + downsampler.
			const mixer = ctx.createGain();
			mixerRef.current = mixer;
			srcNodesRef.current = [];
			if (micStream) {
				const n = ctx.createMediaStreamSource(micStream);
				const g = ctx.createGain();
				g.gain.value = source === "both" ? micGain : 1;
				micGainRef.current = g;
				n.connect(g);
				g.connect(mixer);
				srcNodesRef.current.push(n);
			}
			if (sysStream) {
				const n = ctx.createMediaStreamSource(sysStream);
				const g = ctx.createGain();
				g.gain.value = source === "both" ? sysGain : 1;
				sysGainRef.current = g;
				n.connect(g);
				g.connect(mixer);
				srcNodesRef.current.push(n);
			}

			const analyser = ctx.createAnalyser();
			analyser.fftSize = 1024;
			analyserRef.current = analyser;
			mixer.connect(analyser);

			const proc = ctx.createScriptProcessor(4096, 1, 1);
			procRef.current = proc;
			const sink = ctx.createGain();
			sink.gain.value = 0; // mute local playback; we only stream
			sinkRef.current = sink;
			proc.onaudioprocess = (ev) => {
				const pcm = downsampleToPcm16(
					ev.inputBuffer.getChannelData(0),
					ctx.sampleRate,
				);
				const sock = wsRef.current;
				if (sock && sock.readyState === WebSocket.OPEN)
					sock.send(pcm.buffer as ArrayBuffer);
			};
			mixer.connect(proc);
			proc.connect(sink);
			sink.connect(ctx.destination);

			columnsRef.current = [];
			lastLoudRef.current = performance.now();
			wantStreamRef.current = true;
			reconnectsRef.current = 0;
			openWs();
			setRecording(true);
		} catch (e) {
			toast.error(`Capture failed: ${(e as Error).message}`);
			stop();
		}
	};

	// Live per-source gain (mic+system mix) without a restart.
	useEffect(() => {
		if (micGainRef.current && source === "both")
			micGainRef.current.gain.value = micGain;
	}, [micGain, source]);
	useEffect(() => {
		if (sysGainRef.current && source === "both")
			sysGainRef.current.gain.value = sysGain;
	}, [sysGain, source]);

	// Scrolling waveform + level meter, driven off the analyser.
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
			let sumSq = 0;
			for (let i = 0; i < buf.length; i++) {
				const v = buf[i];
				if (v < lo) lo = v;
				if (v > hi) hi = v;
				sumSq += v * v;
			}
			levelRef.current = Math.sqrt(sumSq / buf.length);
			if (levelRef.current >= SILENCE_RMS)
				lastLoudRef.current = performance.now();
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

	// Throttled level/silence state update (avoids re-rendering every frame).
	useEffect(() => {
		if (!recording) return;
		const id = window.setInterval(() => {
			setLevel(levelRef.current);
			setSilent(performance.now() - lastLoudRef.current > SILENCE_AFTER_MS);
		}, 250);
		return () => clearInterval(id);
	}, [recording]);

	// biome-ignore lint/correctness/useExhaustiveDependencies: tear down on unmount only.
	useEffect(() => () => stop(), []);

	const unavailable = health.data?.status === "unavailable";
	const hasText = committed || partial;
	const levelPct = Math.min(100, Math.round(level * 320));

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
					<div className="flex items-center gap-2">
						<span className="font-mono text-[10px] text-muted-foreground">
							level
						</span>
						<div className="h-1.5 flex-1 overflow-hidden rounded bg-muted">
							<div
								className={`h-full transition-[width] duration-150 ${
									silent ? "bg-destructive/60" : "bg-primary"
								}`}
								style={{ width: `${levelPct}%` }}
							/>
						</div>
						{silent && (
							<Badge variant="destructive" className="text-[10px]">
								no audio detected
							</Badge>
						)}
					</div>
					{silent && (
						<div className="font-mono text-[10px] text-destructive">
							Nothing audible is reaching whisper-live. Check the source is
							actually playing
							{source !== "mic"
								? ', and that you ticked "Share tab/system audio" — capturing a Window never carries audio.'
								: " and the right mic is selected."}
						</div>
					)}
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
						disabled={recording || !sysOk}
						onClick={() => setSource("system")}
						title={
							sysOk
								? "Capture desktop / tab / OBS audio (tick 'share audio' in the picker)"
								: "System audio needs a Chromium browser (Chrome/Edge) — Firefox/Zen can't capture it"
						}
					>
						System / OBS
					</Button>
					<Button
						variant={source === "both" ? "default" : "outline"}
						size="sm"
						disabled={recording || !sysOk}
						onClick={() => setSource("both")}
						title={
							sysOk
								? "Mix your mic with system/tab audio — a meeting (you + everyone else)"
								: "System audio needs a Chromium browser (Chrome/Edge) — Firefox/Zen can't capture it"
						}
					>
						Mic + System
					</Button>
				</div>
				<Select
					value={deviceId}
					onValueChange={setDeviceId}
					disabled={recording || source === "system"}
				>
					<SelectTrigger className="w-60">
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
				<Select
					value={language}
					onValueChange={setLanguage}
					disabled={recording}
				>
					<SelectTrigger className="w-36" title="Pin the spoken language">
						<SelectValue />
					</SelectTrigger>
					<SelectContent>
						{LANGUAGES.map(([code, label]) => (
							<SelectItem key={code} value={code}>
								{label}
							</SelectItem>
						))}
					</SelectContent>
				</Select>
				<Button
					variant={translate ? "default" : "outline"}
					size="sm"
					disabled={recording}
					onClick={() => setTranslate((v) => !v)}
					title="Translate the transcript to English"
				>
					Translate→EN
				</Button>
				{!recording ? (
					<Button onClick={start} disabled={unavailable}>
						{source === "mic" ? "● Start mic" : "● Capture audio"}
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

			{source === "both" && (
				<div className="flex flex-wrap items-center gap-4 font-mono text-[10px] text-muted-foreground">
					<label className="flex items-center gap-2">
						mic gain {micGain.toFixed(1)}
						<input
							type="range"
							min={0}
							max={3}
							step={0.1}
							value={micGain}
							onChange={(e) => setMicGain(Number(e.target.value))}
							className="w-28"
						/>
					</label>
					<label className="flex items-center gap-2">
						system gain {sysGain.toFixed(1)}
						<input
							type="range"
							min={0}
							max={3}
							step={0.1}
							value={sysGain}
							onChange={(e) => setSysGain(Number(e.target.value))}
							className="w-28"
						/>
					</label>
				</div>
			)}

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
				{source === "both"
					? "Mic + system/OBS audio are mixed (your voice + everyone else) and streamed to whisper-live."
					: source === "system"
						? "System / OBS audio (desktop, tab, or window — tick 'share audio' in the picker) streams to whisper-live."
						: "Mic audio streams to whisper-live."}{" "}
				LocalAgreement streaming commits only words confirmed across consecutive
				passes — low latency without short-window hallucinations. The stream
				auto-reconnects if whisper-live blips. For a headless / no-browser (or
				Firefox) capture incl. native system-audio loopback, use the standalone
				CLI in <code>live-tap/</code>.
			</p>
		</div>
	);
}
