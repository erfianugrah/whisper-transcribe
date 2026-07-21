import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Trash2 } from "lucide-react";
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
import {
	Table,
	TableBody,
	TableCell,
	TableHead,
	TableHeader,
	TableRow,
} from "@/components/ui/table";
import {
	addVoiceprint,
	deleteVoiceprint,
	getMedia,
	getVocabulary,
	getVoiceprints,
	putVocabulary,
} from "@/lib/api";

// Enrolled voice prints drive server-side speaker naming: a matched speaker is
// relabeled to the real name in EVERY transcript the server produces (this UI,
// the bot, plain curl). Enroll once from a clip where the person speaks; the
// server embeds the dominant voice and stores it on the persistent /data
// volume. Enrolling the same name again appends a second reference vector
// (improves matching).

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

export function VoicesTab() {
	const qc = useQueryClient();
	const prints = useQuery({
		queryKey: ["voiceprints"],
		queryFn: getVoiceprints,
	});
	const media = useQuery({ queryKey: ["media"], queryFn: () => getMedia() });

	const [name, setName] = useState("");
	const [file, setFile] = useState("");
	const [start, setStart] = useState("");
	const [end, setEnd] = useState("");

	const enroll = useMutation({
		mutationFn: () =>
			addVoiceprint({
				name: name.trim(),
				file_path: file,
				start: start ? Number(start) : undefined,
				end: end ? Number(end) : undefined,
			}),
		onSuccess: (r) => {
			toast.success(
				`Enrolled "${r.name}" (${r.count} print${r.count === 1 ? "" : "s"})`,
			);
			setName("");
			setStart("");
			setEnd("");
			qc.invalidateQueries({ queryKey: ["voiceprints"] });
		},
		onError: (e: Error) => toast.error(e.message),
	});

	const remove = useMutation({
		mutationFn: (n: string) => deleteVoiceprint(n),
		onSuccess: (_d, n) => {
			toast.success(`Removed "${n}"`);
			qc.invalidateQueries({ queryKey: ["voiceprints"] });
		},
		onError: (e: Error) => toast.error(e.message),
	});
	const canEnroll =
		name.trim().length > 0 && file.length > 0 && !enroll.isPending;
	const rows = prints.data?.voiceprints ?? [];

	// ── vocabulary (auto-hotwords) ──────────────────────────────────────────
	const vocab = useQuery({ queryKey: ["vocabulary"], queryFn: getVocabulary });
	const [vocabText, setVocabText] = useState<string | null>(null);
	const saveVocab = useMutation({
		mutationFn: () =>
			putVocabulary(
				(vocabText ?? "")
					.split("\n")
					.map((t) => t.trim())
					.filter((t) => t.length > 0 && !t.startsWith("#")),
			),
		onSuccess: (r) => {
			toast.success(`Vocabulary saved (${r.terms.length} terms)`);
			setVocabText(null);
			qc.invalidateQueries({ queryKey: ["vocabulary"] });
		},
		onError: (e: Error) => toast.error(e.message),
	});
	const vocabDisplay =
		vocabText ?? (vocab.data ? vocab.data.terms.join("\n") : "");

	return (
		<div className="flex flex-col gap-6 max-w-3xl">
			<p className="text-sm text-muted-foreground">
				Enrolled voices are matched against each diarized speaker by voice, so
				transcripts show real names instead of{" "}
				<code className="font-mono text-xs">SPEAKER_00</code>. Applies to every
				transcript the server produces - this UI, the bot, and the API.
			</p>

			{/* Enroll form */}
			<div className="flex flex-col gap-3 rounded-md border p-4">
				<h2 className="text-sm font-medium">Enroll a voice</h2>
				<div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
					<Field label="Name">
						<Input
							value={name}
							onChange={(e) => setName(e.target.value)}
							placeholder="e.g. Erfi"
						/>
					</Field>
					<Field label="Source file (a clip where they speak)">
						<Select value={file} onValueChange={setFile}>
							<SelectTrigger>
								<SelectValue placeholder="Select a server file..." />
							</SelectTrigger>
							<SelectContent>
								{(media.data?.files ?? []).map((f) => (
									<SelectItem key={f.path} value={f.path}>
										{f.name}
									</SelectItem>
								))}
							</SelectContent>
						</Select>
					</Field>
					<Field label="Start (sec, optional)">
						<Input
							type="number"
							min={0}
							value={start}
							onChange={(e) => setStart(e.target.value)}
							placeholder="0"
						/>
					</Field>
					<Field label="End (sec, optional)">
						<Input
							type="number"
							min={0}
							value={end}
							onChange={(e) => setEnd(e.target.value)}
							placeholder="whole file"
						/>
					</Field>
				</div>
				<p className="text-xs text-muted-foreground">
					Tip: point at a short clip where only this person talks. The server
					embeds the dominant voice. A time range narrows it further.
				</p>
				<div>
					<Button
						type="button"
						disabled={!canEnroll}
						onClick={() => enroll.mutate()}
					>
						{enroll.isPending ? "Enrolling..." : "Enroll"}
					</Button>
				</div>
			</div>

			{/* Enrolled list */}
			<div className="flex flex-col gap-2">
				<h2 className="text-sm font-medium">
					Enrolled voices{" "}
					<Badge variant="secondary" className="ml-1 font-mono">
						{rows.length}
					</Badge>
				</h2>
				{prints.isLoading ? (
					<p className="text-sm text-muted-foreground">Loading...</p>
				) : rows.length === 0 ? (
					<p className="text-sm text-muted-foreground">
						No voices enrolled yet. Enroll one above and future transcripts will
						name that speaker automatically.
					</p>
				) : (
					<Table>
						<TableHeader>
							<TableRow>
								<TableHead>Name</TableHead>
								<TableHead className="text-right">Reference prints</TableHead>
								<TableHead className="w-10" />
							</TableRow>
						</TableHeader>
						<TableBody>
							{rows.map((v) => (
								<TableRow key={v.name}>
									<TableCell className="font-medium">{v.name}</TableCell>
									<TableCell className="text-right font-mono text-xs">
										{v.count}
									</TableCell>
									<TableCell className="text-right">
										<Button
											type="button"
											variant="ghost"
											size="icon"
											aria-label={`Remove ${v.name}`}
											disabled={remove.isPending}
											onClick={() => remove.mutate(v.name)}
										>
											<Trash2 className="size-4 text-muted-foreground" />
										</Button>
									</TableCell>
								</TableRow>
							))}
						</TableBody>
					</Table>
				)}
			</div>

			{/* Vocabulary (auto-hotwords) */}
			<div className="flex flex-col gap-3 rounded-md border p-4">
				<h2 className="text-sm font-medium">
					Vocabulary{" "}
					<Badge variant="secondary" className="ml-1 font-mono">
						{vocab.data?.terms.length ?? 0}
					</Badge>
				</h2>
				<p className="text-xs text-muted-foreground">
					One term per line. These are injected as hotwords into every
					transcription job automatically (alongside the enrolled voice names
					above) - use it for company, product, and account names the model
					would otherwise mis-hear.
					{vocab.data && !vocab.data.auto_hotwords
						? " Auto-injection is currently DISABLED server-side (AUTO_HOTWORDS=0)."
						: ""}
				</p>
				<Textarea
					rows={8}
					className="font-mono text-xs"
					value={vocabDisplay}
					onChange={(e) => setVocabText(e.target.value)}
					placeholder={
						vocab.isLoading ? "Loading..." : "Supabase\nPostgREST\nAcme Corp"
					}
				/>
				<div>
					<Button
						type="button"
						disabled={vocabText === null || saveVocab.isPending}
						onClick={() => saveVocab.mutate()}
					>
						{saveVocab.isPending ? "Saving..." : "Save vocabulary"}
					</Button>
				</div>
			</div>
		</div>
	);
}
