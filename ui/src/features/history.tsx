import { useQuery } from "@tanstack/react-query";
import {
	Table,
	TableBody,
	TableCell,
	TableHead,
	TableHeader,
	TableRow,
} from "@/components/ui/table";
import { getHistory } from "@/lib/api";

export function HistoryTab() {
	const { data, isLoading } = useQuery({
		queryKey: ["history"],
		queryFn: getHistory,
	});

	if (isLoading)
		return <p className="text-sm text-muted-foreground">Loading…</p>;
	if (!data || data.length === 0)
		return (
			<p className="text-sm text-muted-foreground">
				No transcription history yet.
			</p>
		);

	return (
		<Table>
			<TableHeader>
				<TableRow>
					<TableHead>When</TableHead>
					<TableHead>File</TableHead>
					<TableHead>Lang</TableHead>
					<TableHead className="text-right">Dur</TableHead>
					<TableHead className="text-right">Segs</TableHead>
					<TableHead className="text-right">Spk</TableHead>
					<TableHead className="text-right">Speed</TableHead>
				</TableRow>
			</TableHeader>
			<TableBody>
				{data.map((h, i) => (
					<TableRow key={`${h.timestamp ?? ""}-${h.filename ?? ""}-${i}`}>
						<TableCell className="font-mono text-xs whitespace-nowrap">
							{h.timestamp}
						</TableCell>
						<TableCell
							className="max-w-[18rem] truncate text-xs"
							title={h.filename ?? undefined}
						>
							{h.filename}
						</TableCell>
						<TableCell className="font-mono text-xs">{h.language}</TableCell>
						<TableCell className="text-right font-mono text-xs">
							{h.duration_str}
						</TableCell>
						<TableCell className="text-right font-mono text-xs">
							{h.segments}
						</TableCell>
						<TableCell className="text-right font-mono text-xs">
							{h.speakers || "—"}
						</TableCell>
						<TableCell className="text-right font-mono text-xs">
							{h.speed}
						</TableCell>
					</TableRow>
				))}
			</TableBody>
		</Table>
	);
}
