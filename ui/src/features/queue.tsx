import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
	Table,
	TableBody,
	TableCell,
	TableHead,
	TableHeader,
	TableRow,
} from "@/components/ui/table";
import { cancelJob, getQueue } from "@/lib/api";

function fmtTime(v: unknown): string {
	if (v == null) return "";
	const n = typeof v === "string" ? Number(v) : (v as number);
	if (!Number.isFinite(n)) return String(v);
	const d = new Date(n < 1e12 ? n * 1000 : n);
	return d.toLocaleTimeString();
}

export function QueueTab() {
	const qc = useQueryClient();
	const { data, isError } = useQuery({
		queryKey: ["queue"],
		queryFn: getQueue,
		refetchInterval: 2000,
	});

	const cancel = useMutation({
		mutationFn: cancelJob,
		onSuccess: () => {
			toast.success("Cancelled");
			qc.invalidateQueries({ queryKey: ["queue"] });
		},
		onError: (e: Error) => toast.error(e.message),
	});

	if (isError || (data && !data.available)) {
		return (
			<p className="text-sm text-muted-foreground">
				Queue backend unavailable (Valkey down).
			</p>
		);
	}

	const active = data?.active ?? [];
	const recent = data?.recent ?? [];

	return (
		<div className="flex flex-col gap-6">
			<div className="flex items-center gap-4 font-mono text-xs text-muted-foreground">
				<span>
					depth <span className="text-foreground">{data?.depth ?? 0}</span>
				</span>
				<span>
					active <span className="text-foreground">{active.length}</span>
				</span>
			</div>

			<section>
				<h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
					Active
				</h2>
				<Table>
					<TableHeader>
						<TableRow>
							<TableHead>Job</TableHead>
							<TableHead>Status</TableHead>
							<TableHead>Consumer</TableHead>
							<TableHead>Started</TableHead>
						</TableRow>
					</TableHeader>
					<TableBody>
						{active.length === 0 && (
							<TableRow>
								<TableCell colSpan={4} className="text-muted-foreground">
									idle
								</TableCell>
							</TableRow>
						)}
						{active.map((j) => (
							<TableRow key={j.id}>
								<TableCell className="font-mono text-xs">
									{j.id?.slice(0, 8)}
								</TableCell>
								<TableCell>
									<Badge>{j.status}</Badge>
								</TableCell>
								<TableCell className="text-xs">{j.consumer}</TableCell>
								<TableCell className="font-mono text-xs">
									{fmtTime(j.started_at)}
								</TableCell>
							</TableRow>
						))}
					</TableBody>
				</Table>
			</section>

			<section>
				<h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
					Recent
				</h2>
				<Table>
					<TableHeader>
						<TableRow>
							<TableHead>Job</TableHead>
							<TableHead>Status</TableHead>
							<TableHead>Consumer</TableHead>
							<TableHead>Finished</TableHead>
							<TableHead className="text-right">Action</TableHead>
						</TableRow>
					</TableHeader>
					<TableBody>
						{recent.length === 0 && (
							<TableRow>
								<TableCell colSpan={5} className="text-muted-foreground">
									no recent jobs
								</TableCell>
							</TableRow>
						)}
						{recent.map((j) => (
							<TableRow key={j.id}>
								<TableCell className="font-mono text-xs">
									{j.id?.slice(0, 8)}
								</TableCell>
								<TableCell>
									<Badge
										variant={
											j.status === "failed" ? "destructive" : "secondary"
										}
									>
										{j.status}
									</Badge>
								</TableCell>
								<TableCell className="text-xs">{j.consumer}</TableCell>
								<TableCell className="font-mono text-xs">
									{fmtTime(j.completed_at)}
								</TableCell>
								<TableCell className="text-right">
									{j.status === "queued" && j.id && (
										<Button
											size="sm"
											variant="outline"
											onClick={() => cancel.mutate(j.id!)}
										>
											Cancel
										</Button>
									)}
								</TableCell>
							</TableRow>
						))}
					</TableBody>
				</Table>
			</section>
		</div>
	);
}
