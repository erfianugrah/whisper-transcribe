import { MonitorIcon, MoonIcon, SunIcon } from "lucide-react";
import { useTheme } from "next-themes";
import { cn } from "@/lib/utils";

const OPTIONS = [
	{ value: "light", icon: SunIcon, label: "Light" },
	{ value: "system", icon: MonitorIcon, label: "System" },
	{ value: "dark", icon: MoonIcon, label: "Dark" },
] as const;

export function ThemeToggle() {
	const { theme, setTheme } = useTheme();
	return (
		<div className="inline-flex items-center rounded border p-0.5">
			{OPTIONS.map(({ value, icon: Icon, label }) => (
				<button
					key={value}
					type="button"
					title={label}
					aria-label={label}
					onClick={() => setTheme(value)}
					className={cn(
						"inline-flex size-6 items-center justify-center rounded-sm text-muted-foreground transition-colors hover:text-foreground",
						theme === value && "bg-secondary text-foreground",
					)}
				>
					<Icon className="size-3.5" />
				</button>
			))}
		</div>
	);
}
