import type { PointerEvent as ReactPointerEvent, ReactNode } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  BrowserRouter,
  Link,
  Navigate,
  NavLink,
  Route,
  Routes,
  useLocation
} from "react-router-dom";
import type { LucideIcon } from "lucide-react";
import {
  Activity,
  AlertTriangle,
  Archive,
  BadgeCheck,
  Boxes,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  ClipboardCheck,
  Clock,
  Cpu,
  Download,
  Eye,
  FileText,
  FolderOpen,
  Gauge,
  HardDrive,
  Layers3,
  Library,
  ListChecks,
  Maximize2,
  Minus,
  Monitor,
  PackageCheck,
  Play,
  RefreshCw,
  RotateCcw,
  Search,
  Server,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Square,
  Terminal,
  Video,
  X,
  Zap
} from "lucide-react";
import {
  ApiEnvelope,
  ComplianceIndexPage,
  ComplianceRow,
  ComplianceViolationRow,
  ControlJob,
  ControlJobPage,
  DashboardSummary,
  getJson,
  LogTail,
  ModuleLibraryPage,
  ModuleLibraryRow,
  ModuleReadiness,
  ModuleReadinessRow,
  query,
  QueueDetail,
  QueueLaunchConfig,
  QueuePipelineMode,
  QueueRunRow,
  QueueRunMode,
  QueueVariantMode,
  QueueVodList,
  ScoreDetail,
  ScoreIndexPage,
  ScoreRow,
  sendJson,
  SettingsReadEntry,
  SettingsReadSnapshot,
  SystemStats,
  VariationOption,
  VariationPageData,
  VariationProfile,
  VariationVariant
} from "./api";
import { usePolling } from "./usePolling";

type BadgeKind = "good" | "bad" | "warn" | "info" | "neutral";
type ActionMessage = { kind: BadgeKind; text: string };
type SortDirection = "asc" | "desc";
type HealthPayload = { status: string; mode: string };
type WindowControlAction = "minimize" | "toggle-maximize" | "close";

declare global {
  interface Window {
    clipperDesktop?: {
      getStatus?: () => Promise<unknown>;
      windowControl?: (action: WindowControlAction) => Promise<{ maximized: boolean }>;
    };
  }
}

type NavItem = {
  label: string;
  path: string;
  icon: LucideIcon;
  detail: string;
};

const mainNav: NavItem[] = [
  { label: "Operations", path: "/operations", icon: Gauge, detail: "Queue health and production flow" },
  { label: "Queue", path: "/queue", icon: ListChecks, detail: "Run controls and video status" },
  { label: "Clip Review", path: "/clips", icon: Video, detail: "Scores, flags, previews, and variants" },
  { label: "Compliance", path: "/compliance", icon: ShieldCheck, detail: "Policy status and violations" },
  { label: "Variations", path: "/variations", icon: SlidersHorizontal, detail: "Global variant profiles and previews" },
  { label: "Module Library", path: "/modules", icon: Library, detail: "Reusable hook, main, and CTA inventory" },
  { label: "Exports", path: "/exports", icon: PackageCheck, detail: "Batch packaging jobs" },
  { label: "Jobs", path: "/jobs", icon: Activity, detail: "Control job history and results" }
];

const secondaryNav: NavItem[] = [
  { label: "Logs", path: "/logs", icon: Terminal, detail: "Pipeline log tail" },
  { label: "System", path: "/system", icon: Cpu, detail: "API and machine resources" },
  { label: "Settings", path: "/settings", icon: Settings, detail: "Safe operator overrides" }
];

const allNav = [...mainNav, ...secondaryNav];

function statusClass(value?: string | null): BadgeKind {
  const normalized = String(value ?? "").toLowerCase();
  if (["completed", "strong", "ready", "passed", "ok", "healthy", "approved"].some((item) => normalized.includes(item))) {
    return "good";
  }
  if (["failed", "blocked", "critical", "stalled", "rejected", "interrupted", "outside"].some((item) => normalized.includes(item))) {
    return "bad";
  }
  if (["review", "attention", "waiting", "partial", "paused", "queued", "running", "processing", "stopped"].some((item) => normalized.includes(item))) {
    return "warn";
  }
  if (!normalized || normalized === "none" || normalized === "-") {
    return "neutral";
  }
  return "info";
}

function healthText(summary?: DashboardSummary): string {
  const health = summary?.queue_health ?? {};
  const label = health["status_label"];
  if (typeof label === "string" && label) {
    return label;
  }
  return summary?.queue_status || "Unknown";
}

function healthSummary(summary?: DashboardSummary): string {
  const text = summary?.queue_health?.["summary"];
  return typeof text === "string" && text ? text : "No queue summary yet.";
}

function numberText(value: number | undefined | null, digits = 0): string {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value ?? 0);
}

function scoreText(value?: number | null): string {
  return value === undefined || value === null ? "-" : value.toFixed(1);
}

function parentDir(path: string): string {
  const index = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return index > 0 ? path.slice(0, index) : path;
}

function compactJson(value?: Record<string, unknown> | null): string {
  if (!value || Object.keys(value).length === 0) {
    return "-";
  }
  return JSON.stringify(value, null, 2);
}

function operationLabel(value: string): string {
  return value.replace(/_/g, " ");
}

function uniqueOptions(values: Array<string | undefined | null>): string[] {
  return Array.from(new Set(values.map((value) => String(value ?? "").trim()).filter(Boolean))).sort((a, b) => a.localeCompare(b));
}

const runModeOptions: Array<{ value: QueueRunMode; label: string }> = [
  { value: "folder_repeat", label: "Folder Repeat" },
  { value: "folder_once", label: "Folder Once" },
  { value: "single_video", label: "Single Video" }
];

const pipelineModeOptions: Array<{ value: QueuePipelineMode; label: string }> = [
  { value: "full", label: "Full Pipeline" },
  { value: "clips_only", label: "Clips Only" },
  { value: "modules_only", label: "Modules Only" },
  { value: "raw_cuts_only", label: "Raw Cuts Only" }
];

const variantModeOptions: Array<{ value: QueueVariantMode; label: string }> = [
  { value: "all", label: "All Variants" },
  { value: "original", label: "Original Only" },
  { value: "custom", label: "Custom Count" }
];

type OperationStageKey = "transcribe" | "llm" | "yolo" | "ffmpeg";
type OperationStageState = "done" | "running" | "waiting";

const operationStages: Array<{ key: OperationStageKey; label: string; icon: LucideIcon }> = [
  { key: "transcribe", label: "Transcription", icon: FileText },
  { key: "llm", label: "Sales Moment Detection", icon: ListChecks },
  { key: "yolo", label: "Product/Face Scan", icon: Boxes },
  { key: "ffmpeg", label: "Clip Rendering", icon: Clock }
];

function launchSummary(config?: Partial<QueueLaunchConfig>, fallback = "Folder Repeat • Full Pipeline • All Variants • Unlimited"): string {
  if (!config?.run_mode || !config.pipeline_mode) {
    return fallback;
  }
  const run = runModeOptions.find((item) => item.value === config.run_mode)?.label ?? config.run_mode;
  const pipeline = pipelineModeOptions.find((item) => item.value === config.pipeline_mode)?.label ?? config.pipeline_mode;
  const variantMode = config.pipeline_mode === "raw_cuts_only" ? "original" : (config.variant_mode ?? "all");
  const variants = variantMode === "custom"
    ? `${config.variant_count ?? 1} Variants`
    : variantModeOptions.find((item) => item.value === variantMode)?.label ?? variantMode;
  const maxClips = config.max_clips == null ? "Unlimited" : `${config.max_clips} clip${config.max_clips === 1 ? "" : "s"}`;
  return [run, pipeline, variants, maxClips].filter(Boolean).join(" • ");
}

function isQueueActive(queue?: QueueDetail): boolean {
  const queueStatus = String(queue?.queue_status ?? "").toLowerCase();
  const controlStatus = String(queue?.control_status ?? "").toLowerCase();
  return ["running", "restart_pending", "start_requested", "continue_requested"].includes(controlStatus)
    || ["running"].includes(queueStatus);
}

function isRunActive(row?: QueueRunRow | null): boolean {
  const status = String(row?.status ?? "").toLowerCase();
  return ["running", "processing", "active", "in_progress", "in progress"].some((item) => status.includes(item));
}

function isTerminalRun(row?: QueueRunRow | null): boolean {
  const status = String(row?.status ?? "").toLowerCase();
  return ["completed", "failed", "stopped", "interrupted", "cancelled", "canceled", "skipped"].some((item) => status.includes(item));
}

function runTime(row: QueueRunRow): number {
  const value = row.completed_at || row.started_at;
  const parsed = value ? Date.parse(value) : Number.NaN;
  return Number.isNaN(parsed) ? 0 : parsed;
}

function newestRun(rows: QueueRunRow[]): QueueRunRow | undefined {
  return [...rows].sort((a, b) => runTime(b) - runTime(a))[0];
}

function pickCurrentRun(rows: QueueRunRow[], queueStatus?: string | null): QueueRunRow | undefined {
  const active = rows.filter(isRunActive);
  if (active.length > 0) {
    return newestRun(active);
  }

  const queueIsActive = ["running", "restart_pending", "start_requested", "continue_requested"].includes(
    String(queueStatus ?? "").toLowerCase()
  );
  if (queueIsActive) {
    return newestRun(rows.filter((row) => !isTerminalRun(row) && row.progress > 0 && row.progress < 100));
  }

  return newestRun(rows.filter((row) => String(row.status).toLowerCase().includes("completed"))) ?? newestRun(rows);
}

function runStatusKind(value?: string | null): BadgeKind {
  const normalized = String(value ?? "").toLowerCase();
  if (["running", "processing", "active", "in_progress", "in progress"].some((item) => normalized.includes(item))) {
    return "good";
  }
  return statusClass(value);
}

function stageKeyForRun(row?: QueueRunRow | null): OperationStageKey | undefined {
  const raw = `${row?.current_stage ?? ""} ${row?.current_step ?? ""}`.toLowerCase();
  if (!raw.trim()) {
    return undefined;
  }
  if (["transcribe", "transcription", "whisper"].some((item) => raw.includes(item))) {
    return "transcribe";
  }
  if (["llm", "sales", "moment", "detect"].some((item) => raw.includes(item))) {
    return "llm";
  }
  if (["yolo", "product", "face", "scan"].some((item) => raw.includes(item))) {
    return "yolo";
  }
  if (["ffmpeg", "render", "clip"].some((item) => raw.includes(item))) {
    return "ffmpeg";
  }
  return undefined;
}

function operationStageState(
  stage: OperationStageKey,
  activeStage: OperationStageKey | undefined,
  row: QueueRunRow | undefined,
  summary?: DashboardSummary
): OperationStageState {
  const running = summary?.stage_running?.[stage] ?? 0;
  if (running > 0 || stage === activeStage) {
    return "running";
  }
  const activeIndex = operationStages.findIndex((item) => item.key === activeStage);
  const stageIndex = operationStages.findIndex((item) => item.key === stage);
  if (row && activeIndex >= 0 && stageIndex >= 0 && stageIndex < activeIndex) {
    return "done";
  }
  if (row && row.progress >= 100) {
    return "done";
  }
  return "waiting";
}

function operationStageProgress(
  state: OperationStageState,
  stage: OperationStageKey,
  activeStage: OperationStageKey | undefined,
  row: QueueRunRow | undefined
): number {
  if (state === "done") {
    return 100;
  }
  if (state === "running") {
    if (stage === activeStage && row) {
      return Math.max(8, Math.min(100, row.progress));
    }
    return 64;
  }
  return 0;
}

function displayTime(value?: string | null): string {
  if (!value) {
    return "-";
  }
  const parsed = new Date(value);
  if (!Number.isNaN(parsed.getTime())) {
    return new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(parsed);
  }
  return value;
}

function usePageInfo(): NavItem {
  const location = useLocation();
  return (
    allNav.find((item) => location.pathname === item.path || location.pathname.startsWith(`${item.path}/`)) ??
    mainNav[0]
  );
}

async function submitMutation(
  run: () => Promise<ApiEnvelope<ControlJob>>,
  setMessage: (message: ActionMessage) => void,
  refreshJobs: () => void,
  refreshViews: Array<() => void> = []
): Promise<void> {
  try {
    const envelope = await run();
    const job = envelope.data;
    setMessage({
      kind: statusClass(job.status),
      text: `Job ${job.job_id.slice(0, 8)} ${job.status}: ${operationLabel(job.operation)}`
    });
    refreshJobs();
    refreshViews.forEach((refresh) => refresh());
  } catch (caught: unknown) {
    setMessage({ kind: "bad", text: caught instanceof Error ? caught.message : String(caught) });
  }
}

function AppShell({
  summary,
  system,
  children
}: {
  summary?: DashboardSummary;
  system?: SystemStats;
  children: ReactNode;
}) {
  const page = usePageInfo();
  return (
    <div className="app-shell">
      <aside className="side-rail">
        <Link className="brand-block" to="/operations" aria-label="Clipper operations home">
          <div className="brand-mark">C</div>
          <div>
            <div className="brand-title">Clipper</div>
            <div className="brand-subtitle">Operations</div>
          </div>
        </Link>

        <nav className="nav-list" aria-label="Main navigation">
          {mainNav.map((item) => (
            <NavLink className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`} key={item.path} to={item.path}>
              <item.icon aria-hidden="true" size={18} />
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>

        <nav className="nav-list secondary-nav" aria-label="Support navigation">
          {secondaryNav.map((item) => (
            <NavLink className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`} key={item.path} to={item.path}>
              <item.icon aria-hidden="true" size={18} />
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="rail-metrics" aria-label="Production summary">
          <div>
            <span>Total clips</span>
            <strong>{numberText(summary?.total_clips)}</strong>
          </div>
          <div>
            <span>Videos</span>
            <strong>{numberText(summary?.total_videos)}</strong>
          </div>
        </div>

        <div className="rail-status">
          <span className={`status-dot ${statusClass(healthText(summary))}`} />
          <div>
            <div className="rail-status-main">{healthText(summary)}</div>
            <div className="rail-status-sub">{system?.gpu_label || "System metrics loading"}</div>
          </div>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div>
            <div className="eyebrow">Clipper</div>
            <h1>{page.label}</h1>
            <p>{page.detail}</p>
          </div>
          <div className="topbar-actions">
            <QueueHealthPill summary={summary} />
            <WindowControls />
          </div>
        </header>
        {children}
      </main>
    </div>
  );
}

function QueueHealthPill({ summary }: { summary?: DashboardSummary }) {
  const value = healthText(summary);
  return (
    <Link className={`queue-health-pill ${statusClass(value)}`} to="/system" aria-label="Open queue and system health">
      <span className="status-dot" aria-hidden="true" />
      <span>Queue Health</span>
      <strong>{value}</strong>
    </Link>
  );
}

function WindowControls() {
  const [maximized, setMaximized] = useState(false);
  const canControlWindow = typeof window !== "undefined" && Boolean(window.clipperDesktop?.windowControl);

  if (!canControlWindow) {
    return null;
  }

  async function send(action: WindowControlAction) {
    const result = await window.clipperDesktop?.windowControl?.(action);
    if (result && action !== "close") {
      setMaximized(result.maximized);
    }
  }

  return (
    <div className="window-controls" aria-label="Window controls">
      <button className="window-control-button" onClick={() => void send("minimize")} aria-label="Minimize window">
        <Minus size={15} aria-hidden="true" />
      </button>
      <button className="window-control-button" onClick={() => void send("toggle-maximize")} aria-label={maximized ? "Restore window" : "Maximize window"}>
        <Maximize2 size={14} aria-hidden="true" />
      </button>
      <button className="window-control-button close" onClick={() => void send("close")} aria-label="Close window">
        <X size={15} aria-hidden="true" />
      </button>
    </div>
  );
}

function PageTitle({
  title,
  detail,
  onRefresh,
  children
}: {
  title: string;
  detail: string;
  onRefresh?: () => void;
  children?: ReactNode;
}) {
  return (
    <div className="page-title">
      <div>
        <h2>{title}</h2>
        <p>{detail}</p>
      </div>
      <div className="title-actions">
        {children}
        {onRefresh && (
          <button className="secondary-button" onClick={onRefresh}>
            <RefreshCw size={16} aria-hidden="true" />
            Refresh
          </button>
        )}
      </div>
    </div>
  );
}

function Badge({ value, kind }: { value: string; kind?: BadgeKind }) {
  return (
    <span className={`badge ${kind ?? statusClass(value)}`}>
      <span className="status-dot" aria-hidden="true" />
      {value || "Unknown"}
    </span>
  );
}

function StateBlock({
  kind = "info",
  title,
  detail,
  warnings
}: {
  kind?: BadgeKind;
  title?: string;
  detail?: string;
  warnings?: string[];
}) {
  if (!title && !detail && !warnings?.length) {
    return null;
  }
  return (
    <div className={`state-block ${kind}`}>
      {title && <strong>{title}</strong>}
      {detail && <span>{detail}</span>}
      {warnings?.slice(0, 4).map((warning) => (
        <span key={warning}>{warning}</span>
      ))}
    </div>
  );
}

function ActionNotice({ message }: { message?: ActionMessage }) {
  if (!message) {
    return null;
  }
  return <StateBlock kind={message.kind} detail={message.text} />;
}

function EmptyState({ icon: Icon, title, detail }: { icon: LucideIcon; title: string; detail: string }) {
  return (
    <div className="empty-state">
      <Icon size={22} aria-hidden="true" />
      <strong>{title}</strong>
      <span>{detail}</span>
    </div>
  );
}

function SkeletonLines({ count = 4 }: { count?: number }) {
  return (
    <div className="skeleton-stack" aria-label="Loading">
      {Array.from({ length: count }).map((_, index) => (
        <span className="skeleton-line" key={index} />
      ))}
    </div>
  );
}

function MetricCard({ label, value, hint, icon: Icon }: { label: string; value: string; hint: string; icon?: LucideIcon }) {
  return (
    <div className="metric-card">
      <div className="metric-head">
        <div className="metric-label">{label}</div>
        {Icon && <Icon size={17} aria-hidden="true" />}
      </div>
      <div className="metric-value">{value}</div>
      <div className="metric-hint">{hint}</div>
    </div>
  );
}

function Progress({ value }: { value: number }) {
  const safe = Math.max(0, Math.min(100, value));
  return (
    <div className="progress-cell">
      <div className="progress" aria-label={`Progress ${safe}%`}>
        <span style={{ width: `${safe}%` }} />
      </div>
      <span>{safe}%</span>
    </div>
  );
}

function Drawer({
  open,
  title,
  detail,
  onClose,
  children
}: {
  open: boolean;
  title: string;
  detail?: string;
  onClose: () => void;
  children: ReactNode;
}) {
  if (!open) {
    return null;
  }
  return (
    <aside className="drawer" aria-label={title}>
      <div className="drawer-head">
        <div>
          <h2>{title}</h2>
          {detail && <p>{detail}</p>}
        </div>
        <button className="icon-button" onClick={onClose} aria-label="Close detail panel">
          <X size={18} aria-hidden="true" />
        </button>
      </div>
      <div className="drawer-body">{children}</div>
    </aside>
  );
}

function Pagination({
  total,
  limit,
  offset,
  setOffset
}: {
  total: number;
  limit: number;
  offset: number;
  setOffset: (offset: number) => void;
}) {
  const page = Math.floor(offset / limit) + 1;
  const pages = Math.max(1, Math.ceil(total / limit));
  return (
    <div className="pagination">
      <button className="secondary-button" disabled={offset <= 0} onClick={() => setOffset(Math.max(0, offset - limit))}>
        <ChevronLeft size={16} aria-hidden="true" />
        Previous
      </button>
      <span>Page {page} of {pages}</span>
      <button className="secondary-button" disabled={offset + limit >= total} onClick={() => setOffset(offset + limit)}>
        Next
        <ChevronRight size={16} aria-hidden="true" />
      </button>
    </div>
  );
}

function FilterField({
  label,
  children
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <label className="filter-field">
      <span>{label}</span>
      {children}
    </label>
  );
}

function SearchInput({ value, onChange, placeholder }: { value: string; onChange: (value: string) => void; placeholder: string }) {
  return (
    <div className="search-input">
      <Search size={16} aria-hidden="true" />
      <input value={value} onChange={(event) => onChange(event.target.value)} placeholder={placeholder} />
    </div>
  );
}

function QueueTable({
  rows,
  compact = false,
  selected,
  setSelected
}: {
  rows: QueueRunRow[];
  compact?: boolean;
  selected?: QueueRunRow | null;
  setSelected?: (row: QueueRunRow) => void;
}) {
  if (rows.length === 0) {
    return <EmptyState icon={ListChecks} title="No queue rows" detail="Queue state is empty or not available yet." />;
  }
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Video</th>
            <th>Status</th>
            <th>Step</th>
            <th>Progress</th>
            <th>Clips</th>
            {!compact && <th>Duration</th>}
            {!compact && <th>Attention</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              className={selected?.video_name === row.video_name && selected?.started_at === row.started_at ? "selected-row" : ""}
              key={`${row.video_name}-${row.started_at}`}
              onClick={() => setSelected?.(row)}
            >
              <td>
                <div className="strong">{row.video_name}</div>
                <div className="muted">{row.runs} run(s), {row.redos} redo(s)</div>
              </td>
              <td><Badge value={row.status} /></td>
              <td>{row.current_step}</td>
              <td><Progress value={row.progress} /></td>
              <td>{numberText(row.clips_generated)}</td>
              {!compact && <td>{row.duration}</td>}
              {!compact && <td className="muted attention-cell">{row.attention || "Clear"}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SegmentedControl<T extends string>({
  label,
  value,
  options,
  onChange,
  disabled = false
}: {
  label: string;
  value: T;
  options: Array<{ value: T; label: string }>;
  onChange: (value: T) => void;
  disabled?: boolean;
}) {
  return (
    <div className="launcher-field">
      <span>{label}</span>
      <div className="segmented-control">
        {options.map((option) => (
          <button
            type="button"
            className={value === option.value ? "selected" : ""}
            aria-pressed={value === option.value}
            disabled={disabled}
            key={option.value}
            onClick={() => onChange(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function RunLauncher({
  refreshJobs,
  onQueueRefresh,
  surface = "standard"
}: {
  refreshJobs: () => void;
  onQueueRefresh?: () => void;
  surface?: "standard" | "operations";
}) {
  const queue = usePolling("run-launcher-queue", () => getJson<QueueDetail>("/api/queue"), 2000, true);
  const vods = usePolling("run-launcher-vods", () => getJson<QueueVodList>("/api/queue/vods"), 8000, true);
  const [runMode, setRunMode] = useState<QueueRunMode>("folder_repeat");
  const [pipelineMode, setPipelineMode] = useState<QueuePipelineMode>("full");
  const [variantMode, setVariantMode] = useState<QueueVariantMode>("all");
  const [variantCount, setVariantCount] = useState(2);
  const [maxClips, setMaxClips] = useState("0");
  const [videoPath, setVideoPath] = useState("");
  const [message, setMessage] = useState<ActionMessage>();

  const data = queue.envelope?.data;
  const active = isQueueActive(data);
  const files = vods.envelope?.data?.files ?? [];
  const effectiveVariantMode: QueueVariantMode = pipelineMode === "raw_cuts_only" ? "original" : variantMode;
  const effectiveVariantCount = effectiveVariantMode === "custom" ? variantCount : 1;
  const parsedMaxClips = Math.max(0, Number.parseInt(maxClips || "0", 10) || 0);
  const draftConfig: QueueLaunchConfig = {
    run_mode: runMode,
    pipeline_mode: pipelineMode,
    variant_mode: effectiveVariantMode,
    variant_count: effectiveVariantCount,
    max_clips: parsedMaxClips,
    video_path: runMode === "single_video" ? videoPath : null
  };
  const displayConfig: QueueLaunchConfig = {
    ...draftConfig,
    max_clips: parsedMaxClips === 0 ? null : parsedMaxClips
  };
  const draftSummary = launchSummary(displayConfig);
  const summary = data?.launch_summary || launchSummary(data?.launch_config, draftSummary);
  const needsVod = runMode === "single_video";
  const canStart = !active && (!needsVod || Boolean(videoPath));
  const queueRows = data?.rows ?? [];
  const currentRun = pickCurrentRun(queueRows, active ? "running" : data?.queue_status);
  const activeStage = stageKeyForRun(currentRun);
  const activeStageMeta = operationStages.find((stage) => stage.key === activeStage) ?? operationStages[2];
  const ActiveStageIcon = activeStageMeta.icon;
  const currentProgress = Math.max(0, Math.min(100, currentRun?.progress ?? 0));

  useEffect(() => {
    if (pipelineMode === "raw_cuts_only") {
      setVariantMode("original");
      setVariantCount(1);
    }
  }, [pipelineMode]);

  useEffect(() => {
    if (runMode === "single_video" && !videoPath && files.length > 0) {
      setVideoPath(files[0].path);
    }
  }, [runMode, videoPath, files]);

  function refreshAll() {
    queue.refresh();
    onQueueRefresh?.();
  }

  function startQueue() {
    void submitMutation(
      () => sendJson<ControlJob>("POST", "/api/control/queue", {
        action: "start",
        launch_config: draftConfig
      }),
      setMessage,
      refreshJobs,
      [refreshAll]
    );
  }

  function stopQueue() {
    void submitMutation(
      () => sendJson<ControlJob>("POST", "/api/control/queue", { action: "stop" }),
      setMessage,
      refreshJobs,
      [refreshAll]
    );
  }

  if (surface === "operations") {
    return (
      <>
        <article className="operation-panel current-run-panel">
          <div className="current-run-head">
            <h2>Current Run</h2>
            <Badge value={currentRun?.status || data?.queue_status || "Idle"} kind={runStatusKind(currentRun?.status || data?.queue_status)} />
          </div>

          {currentRun ? (
            <>
              <div className="current-run-main">
                <h3>{currentRun.video_name}</h3>
                <div className="current-stage">
                  <ActiveStageIcon size={28} aria-hidden="true" />
                  <strong>{activeStageMeta.label}</strong>
                </div>
                <div className="run-progress-line" aria-label={`Current run progress ${currentProgress}%`}>
                  <div className="run-progress-track">
                    <span style={{ width: `${currentProgress}%` }} />
                  </div>
                  <strong>{currentProgress}%</strong>
                </div>
              </div>

              <div className="current-run-meta">
                <div className="run-meta-item">
                  <Video size={22} aria-hidden="true" />
                  <span>Clips generated</span>
                  <strong>{numberText(currentRun.clips_generated)}</strong>
                </div>
                <div className="run-meta-item wide">
                  <ListChecks size={22} aria-hidden="true" />
                  <span>Current step</span>
                  <strong>{currentRun.current_step || activeStageMeta.label}</strong>
                </div>
                <div className="run-meta-item">
                  <Clock size={22} aria-hidden="true" />
                  <span>Elapsed</span>
                  <strong>{currentRun.duration || "-"}</strong>
                </div>
              </div>

              <div className="current-run-footer">
                <div className={`run-attention ${currentRun.attention ? "warn" : "good"}`}>
                  {currentRun.attention ? <AlertTriangle size={20} aria-hidden="true" /> : <CheckCircle2 size={20} aria-hidden="true" />}
                  <span>{currentRun.attention || "No issues"}</span>
                </div>
                <button className="danger-button" disabled={!active} onClick={stopQueue}>
                  <Square size={16} aria-hidden="true" />
                  Stop Queue
                </button>
              </div>
            </>
          ) : (
            <div className="operation-empty current-run-empty">
              <span className="operation-empty-mark">
                <Clock size={30} aria-hidden="true" />
              </span>
              <strong>No active run</strong>
              <span>Queue activity will appear here when production starts.</span>
            </div>
          )}
        </article>

        <article className="operation-panel next-run-panel">
          <div className="next-run-head">
            <h2>Next Run</h2>
            <p>Set the next queue pass before production starts.</p>
          </div>
          <div className="next-run-options">
            <div className="next-run-control-card">
              <SegmentedControl label="Run mode" value={runMode} options={runModeOptions} onChange={setRunMode} disabled={active} />
            </div>
            {needsVod && (
              <div className="next-run-control-card wide">
                <FilterField label="VOD">
                  <select value={videoPath} disabled={active} onChange={(event) => setVideoPath(event.target.value)}>
                    <option value="">Select VOD</option>
                    {files.map((file) => (
                      <option value={file.path} key={file.path}>{file.name}</option>
                    ))}
                  </select>
                </FilterField>
              </div>
            )}
            <div className="next-run-control-card wide">
              <SegmentedControl label="Pipeline" value={pipelineMode} options={pipelineModeOptions} onChange={setPipelineMode} disabled={active} />
            </div>
            <div className="next-run-control-card">
              <SegmentedControl
                label="Variants"
                value={effectiveVariantMode}
                options={variantModeOptions}
                onChange={setVariantMode}
                disabled={active || pipelineMode === "raw_cuts_only"}
              />
            </div>
            {effectiveVariantMode === "custom" && (
              <div className="next-run-control-card compact">
                <FilterField label="Variant count">
                  <select value={variantCount} disabled={active} onChange={(event) => setVariantCount(Number.parseInt(event.target.value, 10))}>
                    {[1, 2, 3, 4, 5, 6].map((count) => (
                      <option value={count} key={count}>{count}</option>
                    ))}
                  </select>
                </FilterField>
              </div>
            )}
          </div>
          <div className="next-run-action-row">
            <div className="next-run-summary">
              <span>Ready setup</span>
              <strong>{draftSummary}</strong>
            </div>
            <FilterField label="Max clips">
              <input type="number" min={0} value={maxClips} disabled={active} onChange={(event) => setMaxClips(event.target.value)} />
            </FilterField>
            <button className="primary-button" disabled={!canStart} onClick={startQueue}>
              <Play size={16} aria-hidden="true" />
              Start Queue
            </button>
          </div>
          {active && <StateBlock kind="info" detail={summary} />}
          {needsVod && vods.error && <StateBlock kind="bad" detail={vods.error} />}
          {needsVod && !vods.loading && files.length === 0 && <StateBlock kind="warn" detail="No supported VOD files found." />}
          <ActionNotice message={message} />
        </article>
      </>
    );
  }

  return (
    <article className="panel action-panel launcher-panel">
      <div className="panel-head">
        <div>
          <h2>Run launcher</h2>
          <p>{active ? summary : "Choose the next queue run."}</p>
        </div>
        <Badge value={data?.control_status || data?.queue_status || "idle"} />
      </div>

      {active ? (
        <div className="launcher-running">
          <div className="launcher-summary">
            <Badge value={data?.queue_status ?? "running"} />
            <strong>{summary}</strong>
          </div>
          <button className="danger-button" onClick={stopQueue}>
            <Square size={16} aria-hidden="true" />
            Stop Queue
          </button>
        </div>
      ) : (
        <>
          <div className="launcher-grid">
            <SegmentedControl label="Run mode" value={runMode} options={runModeOptions} onChange={setRunMode} />
            {needsVod && (
              <FilterField label="VOD">
                <select value={videoPath} onChange={(event) => setVideoPath(event.target.value)}>
                  <option value="">Select VOD</option>
                  {files.map((file) => (
                    <option value={file.path} key={file.path}>{file.name}</option>
                  ))}
                </select>
              </FilterField>
            )}
            <SegmentedControl label="Pipeline" value={pipelineMode} options={pipelineModeOptions} onChange={setPipelineMode} />
            <SegmentedControl
              label="Variants"
              value={effectiveVariantMode}
              options={variantModeOptions}
              onChange={setVariantMode}
              disabled={pipelineMode === "raw_cuts_only"}
            />
            {effectiveVariantMode === "custom" && (
              <FilterField label="Variant count">
                <select value={variantCount} onChange={(event) => setVariantCount(Number.parseInt(event.target.value, 10))}>
                  {[1, 2, 3, 4, 5, 6].map((count) => (
                    <option value={count} key={count}>{count}</option>
                  ))}
                </select>
              </FilterField>
            )}
            <FilterField label="Max clips">
              <input type="number" min={0} value={maxClips} onChange={(event) => setMaxClips(event.target.value)} />
            </FilterField>
          </div>
          <div className="launcher-footer">
            <div className="launcher-summary">
              <Badge value={parsedMaxClips === 0 ? "Unlimited" : `${parsedMaxClips} max`} kind="info" />
              <strong>{launchSummary(draftConfig)}</strong>
            </div>
            <button className="primary-button" disabled={!canStart} onClick={startQueue}>
              <Play size={16} aria-hidden="true" />
              Start Queue
            </button>
          </div>
          {needsVod && vods.error && <StateBlock kind="bad" detail={vods.error} />}
          {needsVod && !vods.loading && files.length === 0 && <StateBlock kind="warn" detail="No supported VOD files found." />}
        </>
      )}
      <ActionNotice message={message} />
    </article>
  );
}

function OperationsPage({
  summary,
  system,
  jobs,
  loading,
  error,
  warnings,
  refresh,
  refreshJobs
}: {
  summary?: DashboardSummary;
  system?: SystemStats;
  jobs?: ControlJobPage;
  loading: boolean;
  error?: string;
  warnings?: string[];
  refresh: () => void;
  refreshJobs: () => void;
}) {
  const rows = summary?.rows ?? [];
  const attentionRows = rows.filter((row) => row.attention || ["Needs Attention", "Failed", "Paused"].includes(row.status)).slice(0, 6);
  const recentJobs = jobs?.jobs ?? [];
  const currentRun = pickCurrentRun(rows, summary?.queue_status);
  const activeStage = stageKeyForRun(currentRun);

  return (
    <section className="page-stack operations-page">
      {loading && <SkeletonLines count={4} />}
      {error && <StateBlock kind="bad" title="Dashboard read failed" detail={error} />}
      <StateBlock kind="warn" warnings={warnings} />
      <RunLauncher refreshJobs={refreshJobs} onQueueRefresh={refresh} surface="operations" />

      <article className="operation-panel pipeline-progress-panel">
        <h2>Pipeline Progress</h2>
        <div className="operation-stage-grid">
          {operationStages.map((stage) => {
            const state = operationStageState(stage.key, activeStage, currentRun, summary);
            const progress = operationStageProgress(state, stage.key, activeStage, currentRun);
            const StageIcon = stage.icon;
            const status = state === "done" ? "Done" : state === "running" ? "Running" : "Waiting";
            return (
              <div className={`operation-stage-card ${state}`} key={stage.key}>
                <div className="operation-stage-head">
                  <span className="operation-stage-icon">
                    <StageIcon size={26} aria-hidden="true" />
                  </span>
                  <strong>{stage.label}</strong>
                  <span className="stage-status-pill">{status}</span>
                </div>
                <div className="stage-progress-track" aria-label={`${stage.label} ${status}`}>
                  <span style={{ width: `${progress}%` }} />
                </div>
              </div>
            );
          })}
        </div>
      </article>

      <div className="operation-bottom-grid">
        <article className="operation-panel attention-panel">
          <h2>Needs Attention</h2>
          <div className="attention-list operation-attention-list">
            {attentionRows.map((row) => (
              <Link className="attention-row" to="/queue" key={`${row.video_name}-${row.started_at}`}>
                <AlertTriangle size={16} aria-hidden="true" />
                <div>
                  <strong>{row.video_name}</strong>
                  <span>{row.attention || row.status}</span>
                </div>
                <Badge value={row.status} />
              </Link>
            ))}
            {attentionRows.length === 0 && (
              <div className="operation-empty">
                <span className="operation-empty-mark">
                  <CheckCircle2 size={30} aria-hidden="true" />
                </span>
                <span>No active attention items</span>
              </div>
            )}
          </div>
        </article>

        <article className="operation-panel activity-panel">
          <h2>Recent Activity</h2>
          <div className="operation-activity-list">
            {recentJobs.slice(0, 6).map((job) => (
              <Link className="activity-row" to={`/jobs?job=${encodeURIComponent(job.job_id)}`} key={job.job_id}>
                <span className="activity-icon">
                  <Activity size={17} aria-hidden="true" />
                </span>
                <div>
                  <strong>{operationLabel(job.operation)}</strong>
                  <span>{job.error || job.conflict_key || job.actor}</span>
                </div>
                <Badge value={job.status} />
                <time>{displayTime(job.updated_at)}</time>
              </Link>
            ))}
            {recentJobs.length === 0 && (
              <div className="operation-empty">
                <span className="operation-empty-mark">
                  <Activity size={30} aria-hidden="true" />
                </span>
                <span>No recent activity</span>
              </div>
            )}
          </div>
        </article>
      </div>
    </section>
  );
}

function QueuePage({ refreshJobs }: { refreshJobs: () => void }) {
  const queue = usePolling("queue", () => getJson<QueueDetail>("/api/queue"), 2000, true);
  const [selected, setSelected] = useState<QueueRunRow | null>(null);
  const data = queue.envelope?.data;

  return (
    <section className="page-stack">
      <PageTitle title="Queue runs" detail="Start, continue, stop, and inspect the local production queue." onRefresh={queue.refresh} />
      <RunLauncher refreshJobs={refreshJobs} onQueueRefresh={queue.refresh} />
      {queue.loading && <SkeletonLines count={4} />}
      {queue.error && <StateBlock kind="bad" title="Queue read failed" detail={queue.error} />}
      <StateBlock kind="warn" warnings={queue.envelope?.warnings} />
      <QueueTable rows={data?.rows ?? []} selected={selected} setSelected={setSelected} />
      <Drawer
        open={Boolean(selected)}
        title={selected?.video_name ?? "Queue run"}
        detail={selected?.current_step}
        onClose={() => setSelected(null)}
      >
        {selected && (
          <div className="detail-list">
            <DetailItem label="Status" value={<Badge value={selected.status} />} />
            <DetailItem label="Progress" value={<Progress value={selected.progress} />} />
            <DetailItem label="Video path" value={selected.video_path || "-"} />
            <DetailItem label="Output dir" value={selected.output_dir || "-"} />
            <DetailItem label="Working dir" value={selected.working_dir || "-"} />
            <DetailItem label="Started" value={selected.started_at || "-"} />
            <DetailItem label="Completed" value={selected.completed_at || "-"} />
            <DetailItem label="Attention" value={selected.attention || "Clear"} />
          </div>
        )}
      </Drawer>
    </section>
  );
}

function DetailItem({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="detail-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ClipReviewPage({ active, refreshJobs }: { active: boolean; refreshJobs: () => void }) {
  const limit = 50;
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("");
  const [product, setProduct] = useState("");
  const [sort, setSort] = useState("scored_at");
  const [direction, setDirection] = useState<SortDirection>("desc");
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<string>("");
  const [outputDir, setOutputDir] = useState("");
  const [forceRescore, setForceRescore] = useState(false);
  const [confirm, setConfirm] = useState(false);
  const [message, setMessage] = useState<ActionMessage>();

  useEffect(() => {
    setOffset(0);
  }, [search, status, product, sort, direction]);

  const path = `/api/scores${query({ limit, offset, search, status, product, sort, direction })}`;
  const scores = usePolling(`scores:${path}`, () => getJson<ScoreIndexPage>(path), 10000, active);
  const detail = usePolling(
    `score-detail:${selected}`,
    () => getJson<ScoreDetail>(`/api/scores/${selected}`),
    0,
    active && Boolean(selected)
  );
  const page = scores.envelope?.data;
  const rows = page?.rows ?? [];
  const productOptions = uniqueOptions(rows.map((row) => row.product));

  useEffect(() => {
    const row = rows.find((item) => item.score_key === selected);
    if (row?.clip_path) {
      setOutputDir(parentDir(row.clip_path));
    }
  }, [rows, selected]);

  function submitRescore() {
    void submitMutation(
      () => sendJson<ControlJob>("POST", "/api/operations/rescore", {
        output_dir: outputDir,
        force_rescore: forceRescore
      }),
      setMessage,
      refreshJobs,
      [scores.refresh, detail.refresh]
    );
  }

  return (
    <section className="page-stack">
      <PageTitle title="Clip review" detail="Score, filter, preview, and rescore generated clips." onRefresh={scores.refresh} />
      <div className="filter-row">
        <SearchInput value={search} onChange={setSearch} placeholder="Search clips, products, sources..." />
        <FilterField label="Status">
          <select value={status} onChange={(event) => setStatus(event.target.value)}>
            <option value="">All statuses</option>
            {["Strong", "Okay", "Review", "Blocked"].map((item) => <option value={item} key={item}>{item}</option>)}
          </select>
        </FilterField>
        <FilterField label="Product">
          <select value={product} onChange={(event) => setProduct(event.target.value)}>
            <option value="">All products</option>
            {productOptions.map((item) => <option value={item} key={item}>{item}</option>)}
          </select>
        </FilterField>
        <FilterField label="Sort">
          <select value={sort} onChange={(event) => setSort(event.target.value)}>
            <option value="scored_at">Scored time</option>
            <option value="total_score">Total score</option>
            <option value="quality_score">Quality score</option>
            <option value="similarity_score">Similarity score</option>
            <option value="product">Product</option>
            <option value="status">Status</option>
          </select>
        </FilterField>
        <FilterField label="Direction">
          <select value={direction} onChange={(event) => setDirection(event.target.value as SortDirection)}>
            <option value="desc">Descending</option>
            <option value="asc">Ascending</option>
          </select>
        </FilterField>
      </div>

      <article className="panel action-panel">
        <div className="panel-head">
          <div>
            <h2>Rescore output directory</h2>
            <p>Select a row to fill the output directory, or paste one manually.</p>
          </div>
          <Badge value={forceRescore ? "Force rescore" : "Incremental"} kind={forceRescore ? "warn" : "info"} />
        </div>
        <div className="action-row">
          <FilterField label="Output directory">
            <input value={outputDir} onChange={(event) => setOutputDir(event.target.value)} placeholder="D:\output_clips\vod__run_001" />
          </FilterField>
          <label className="confirm-check">
            <input type="checkbox" checked={forceRescore} onChange={(event) => setForceRescore(event.target.checked)} />
            Force rescore
          </label>
          <label className="confirm-check">
            <input type="checkbox" checked={confirm} onChange={(event) => setConfirm(event.target.checked)} />
            Confirm operation
          </label>
          <button className="primary-button" disabled={!outputDir || !confirm} onClick={submitRescore}>
            <RotateCcw size={16} aria-hidden="true" />
            Create rescore job
          </button>
        </div>
        <ActionNotice message={message} />
      </article>

      {scores.loading && <SkeletonLines count={5} />}
      {scores.error && <StateBlock kind="bad" title="Score read failed" detail={scores.error} />}
      <StateBlock kind="warn" warnings={scores.envelope?.warnings} />
      <ScoreTable rows={rows} selected={selected} setSelected={setSelected} total={page?.total ?? 0} />
      <Pagination total={page?.total ?? 0} limit={limit} offset={offset} setOffset={setOffset} />
      <ScoreDetailDrawer detail={detail.envelope?.data} loading={detail.loading && Boolean(selected)} error={detail.error} onClose={() => setSelected("")} />
    </section>
  );
}

function ScoreTable({
  rows,
  selected,
  setSelected,
  total
}: {
  rows: ScoreRow[];
  selected: string;
  setSelected: (key: string) => void;
  total: number;
}) {
  if (rows.length === 0) {
    return <EmptyState icon={Video} title="No scored clips" detail="Score summaries will appear after clips are rendered and scored." />;
  }
  return (
    <article className="panel">
      <div className="panel-head">
        <div>
          <h2>Score index</h2>
          <p>{numberText(total)} rows available, showing {rows.length}.</p>
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Clip</th>
              <th>Status</th>
              <th>Product</th>
              <th>Total</th>
              <th>Quality</th>
              <th>Flags</th>
              <th>Scored</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr className={selected === row.score_key ? "selected-row" : ""} key={row.score_key} onClick={() => setSelected(row.score_key)}>
                <td>
                  <div className="strong">{row.clip_id || row.source_video}</div>
                  <div className="muted">{row.source_video} - {row.row_type}</div>
                </td>
                <td><Badge value={row.status} /></td>
                <td>{row.product}</td>
                <td>{scoreText(row.total_score)}</td>
                <td>{scoreText(row.quality_score)}</td>
                <td>{row.flag_count}</td>
                <td className="muted">{row.scored_at || "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </article>
  );
}

function ScoreDetailDrawer({
  detail,
  loading,
  error,
  onClose
}: {
  detail?: ScoreDetail;
  loading: boolean;
  error?: string;
  onClose: () => void;
}) {
  const selected = detail?.selected;
  return (
    <Drawer open={Boolean(selected) || loading || Boolean(error)} title={selected?.clip_id || "Selected clip"} detail={selected?.source_video} onClose={onClose}>
      {loading && <SkeletonLines count={5} />}
      {error && <StateBlock kind="bad" title="Clip detail failed" detail={error} />}
      {selected?.artifact?.exists && selected.artifact.kind === "video" && (
        <video className="preview-video" controls preload="metadata" src={selected.artifact.url} />
      )}
      {selected && (
        <>
          <div className="detail-grid">
            <MetricCard label="Total" value={scoreText(selected.total_score)} hint={selected.status} icon={BadgeCheck} />
            <MetricCard label="Quality" value={scoreText(selected.quality_score)} hint="Rendered quality" icon={Eye} />
            <MetricCard label="Similarity" value={scoreText(selected.similarity_score)} hint="Variant distance" icon={Layers3} />
            <MetricCard label="Flags" value={numberText(selected.flag_count)} hint={selected.flag_severity || "none"} icon={AlertTriangle} />
          </div>
          <section className="drawer-section">
            <h3>Flags</h3>
            <div className="chip-row">
              {selected.flags.length ? selected.flags.map((flag) => <span className="chip" key={flag}>{flag}</span>) : <span className="muted">No flags</span>}
            </div>
          </section>
          <section className="drawer-section">
            <h3>Variants</h3>
            <div className="mini-list">
              {(detail?.variants ?? []).map((variant) => (
                <div className="mini-row" key={variant.score_key}>
                  <span>{variant.clip_id || variant.row_type}</span>
                  <Badge value={variant.status} />
                  <strong>{scoreText(variant.similarity_score ?? variant.total_score)}</strong>
                </div>
              ))}
            </div>
          </section>
          <section className="drawer-section">
            <h3>Raw summary</h3>
            <pre className="json-panel">{compactJson(detail?.raw)}</pre>
          </section>
        </>
      )}
    </Drawer>
  );
}

function CompliancePage({ active, refreshJobs }: { active: boolean; refreshJobs: () => void }) {
  const limit = 50;
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("");
  const [product, setProduct] = useState("");
  const [sort, setSort] = useState("checked_at");
  const [direction, setDirection] = useState<SortDirection>("desc");
  const [offset, setOffset] = useState(0);
  const [outputDir, setOutputDir] = useState("");
  const [force, setForce] = useState(true);
  const [confirm, setConfirm] = useState(false);
  const [message, setMessage] = useState<ActionMessage>();

  useEffect(() => {
    setOffset(0);
  }, [search, status, product, sort, direction]);

  const path = `/api/compliance${query({ limit, offset, search, status, product, sort, direction })}`;
  const compliance = usePolling(`compliance:${path}`, () => getJson<ComplianceIndexPage>(path), 10000, active);
  const detailPath = `/api/compliance/detail${query({ output_dir: outputDir })}`;
  const detail = usePolling(
    `compliance-detail:${outputDir}`,
    () => getJson<ComplianceIndexPage>(detailPath),
    0,
    active && Boolean(outputDir)
  );
  const data = compliance.envelope?.data;
  const rows = data?.rows ?? [];
  const detailData = detail.envelope?.data;
  const visibleViolations = detailData?.violations.length ? detailData.violations : data?.violations ?? [];
  const summary = detailData?.summary ?? data?.summary ?? {};
  const productOptions = uniqueOptions(rows.map((row) => row.product));

  function submitScan() {
    void submitMutation(
      () => sendJson<ControlJob>("POST", "/api/operations/compliance-scan", {
        output_dir: outputDir,
        force
      }),
      setMessage,
      refreshJobs,
      [compliance.refresh, detail.refresh]
    );
  }

  return (
    <section className="page-stack">
      <PageTitle title="Compliance" detail="Review policy status, inspect violations, and launch scans." onRefresh={compliance.refresh} />
      <div className="metric-grid compact">
        <MetricCard label="Scanned" value={numberText(summary.scanned)} hint="Filtered rows" icon={ClipboardCheck} />
        <MetricCard label="Passed" value={numberText(summary.passed)} hint="Policy clear" icon={CheckCircle2} />
        <MetricCard label="Blocked" value={numberText(summary.blocked)} hint="Needs action" icon={AlertTriangle} />
        <MetricCard label="Violations" value={numberText(summary.violation_count)} hint="Visible manifest count" icon={ShieldCheck} />
      </div>

      <article className="panel action-panel">
        <div className="panel-head">
          <div>
            <h2>Compliance scan</h2>
            <p>Select a row to fill the output directory, or paste a target under the output root.</p>
          </div>
          <Badge value={force ? "Force scan" : "Incremental"} kind={force ? "warn" : "info"} />
        </div>
        <div className="action-row">
          <FilterField label="Output directory">
            <input value={outputDir} onChange={(event) => setOutputDir(event.target.value)} placeholder="D:\output_clips\vod__run_001" />
          </FilterField>
          <label className="confirm-check">
            <input type="checkbox" checked={force} onChange={(event) => setForce(event.target.checked)} />
            Force scan
          </label>
          <label className="confirm-check">
            <input type="checkbox" checked={confirm} onChange={(event) => setConfirm(event.target.checked)} />
            Confirm operation
          </label>
          <button className="primary-button" disabled={!outputDir || !confirm} onClick={submitScan}>
            <ShieldCheck size={16} aria-hidden="true" />
            Create scan job
          </button>
        </div>
        <ActionNotice message={message} />
      </article>

      <div className="filter-row">
        <SearchInput value={search} onChange={setSearch} placeholder="Search clips, products, sources..." />
        <FilterField label="Status">
          <select value={status} onChange={(event) => setStatus(event.target.value)}>
            <option value="">All statuses</option>
            {["passed", "blocked", "auto_fixed"].map((item) => <option value={item} key={item}>{item}</option>)}
          </select>
        </FilterField>
        <FilterField label="Product">
          <select value={product} onChange={(event) => setProduct(event.target.value)}>
            <option value="">All products</option>
            {productOptions.map((item) => <option value={item} key={item}>{item}</option>)}
          </select>
        </FilterField>
        <FilterField label="Sort">
          <select value={sort} onChange={(event) => setSort(event.target.value)}>
            <option value="checked_at">Checked time</option>
            <option value="violation_count">Violations</option>
            <option value="source_video">Source video</option>
            <option value="product">Product</option>
            <option value="status">Status</option>
          </select>
        </FilterField>
        <FilterField label="Direction">
          <select value={direction} onChange={(event) => setDirection(event.target.value as SortDirection)}>
            <option value="desc">Descending</option>
            <option value="asc">Ascending</option>
          </select>
        </FilterField>
      </div>

      {compliance.loading && <SkeletonLines count={5} />}
      {compliance.error && <StateBlock kind="bad" title="Compliance read failed" detail={compliance.error} />}
      <StateBlock kind="warn" warnings={compliance.envelope?.warnings} />
      <ComplianceTable rows={rows} selectedOutput={outputDir} setSelectedOutput={setOutputDir} />
      <Pagination total={data?.total ?? 0} limit={limit} offset={offset} setOffset={setOffset} />
      <ViolationPanel violations={visibleViolations} loading={detail.loading && Boolean(outputDir)} error={detail.error} />
    </section>
  );
}

function ComplianceTable({
  rows,
  selectedOutput,
  setSelectedOutput
}: {
  rows: ComplianceRow[];
  selectedOutput: string;
  setSelectedOutput: (value: string) => void;
}) {
  if (rows.length === 0) {
    return <EmptyState icon={ShieldCheck} title="No compliance rows" detail="Compliance results will appear after scans run." />;
  }
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Clip</th>
            <th>Status</th>
            <th>Product</th>
            <th>Violations</th>
            <th>Checked</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              className={selectedOutput === row.output_dir ? "selected-row" : ""}
              key={`${row.output_dir}-${row.clip_id}-${row.checked_at}`}
              onClick={() => setSelectedOutput(row.output_dir)}
            >
              <td>
                <div className="strong">{row.clip_id || row.source_video}</div>
                <div className="muted">{row.source_video}</div>
              </td>
              <td><Badge value={row.blocked ? "Blocked" : row.passed ? "Passed" : "Unknown"} /></td>
              <td>{row.product}</td>
              <td>{row.violation_count}</td>
              <td>{row.checked_at || "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ViolationPanel({
  violations,
  loading,
  error
}: {
  violations: ComplianceViolationRow[];
  loading: boolean;
  error?: string;
}) {
  return (
    <article className="panel">
      <div className="panel-head">
        <div>
          <h2>Violation review</h2>
          <p>Severity, source field, original text, and suggested replacement.</p>
        </div>
      </div>
      {loading && <SkeletonLines count={4} />}
      {error && <StateBlock kind="bad" title="Violation detail failed" detail={error} />}
      {violations.length === 0 ? (
        <EmptyState icon={CheckCircle2} title="No visible violations" detail="Select another output directory or run a fresh scan." />
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Clip</th>
                <th>Severity</th>
                <th>Type</th>
                <th>Original</th>
                <th>Suggested</th>
              </tr>
            </thead>
            <tbody>
              {violations.map((row, index) => (
                <tr key={`${row.compliance_file}-${row.clip_id}-${row.field}-${index}`}>
                  <td>
                    <div className="strong">{row.clip_id || row.source_video}</div>
                    <div className="muted">{row.field}</div>
                  </td>
                  <td><Badge value={row.severity || "Review"} /></td>
                  <td>{row.violation_type || "-"}</td>
                  <td className="wide-cell">{row.original_text || "-"}</td>
                  <td className="wide-cell">{row.suggested_replacement || "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </article>
  );
}

function ModulesPage({ active, refreshJobs }: { active: boolean; refreshJobs: () => void }) {
  const limit = 50;
  const readiness = usePolling("module-readiness", () => getJson<ModuleReadiness>("/api/modules/readiness"), 10000, active);
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("");
  const [product, setProduct] = useState("");
  const [sort, setSort] = useState("product");
  const [direction, setDirection] = useState<SortDirection>("asc");
  const [offset, setOffset] = useState(0);
  const libraryPath = `/api/modules/library${query({ limit, offset, search, status, product, sort, direction })}`;
  const library = usePolling(`module-library:${libraryPath}`, () => getJson<ModuleLibraryPage>(libraryPath), 10000, active);
  const [assemblyOpen, setAssemblyOpen] = useState(false);
  const [assemblyLimit, setAssemblyLimit] = useState("");
  const [assemblyProduct, setAssemblyProduct] = useState("");
  const [assemblyZoom, setAssemblyZoom] = useState(false);
  const [assemblyConfirm, setAssemblyConfirm] = useState(false);
  const [selectedModule, setSelectedModule] = useState<ModuleLibraryRow | null>(null);
  const [reviewStatus, setReviewStatus] = useState("approved");
  const [reviewer, setReviewer] = useState("operator");
  const [note, setNote] = useState("");
  const [blockConfirm, setBlockConfirm] = useState(false);
  const [message, setMessage] = useState<ActionMessage>();

  useEffect(() => {
    setOffset(0);
  }, [search, status, product, sort, direction]);

  useEffect(() => {
    if (selectedModule) {
      setReviewStatus(selectedModule.review_status || "approved");
      setNote("");
      setBlockConfirm(false);
    }
  }, [selectedModule?.module_id]);

  const libraryData = library.envelope?.data;
  const rows = libraryData?.rows ?? [];
  const readyProducts = readiness.envelope?.data.rows.filter((row) => row.readiness === "ready") ?? [];
  const productOptions = libraryData?.filter_options.product ?? uniqueOptions(rows.map((row) => row.product_key || row.product));
  const statusOptions = uniqueOptions([
    ...(libraryData?.filter_options.quality_status ?? []),
    ...(libraryData?.filter_options.visual_validation_status ?? []),
    ...(libraryData?.filter_options.review_status ?? [])
  ]);

  function refreshAll() {
    readiness.refresh();
    library.refresh();
  }

  function openAssembly(productKey?: string) {
    setAssemblyProduct(productKey ?? "");
    setAssemblyOpen(true);
  }

  function submitAssembly() {
    const limitValue = assemblyLimit ? Number(assemblyLimit) : null;
    void submitMutation(
      () => sendJson<ControlJob>("POST", "/api/operations/module-assembly", {
        product: assemblyProduct || null,
        module_assembly_limit: limitValue,
        module_product_zoom: assemblyZoom
      }),
      setMessage,
      refreshJobs,
      [refreshAll]
    );
  }

  function submitReview() {
    if (!selectedModule) {
      return;
    }
    void submitMutation(
      () => sendJson<ControlJob>("POST", `/api/modules/${encodeURIComponent(selectedModule.module_id)}/review`, {
        status: reviewStatus,
        reviewer,
        note
      }),
      setMessage,
      refreshJobs,
      [refreshAll]
    );
  }

  return (
    <section className="page-stack">
      <PageTitle title="Module library" detail="Readiness, inventory, assembly, and module review in one workspace." onRefresh={refreshAll}>
        <button className="primary-button" disabled={readyProducts.length === 0} onClick={() => openAssembly()}>
          <Archive size={16} aria-hidden="true" />
          Assemble
        </button>
      </PageTitle>
      <ActionNotice message={message} />
      <StateBlock kind="warn" warnings={[...(readiness.envelope?.warnings ?? []), ...(library.envelope?.warnings ?? [])]} />
      {(readiness.loading || library.loading) && <SkeletonLines count={4} />}
      {(readiness.error || library.error) && <StateBlock kind="bad" title="Module read failed" detail={readiness.error || library.error} />}

      <div className="module-grid">
        {(readiness.envelope?.data.rows ?? []).map((row) => (
          <ReadinessCard row={row} key={row.product_key} onAssemble={() => openAssembly(row.product_key)} />
        ))}
      </div>

      <div className="filter-row">
        <SearchInput value={search} onChange={setSearch} placeholder="Search modules, transcripts, sources..." />
        <FilterField label="Product">
          <select value={product} onChange={(event) => setProduct(event.target.value)}>
            <option value="">All products</option>
            {productOptions.map((item) => <option value={item} key={item}>{item}</option>)}
          </select>
        </FilterField>
        <FilterField label="Status">
          <select value={status} onChange={(event) => setStatus(event.target.value)}>
            <option value="">All statuses</option>
            {statusOptions.map((item) => <option value={item} key={item}>{item}</option>)}
          </select>
        </FilterField>
        <FilterField label="Sort">
          <select value={sort} onChange={(event) => setSort(event.target.value)}>
            <option value="product">Product</option>
            <option value="source_date">Source date</option>
            <option value="duration">Duration</option>
            <option value="confidence">Confidence</option>
            <option value="role">Role</option>
            <option value="status">Status</option>
          </select>
        </FilterField>
        <FilterField label="Direction">
          <select value={direction} onChange={(event) => setDirection(event.target.value as SortDirection)}>
            <option value="asc">Ascending</option>
            <option value="desc">Descending</option>
          </select>
        </FilterField>
      </div>

      <ModuleLibraryTable page={libraryData} selected={selectedModule?.module_id ?? ""} setSelected={setSelectedModule} />
      <Pagination total={libraryData?.total ?? 0} limit={limit} offset={offset} setOffset={setOffset} />

      <Drawer open={assemblyOpen} title="Assemble modules" detail="Build reusable clips from ready module inventory." onClose={() => setAssemblyOpen(false)}>
        <div className="detail-list">
          <DetailItem label="Ready products" value={readyProducts.length ? readyProducts.map((row) => row.product).join(", ") : "None ready"} />
        </div>
        <div className="form-stack">
          <FilterField label="Product">
            <select value={assemblyProduct} onChange={(event) => setAssemblyProduct(event.target.value)}>
              <option value="">All ready products</option>
              {readyProducts.map((row) => <option value={row.product_key} key={row.product_key}>{row.product}</option>)}
            </select>
          </FilterField>
          <FilterField label="Limit">
            <input value={assemblyLimit} onChange={(event) => setAssemblyLimit(event.target.value)} placeholder="optional" inputMode="numeric" />
          </FilterField>
          <label className="confirm-check">
            <input type="checkbox" checked={assemblyZoom} onChange={(event) => setAssemblyZoom(event.target.checked)} />
            Product zoom
          </label>
          <label className="confirm-check">
            <input type="checkbox" checked={assemblyConfirm} onChange={(event) => setAssemblyConfirm(event.target.checked)} />
            Confirm assembly
          </label>
          <button className="primary-button" disabled={!assemblyConfirm || readyProducts.length === 0} onClick={submitAssembly}>
            <Archive size={16} aria-hidden="true" />
            Create assembly job
          </button>
        </div>
      </Drawer>

      <ModuleDetailDrawer
        module={selectedModule}
        reviewStatus={reviewStatus}
        setReviewStatus={setReviewStatus}
        reviewer={reviewer}
        setReviewer={setReviewer}
        note={note}
        setNote={setNote}
        blockConfirm={blockConfirm}
        setBlockConfirm={setBlockConfirm}
        onSubmit={submitReview}
        onClose={() => setSelectedModule(null)}
      />
    </section>
  );
}

function ReadinessCard({ row, onAssemble }: { row: ModuleReadinessRow; onAssemble: () => void }) {
  return (
    <article className="module-card">
      <div className="panel-head">
        <div>
          <h3>{row.product}</h3>
          <p>{row.total} text modules, {row.visual_total} visual records</p>
        </div>
        <Badge value={row.readiness} />
      </div>
      <div className="stage-counts">
        <span>Hook {row.hook}</span>
        <span>Main {row.main}</span>
        <span>CTA {row.cta}</span>
        <span>Zoom {row.zoom_ready_candidates}</span>
      </div>
      <button className="secondary-button module-action" disabled={row.readiness !== "ready"} onClick={onAssemble}>
        <Archive size={16} aria-hidden="true" />
        Assemble
      </button>
    </article>
  );
}

function ModuleLibraryTable({
  page,
  selected,
  setSelected
}: {
  page?: ModuleLibraryPage;
  selected: string;
  setSelected: (row: ModuleLibraryRow) => void;
}) {
  const rows = page?.rows ?? [];
  if (rows.length === 0) {
    return <EmptyState icon={Library} title="No modules indexed" detail="Module inventory will appear after extraction and indexing." />;
  }
  return (
    <article className="panel">
      <div className="panel-head">
        <div>
          <h2>Library inventory</h2>
          <p>{numberText(page?.total)} modules indexed.</p>
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Module</th>
              <th>Product</th>
              <th>Role</th>
              <th>Duration</th>
              <th>Quality</th>
              <th>Review</th>
              <th>Visual</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr className={selected === row.module_id ? "selected-row" : ""} key={row.module_id} onClick={() => setSelected(row)}>
                <td>
                  <div className="strong">{row.module_id}</div>
                  <div className="muted">{row.source_video}</div>
                </td>
                <td>{row.product}</td>
                <td>{row.role}</td>
                <td>{row.duration.toFixed(1)}s</td>
                <td>{row.quality_status || "-"}</td>
                <td>{row.review_status || "-"}</td>
                <td>{row.visual_validation_status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </article>
  );
}

function ModuleDetailDrawer({
  module,
  reviewStatus,
  setReviewStatus,
  reviewer,
  setReviewer,
  note,
  setNote,
  blockConfirm,
  setBlockConfirm,
  onSubmit,
  onClose
}: {
  module: ModuleLibraryRow | null;
  reviewStatus: string;
  setReviewStatus: (value: string) => void;
  reviewer: string;
  setReviewer: (value: string) => void;
  note: string;
  setNote: (value: string) => void;
  blockConfirm: boolean;
  setBlockConfirm: (value: boolean) => void;
  onSubmit: () => void;
  onClose: () => void;
}) {
  return (
    <Drawer open={Boolean(module)} title={module?.module_id ?? "Module detail"} detail={module?.product} onClose={onClose}>
      {module && (
        <>
          {module.file_artifact?.exists && (
            <a className="secondary-button full-width" href={module.file_artifact.url} target="_blank" rel="noreferrer">
              <Eye size={16} aria-hidden="true" />
              Open artifact
            </a>
          )}
          <div className="detail-grid">
            <MetricCard label="Role" value={module.role || "-"} hint={module.product} icon={Layers3} />
            <MetricCard label="Duration" value={`${module.duration.toFixed(1)}s`} hint="Module length" icon={Clock} />
            <MetricCard label="Confidence" value={scoreText(module.confidence)} hint="Extraction confidence" icon={Gauge} />
            <MetricCard label="Visual hits" value={numberText(module.visual_product_hits)} hint={module.visual_validation_status} icon={Eye} />
          </div>
          <section className="drawer-section">
            <h3>Transcript</h3>
            <p className="transcript-box">{module.transcript_text || "No transcript text available."}</p>
          </section>
          <section className="drawer-section">
            <h3>Review action</h3>
            <div className="form-stack">
              <FilterField label="Status">
                <select value={reviewStatus} onChange={(event) => setReviewStatus(event.target.value)}>
                  <option value="approved">Approve</option>
                  <option value="needs_review">Needs review</option>
                  <option value="blocked">Block</option>
                </select>
              </FilterField>
              <FilterField label="Reviewer">
                <input value={reviewer} onChange={(event) => setReviewer(event.target.value)} />
              </FilterField>
              <FilterField label="Note">
                <input value={note} onChange={(event) => setNote(event.target.value)} placeholder="optional" />
              </FilterField>
              {reviewStatus === "blocked" && (
                <label className="confirm-check">
                  <input type="checkbox" checked={blockConfirm} onChange={(event) => setBlockConfirm(event.target.checked)} />
                  Confirm block
                </label>
              )}
              <button className="primary-button" disabled={reviewStatus === "blocked" && !blockConfirm} onClick={onSubmit}>
                <BadgeCheck size={16} aria-hidden="true" />
                Submit review
              </button>
            </div>
          </section>
        </>
      )}
    </Drawer>
  );
}

function ExportsPage({ jobs, refreshJobs }: { jobs?: ControlJobPage; refreshJobs: () => void }) {
  const [outputRoot, setOutputRoot] = useState("");
  const [batchSize, setBatchSize] = useState("");
  const [dryRun, setDryRun] = useState(true);
  const [confirm, setConfirm] = useState(false);
  const [message, setMessage] = useState<ActionMessage>();
  const exportJobs = (jobs?.jobs ?? []).filter((job) => job.operation === "export_batches");

  function submitExport() {
    void submitMutation(
      () => sendJson<ControlJob>("POST", "/api/operations/export-batches", {
        output_root: outputRoot || null,
        batch_size: batchSize ? Number(batchSize) : null,
        dry_run: dryRun
      }),
      setMessage,
      refreshJobs
    );
  }

  return (
    <section className="page-stack">
      <PageTitle title="Exports" detail="Package approved clips into delivery batches." onRefresh={refreshJobs} />
      <article className="panel action-panel">
        <div className="panel-head">
          <div>
            <h2>Package export batches</h2>
            <p>Leave output root empty to use the configured output directory.</p>
          </div>
          <Badge value={dryRun ? "Dry run" : "Final run"} kind={dryRun ? "info" : "warn"} />
        </div>
        <div className="action-row">
          <FilterField label="Output root">
            <input value={outputRoot} onChange={(event) => setOutputRoot(event.target.value)} placeholder="optional output root override" />
          </FilterField>
          <FilterField label="Batch size">
            <input value={batchSize} onChange={(event) => setBatchSize(event.target.value)} placeholder="default" inputMode="numeric" />
          </FilterField>
          <label className="confirm-check">
            <input type="checkbox" checked={dryRun} onChange={(event) => setDryRun(event.target.checked)} />
            Dry run
          </label>
          <label className="confirm-check">
            <input type="checkbox" checked={confirm} onChange={(event) => setConfirm(event.target.checked)} />
            Confirm packaging
          </label>
          <button className="primary-button" disabled={!confirm} onClick={submitExport}>
            <Download size={16} aria-hidden="true" />
            Create export job
          </button>
        </div>
        <ActionNotice message={message} />
      </article>
      <article className="panel">
        <div className="panel-head">
          <div>
            <h2>Recent export jobs</h2>
            <p>Packaging results and dry runs from the control job ledger.</p>
          </div>
        </div>
        {exportJobs.length === 0 ? (
          <EmptyState icon={Download} title="No export jobs yet" detail="Run a dry run first, then package final batches when ready." />
        ) : (
          <JobTable rows={exportJobs} selected="" setSelected={() => undefined} compact />
        )}
      </article>
    </section>
  );
}

const zoomSteps: Array<VariationVariant["zoom_intensity"]> = ["none", "subtle", "normal", "strong"];

function VariationsPage({ active }: { active: boolean }) {
  const variations = usePolling("variations", () => getJson<VariationPageData>("/api/variations"), 30000, active);
  const data = variations.envelope?.data;
  const [draft, setDraft] = useState<VariationProfile | null>(null);
  const [openVariant, setOpenVariant] = useState(0);
  const [selectedPreviewIndex, setSelectedPreviewIndex] = useState(0);
  const [message, setMessage] = useState<ActionMessage>();
  const [presetName, setPresetName] = useState("");
  const [selectedPreset, setSelectedPreset] = useState("");
  const [busy, setBusy] = useState("");
  const previewFrameRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (data?.profile) {
      setDraft(copyProfile(data.profile));
      setSelectedPreviewIndex(0);
    }
  }, [data?.profile.revision]);

  const dirty = Boolean(draft && data?.profile && JSON.stringify(draft) !== JSON.stringify(data.profile));
  const visibleVariants = draft?.variants.slice(0, draft.variant_count) ?? [];
  const limits = data?.limits ?? { min_variants: 1, max_variants: 6 };
  const previewIndex = Math.max(0, Math.min(selectedPreviewIndex, Math.max(0, visibleVariants.length - 1)));
  const previewVariant = visibleVariants[previewIndex];

  function updateDraft(next: VariationProfile) {
    setDraft(next);
  }

  function updateVariant(index: number, patch: Partial<VariationVariant>) {
    if (!draft) {
      return;
    }
    const variants = draft.variants.map((variant, itemIndex) => itemIndex === index ? { ...variant, ...patch } : variant);
    updateDraft({ ...draft, variants });
  }

  function selectPreviewVariant(index: number) {
    const nextIndex = Math.max(0, Math.min(index, Math.max(0, visibleVariants.length - 1)));
    setSelectedPreviewIndex(nextIndex);
    setOpenVariant(nextIndex);
  }

  function updateSubtitleY(index: number, value: number) {
    const subtitle_y_frac = clampNumber(value, 0.08, 0.92);
    updateVariant(index, {
      subtitle_y_frac,
      subtitle_position: subtitlePositionFromY(subtitle_y_frac)
    });
  }

  function updateLetterboxEnabled(index: number, enabled: boolean) {
    const current = visibleVariants[index];
    if (!current) {
      return;
    }
    updateVariant(index, {
      letterbox_enabled: enabled,
      letterbox_top_frac: enabled && current.letterbox_top_frac <= 0 ? 0.2 : current.letterbox_top_frac,
      letterbox_bottom_frac: enabled && current.letterbox_bottom_frac <= 0 ? 0.2 : current.letterbox_bottom_frac
    });
  }

  function moveSubtitleFromPointer(event: ReactPointerEvent<HTMLElement>) {
    if (!previewVariant) {
      return;
    }
    const rect = previewFrameRef.current?.getBoundingClientRect();
    if (!rect || rect.height <= 0) {
      return;
    }
    updateSubtitleY(previewIndex, (event.clientY - rect.top) / rect.height);
  }

  function startSubtitleDrag(event: ReactPointerEvent<HTMLButtonElement>) {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    moveSubtitleFromPointer(event);
  }

  function updateVariantCount(value: number) {
    if (!draft) {
      return;
    }
    const count = Math.max(limits.min_variants, Math.min(limits.max_variants, value));
    const variants = [...draft.variants];
    while (variants.length < count) {
      variants.push(createUiVariant(variants.length, variants[variants.length - 1]));
    }
    updateDraft({ ...draft, variant_count: count, variants });
    setOpenVariant(Math.min(openVariant, count - 1));
    setSelectedPreviewIndex(Math.min(selectedPreviewIndex, count - 1));
  }

  async function saveProfile() {
    if (!draft || !data?.profile) {
      return;
    }
    setBusy("save");
    try {
      const envelope = await sendJson<VariationPageData>("PUT", "/api/variations", {
        profile: draft,
        expected_revision: data.profile.revision
      });
      setDraft(copyProfile(envelope.data.profile));
      setMessage({ kind: "good", text: "Variation profile saved for future renders." });
      variations.refresh();
    } catch (caught) {
      setMessage({ kind: "bad", text: caught instanceof Error ? caught.message : String(caught) });
    } finally {
      setBusy("");
    }
  }

  async function savePreset() {
    if (!draft || !presetName.trim()) {
      return;
    }
    setBusy("preset");
    try {
      await sendJson<Record<string, unknown>>("POST", "/api/variations/presets", {
        name: presetName,
        profile: draft
      });
      setPresetName("");
      setMessage({ kind: "good", text: "Preset saved." });
      variations.refresh();
    } catch (caught) {
      setMessage({ kind: "bad", text: caught instanceof Error ? caught.message : String(caught) });
    } finally {
      setBusy("");
    }
  }

  async function loadPreset() {
    if (!selectedPreset) {
      return;
    }
    setBusy("load");
    try {
      const envelope = await getJson<VariationProfile>(`/api/variations/presets/${encodeURIComponent(selectedPreset)}`);
      setDraft(copyProfile(envelope.data));
      setSelectedPreviewIndex(0);
      setMessage({ kind: "info", text: "Preset loaded into the editor. Save to apply it." });
    } catch (caught) {
      setMessage({ kind: "bad", text: caught instanceof Error ? caught.message : String(caught) });
    } finally {
      setBusy("");
    }
  }

  return (
    <section className="page-stack">
      <PageTitle title="Variations" detail="Configure global clip variants before the next render." onRefresh={variations.refresh}>
        <button className="primary-button" disabled={!draft || !dirty || busy === "save"} onClick={saveProfile}>
          <CheckCircle2 size={16} aria-hidden="true" />
          {busy === "save" ? "Saving" : "Apply to future clips"}
        </button>
      </PageTitle>
      {variations.loading && <SkeletonLines count={5} />}
      {variations.error && <StateBlock kind="bad" title="Variation profile read failed" detail={variations.error} />}
      <StateBlock kind="warn" warnings={variations.envelope?.warnings} />
      <ActionNotice message={message} />
      {draft && data && (
        <div className="variation-layout">
          <article className="panel variation-editor">
            <div className="panel-head">
              <div>
                <h2>Profile</h2>
                <p>Revision {data.profile.revision ? data.profile.revision.slice(0, 12) : "new"}.</p>
              </div>
              <Badge value={dirty ? "Unsaved" : "Saved"} kind={dirty ? "warn" : "good"} />
            </div>
            <div className="variation-top-controls">
              <FilterField label="Number of variants">
                <input
                  type="number"
                  min={limits.min_variants}
                  max={limits.max_variants}
                  value={draft.variant_count}
                  onChange={(event) => updateVariantCount(Number.parseInt(event.target.value || "1", 10))}
                />
              </FilterField>
              <span className="variation-count-note">({limits.min_variants}-{limits.max_variants})</span>
            </div>
            <div className="variant-accordion">
              {visibleVariants.map((variant, index) => (
                <section className={`variant-editor-row ${openVariant === index ? "open" : ""}`} key={`${index}-${variant.name}`}>
                  <button className="variant-row-head" onClick={() => setOpenVariant(openVariant === index ? -1 : index)}>
                    <span className="variant-index">V{index + 1}</span>
                    <strong>{variant.name || `Variant ${index + 1}`}</strong>
                    {variant.letterbox_enabled && <span className="letterbox-chip">Letterbox</span>}
                    <ChevronRight size={16} aria-hidden="true" />
                  </button>
                  {openVariant === index && (
                    <div className="variant-row-body">
                      <FilterField label="Name">
                        <input value={variant.name} onChange={(event) => updateVariant(index, { name: event.target.value })} />
                      </FilterField>
                      <FilterField label="Hook type">
                        <select value={variant.hook_type} onChange={(event) => updateVariant(index, { hook_type: event.target.value })}>
                          {data.hook_types.map((item) => <option value={item} key={item}>{variationLabel(item)}</option>)}
                        </select>
                      </FilterField>
                      <FilterField label="Font">
                        <select value={variant.font_id} onChange={(event) => updateVariant(index, { font_id: event.target.value })}>
                          {data.fonts.map((font) => <option value={font.id ?? font.path ?? ""} key={font.id ?? font.path}>{font.label}</option>)}
                        </select>
                      </FilterField>
                      <ColorField label="Font color" value={variant.font_color} onChange={(value) => updateVariant(index, { font_color: value })} />
                      <ColorField label="Highlight color" value={variant.highlight_color} onChange={(value) => updateVariant(index, { highlight_color: value })} />
                      <ToggleField label="Subtitles" checked={variant.subtitle_enabled} onChange={(value) => updateVariant(index, { subtitle_enabled: value })} />
                      <SegmentedField label="Subtitle placement" value={variant.subtitle_position} options={data.subtitle_positions} disabled={!variant.subtitle_enabled} onChange={(value) => {
                        const subtitle_position = value as VariationVariant["subtitle_position"];
                        updateVariant(index, {
                          subtitle_position,
                          subtitle_y_frac: subtitleYDefault(subtitle_position)
                        });
                      }} />
                      <FilterField label="Color grade">
                        <select value={variant.color_grade} onChange={(event) => updateVariant(index, { color_grade: event.target.value })}>
                          {data.color_grades.map((item) => <option value={item} key={item}>{variationLabel(item)}</option>)}
                        </select>
                      </FilterField>
                      <FilterField label="BGM">
                        <select value={variant.bgm_mode === "selected" ? variant.bgm_path : variant.bgm_mode} onChange={(event) => {
                          const value = event.target.value;
                          if (value === "auto" || value === "none") {
                            updateVariant(index, { bgm_mode: value, bgm_path: "" });
                          } else {
                            updateVariant(index, { bgm_mode: "selected", bgm_path: value });
                          }
                        }}>
                          <option value="auto">Auto from folder</option>
                          <option value="none">No BGM</option>
                          {data.bgm_tracks.map((track) => <option value={track.path ?? ""} key={track.path}>{track.label}</option>)}
                        </select>
                      </FilterField>
                      <ToggleField label="SFX" checked={variant.sfx_enabled} onChange={(value) => updateVariant(index, { sfx_enabled: value })} />
                      <ToggleField label="Product zoom" checked={variant.product_zoom_enabled} onChange={(value) => updateVariant(index, { product_zoom_enabled: value })} />
                      <ZoomField value={variant.zoom_intensity} onChange={(value) => updateVariant(index, { zoom_intensity: value })} />
                      <ToggleField label="Black bars" checked={variant.letterbox_enabled} onChange={(value) => updateLetterboxEnabled(index, value)} />
                    </div>
                  )}
                </section>
              ))}
            </div>
            <div className="preset-row">
              <FilterField label="Save preset">
                <input value={presetName} onChange={(event) => setPresetName(event.target.value)} placeholder="Preset name" />
              </FilterField>
              <button className="secondary-button" disabled={!presetName.trim() || busy === "preset"} onClick={savePreset}>Save preset</button>
              <FilterField label="Load preset">
                <select value={selectedPreset} onChange={(event) => setSelectedPreset(event.target.value)}>
                  <option value="">Choose preset</option>
                  {data.presets.map((preset) => <option value={preset.preset_id} key={preset.preset_id}>{preset.name}</option>)}
                </select>
              </FilterField>
              <button className="secondary-button" disabled={!selectedPreset || busy === "load"} onClick={loadPreset}>Load</button>
            </div>
          </article>
          <article className="panel variation-preview-panel">
            <div className="panel-head">
              <div>
                <h2>Preview</h2>
                <p>{data.preview_source.exists ? `Source ${parentDir(data.preview_source.path)}` : "Missing assets/variation_preview/raw_cut_preview.mp4"}</p>
              </div>
              <Badge value={data.preview_source.exists ? "Fixed clip" : "Missing"} kind={data.preview_source.exists ? "good" : "warn"} />
            </div>
            {previewVariant && (
              <div className="single-preview-stack">
                <div className="variation-preview-toolbar">
                  <FilterField label="Preview variant">
                    <select value={previewIndex} onChange={(event) => selectPreviewVariant(Number.parseInt(event.target.value, 10))}>
                      {visibleVariants.map((variant, index) => (
                        <option value={index} key={`${index}-${variant.name}`}>
                          V{index + 1} {variant.name || `Variant ${index + 1}`}
                        </option>
                      ))}
                    </select>
                  </FilterField>
                  <div className="preview-variant-meta">
                    <span className="variant-index">V{previewIndex + 1}</span>
                    <span>{variationLabel(previewVariant.color_grade)} / {previewVariant.product_zoom_enabled ? variationLabel(previewVariant.zoom_intensity) : "No product zoom"}</span>
                  </div>
                </div>
                <div className="single-preview-shell">
                  <div
                    className={`single-preview-frame grade-${previewVariant.color_grade} ${previewVariant.letterbox_enabled ? "has-bars" : ""}`}
                    ref={previewFrameRef}
                  >
                    {data.preview_source.exists ? (
                      <video src={data.preview_source.url} muted autoPlay loop playsInline />
                    ) : (
                      <div className="preview-placeholder preview-missing">
                        <Video size={34} aria-hidden="true" />
                        <strong>Preview asset missing</strong>
                        <span>assets/variation_preview/raw_cut_preview.mp4</span>
                      </div>
                    )}
                    {previewVariant.letterbox_enabled && (
                      <>
                        <div
                          className="preview-blackbar top"
                          style={{ height: `${clampNumber(previewVariant.letterbox_top_frac, 0, 0.4) * 100}%` }}
                        />
                        <div
                          className="preview-blackbar bottom"
                          style={{ height: `${clampNumber(previewVariant.letterbox_bottom_frac, 0, 0.4) * 100}%` }}
                        />
                      </>
                    )}
                    {previewVariant.subtitle_enabled && (
                      <button
                        type="button"
                        className="subtitle-drag-handle"
                        style={{
                          top: `${clampNumber(previewVariant.subtitle_y_frac, 0.08, 0.92) * 100}%`,
                          color: previewVariant.font_color,
                          borderColor: previewVariant.highlight_color
                        }}
                        aria-label="Subtitle position"
                        onPointerDown={startSubtitleDrag}
                        onPointerMove={(event) => {
                          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                            moveSubtitleFromPointer(event);
                          }
                        }}
                      >
                        <span style={{ color: previewVariant.highlight_color }}>Subtitle</span>
                        <small>{Math.round(clampNumber(previewVariant.subtitle_y_frac, 0.08, 0.92) * 100)}%</small>
                      </button>
                    )}
                  </div>
                </div>
                <div className="preview-adjustments">
                  <PercentControl
                    label="Top bar"
                    value={previewVariant.letterbox_top_frac}
                    max={0.4}
                    disabled={!previewVariant.letterbox_enabled}
                    onChange={(value) => updateVariant(previewIndex, { letterbox_top_frac: value })}
                  />
                  <PercentControl
                    label="Bottom bar"
                    value={previewVariant.letterbox_bottom_frac}
                    max={0.4}
                    disabled={!previewVariant.letterbox_enabled}
                    onChange={(value) => updateVariant(previewIndex, { letterbox_bottom_frac: value })}
                  />
                  <PercentControl
                    label="Subtitle Y"
                    value={previewVariant.subtitle_y_frac}
                    min={0.08}
                    max={0.92}
                    disabled={!previewVariant.subtitle_enabled}
                    onChange={(value) => updateSubtitleY(previewIndex, value)}
                  />
                </div>
              </div>
            )}
          </article>
        </div>
      )}
    </section>
  );
}

function ColorField({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <FilterField label={label}>
      <div className="color-field">
        <input type="color" value={value} onChange={(event) => onChange(event.target.value.toUpperCase())} />
        <input value={value} onChange={(event) => onChange(event.target.value.toUpperCase())} maxLength={7} />
      </div>
    </FilterField>
  );
}

function PercentControl({
  label,
  value,
  min = 0,
  max = 1,
  disabled = false,
  onChange
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  disabled?: boolean;
  onChange: (value: number) => void;
}) {
  const clamped = clampNumber(value, min, max);
  const percent = Math.round(clamped * 100);
  const minPercent = Math.round(min * 100);
  const maxPercent = Math.round(max * 100);
  return (
    <div className={`filter-field percent-control ${disabled ? "control-disabled" : ""}`}>
      <span>{label}</span>
      <div className="percent-control-row">
        <input
          type="range"
          min={minPercent}
          max={maxPercent}
          step={1}
          value={percent}
          disabled={disabled}
          onChange={(event) => onChange(clampNumber(Number.parseInt(event.target.value, 10) / 100, min, max))}
        />
        <input
          type="number"
          min={minPercent}
          max={maxPercent}
          step={1}
          value={percent}
          disabled={disabled}
          aria-label={label}
          onChange={(event) => onChange(clampNumber(Number.parseInt(event.target.value || "0", 10) / 100, min, max))}
        />
        <span>%</span>
      </div>
    </div>
  );
}

function SegmentedField({ label, value, options, disabled = false, onChange }: { label: string; value: string; options: string[]; disabled?: boolean; onChange: (value: string) => void }) {
  return (
    <div className={`filter-field ${disabled ? "control-disabled" : ""}`}>
      <span>{label}</span>
      <div className="segmented-control">
        {options.map((option) => (
          <button className={value === option ? "active" : ""} disabled={disabled} key={option} onClick={() => onChange(option)}>
            {variationLabel(option)}
          </button>
        ))}
      </div>
    </div>
  );
}

function ToggleField({ label, checked, onChange }: { label: string; checked: boolean; onChange: (value: boolean) => void }) {
  return (
    <label className="toggle-field">
      <span>{label}</span>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
    </label>
  );
}

function ZoomField({ value, onChange }: { value: VariationVariant["zoom_intensity"]; onChange: (value: VariationVariant["zoom_intensity"]) => void }) {
  const index = Math.max(0, zoomSteps.indexOf(value));
  return (
    <div className="filter-field">
      <span>Zoom intensity</span>
      <input type="range" min={0} max={zoomSteps.length - 1} step={1} value={index} onChange={(event) => onChange(zoomSteps[Number(event.target.value)] ?? "normal")} />
      <div className="zoom-labels">
        {zoomSteps.map((step) => <span key={step}>{variationLabel(step)}</span>)}
      </div>
    </div>
  );
}

function copyProfile(profile: VariationProfile): VariationProfile {
  return JSON.parse(JSON.stringify(profile)) as VariationProfile;
}

function clampNumber(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.max(min, Math.min(max, value));
}

function subtitleYDefault(position: VariationVariant["subtitle_position"]): number {
  if (position === "top") {
    return 0.34;
  }
  if (position === "center") {
    return 0.58;
  }
  return 0.84;
}

function subtitlePositionFromY(value: number): VariationVariant["subtitle_position"] {
  if (value < 0.46) {
    return "top";
  }
  if (value < 0.70) {
    return "center";
  }
  return "bottom";
}

function createUiVariant(index: number, base?: VariationVariant): VariationVariant {
  return {
    name: `Variant ${index + 1}`,
    hook_type: base?.hook_type ?? "text",
    font_id: base?.font_id ?? "",
    font_color: base?.font_color ?? "#FFFFFF",
    highlight_color: base?.highlight_color ?? "#FFD600",
    subtitle_position: base?.subtitle_position ?? "bottom",
    color_grade: base?.color_grade ?? "original",
    bgm_mode: base?.bgm_mode ?? "auto",
    bgm_path: base?.bgm_path ?? "",
    sfx_enabled: base?.sfx_enabled ?? true,
    zoom_intensity: base?.zoom_intensity ?? "normal",
    product_zoom_enabled: base?.product_zoom_enabled ?? true,
    subtitle_enabled: base?.subtitle_enabled ?? true,
    letterbox_enabled: false,
    subtitle_y_frac: base?.subtitle_y_frac ?? subtitleYDefault(base?.subtitle_position ?? "bottom"),
    letterbox_top_frac: 0,
    letterbox_bottom_frac: 0
  };
}

function variationLabel(value: string): string {
  const labels: Record<string, string> = {
    text: "Text",
    before_after_image: "Before/After image",
    text_before_after_image: "Text + Before/After image",
    b_roll: "B-roll",
    text_b_roll: "Text + B-roll"
  };
  return labels[value] ?? String(value || "text").replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function JobsPage({ active }: { active: boolean }) {
  const params = new URLSearchParams(window.location.search);
  const initialJob = params.get("job") ?? "";
  const [status, setStatus] = useState("");
  const [operation, setOperation] = useState("");
  const [selected, setSelected] = useState(initialJob);
  const jobs = usePolling("jobs-page", () => getJson<ControlJobPage>("/api/control/jobs?limit=100"), 5000, active);
  const detail = usePolling(
    `job-detail:${selected}`,
    () => getJson<ControlJob>(`/api/control/jobs/${selected}`),
    0,
    active && Boolean(selected)
  );
  const rows = (jobs.envelope?.data.jobs ?? []).filter((job) => {
    return (!status || job.status === status) && (!operation || job.operation === operation);
  });
  const operations = uniqueOptions((jobs.envelope?.data.jobs ?? []).map((job) => job.operation));

  return (
    <section className="page-stack">
      <PageTitle title="Jobs" detail="Audit control operations, conflicts, errors, and results." onRefresh={jobs.refresh} />
      <div className="filter-row">
        <FilterField label="Operation">
          <select value={operation} onChange={(event) => setOperation(event.target.value)}>
            <option value="">All operations</option>
            {operations.map((item) => <option value={item} key={item}>{operationLabel(item)}</option>)}
          </select>
        </FilterField>
        <FilterField label="Status">
          <select value={status} onChange={(event) => setStatus(event.target.value)}>
            <option value="">All statuses</option>
            {["queued", "running", "completed", "failed", "interrupted", "rejected"].map((item) => <option value={item} key={item}>{item}</option>)}
          </select>
        </FilterField>
      </div>
      {jobs.loading && <SkeletonLines count={5} />}
      {jobs.error && <StateBlock kind="bad" title="Jobs read failed" detail={jobs.error} />}
      <JobTable rows={rows} selected={selected} setSelected={setSelected} />
      <JobDetailDrawer job={detail.envelope?.data} loading={detail.loading && Boolean(selected)} error={detail.error} onClose={() => setSelected("")} />
    </section>
  );
}

function JobTable({
  rows,
  selected,
  setSelected,
  compact = false
}: {
  rows: ControlJob[];
  selected: string;
  setSelected: (id: string) => void;
  compact?: boolean;
}) {
  if (rows.length === 0) {
    return <EmptyState icon={Activity} title="No jobs match" detail="Change filters or run an operation to create a job." />;
  }
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Job</th>
            <th>Operation</th>
            <th>Status</th>
            <th>Updated</th>
            {!compact && <th>Actor</th>}
            {!compact && <th>Error</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((job) => (
            <tr className={selected === job.job_id ? "selected-row" : ""} key={job.job_id} onClick={() => setSelected(job.job_id)}>
              <td>
                <div className="strong">{job.job_id.slice(0, 12)}</div>
                <div className="muted">{job.conflict_key || "no conflict key"}</div>
              </td>
              <td>{operationLabel(job.operation)}</td>
              <td><Badge value={job.status} /></td>
              <td>{job.updated_at}</td>
              {!compact && <td>{job.actor}</td>}
              {!compact && <td className="wide-cell muted">{job.error || "-"}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function JobDetailDrawer({
  job,
  loading,
  error,
  onClose
}: {
  job?: ControlJob;
  loading: boolean;
  error?: string;
  onClose: () => void;
}) {
  return (
    <Drawer open={Boolean(job) || loading || Boolean(error)} title={job ? operationLabel(job.operation) : "Job detail"} detail={job?.job_id} onClose={onClose}>
      {loading && <SkeletonLines count={5} />}
      {error && <StateBlock kind="bad" title="Job detail failed" detail={error} />}
      {job && (
        <>
          <div className="detail-grid">
            <MetricCard label="Status" value={job.status} hint={job.updated_at} icon={Activity} />
            <MetricCard label="Actor" value={job.actor} hint="Submitted by" icon={BadgeCheck} />
            <MetricCard label="Started" value={job.started_at ? "Yes" : "No"} hint={job.started_at || "-"} icon={Clock} />
            <MetricCard label="Finished" value={job.finished_at ? "Yes" : "No"} hint={job.finished_at || "-"} icon={CheckCircle2} />
          </div>
          {job.error && <StateBlock kind="bad" title="Error" detail={job.error} />}
          <section className="drawer-section">
            <h3>Request</h3>
            <pre className="json-panel">{compactJson(job.request)}</pre>
          </section>
          <section className="drawer-section">
            <h3>Result</h3>
            <pre className="json-panel">{compactJson(job.result)}</pre>
          </section>
        </>
      )}
    </Drawer>
  );
}

function LogsPage({ active }: { active: boolean }) {
  const [lines, setLines] = useState(200);
  const [search, setSearch] = useState("");
  const logs = usePolling(`logs:${lines}`, () => getJson<LogTail>(`/api/logs?lines=${lines}`), 5000, active);
  const visible = (logs.envelope?.data.lines ?? []).filter((line) => !search || line.text.toLowerCase().includes(search.toLowerCase()));
  return (
    <section className="page-stack">
      <PageTitle title="Logs" detail="Newest pipeline.log entries first for troubleshooting." onRefresh={logs.refresh} />
      <div className="filter-row">
        <SearchInput value={search} onChange={setSearch} placeholder="Search visible log lines..." />
        <FilterField label="Lines">
          <select value={lines} onChange={(event) => setLines(Number(event.target.value))}>
            {[100, 200, 500, 1000].map((value) => <option value={value} key={value}>{value}</option>)}
          </select>
        </FilterField>
      </div>
      {logs.loading && <SkeletonLines count={4} />}
      {logs.error && <StateBlock kind="bad" title="Log read failed" detail={logs.error} />}
      <StateBlock kind="warn" warnings={logs.envelope?.warnings} />
      <pre className="log-panel">
        {visible.map((line) => (
          <div key={line.line_number}><span>{line.line_number}</span>{line.text}</div>
        ))}
      </pre>
    </section>
  );
}

function SystemPage({ active }: { active: boolean }) {
  const health = usePolling("health", () => getJson<HealthPayload>("/api/health"), 5000, active);
  const system = usePolling("system-page", () => getJson<SystemStats>("/api/system"), 5000, active);
  const data = system.envelope?.data;
  return (
    <section className="page-stack">
      <PageTitle title="System" detail="API health and local machine resource status." onRefresh={() => { health.refresh(); system.refresh(); }} />
      <div className="metric-grid">
        <MetricCard label="API" value={health.envelope?.data.status ?? "Unknown"} hint={health.envelope?.data.mode ?? "control"} icon={Server} />
        <MetricCard label="CPU" value={data?.cpu_percent == null ? "-" : `${data.cpu_percent.toFixed(0)}%`} hint="Current utilization" icon={Cpu} />
        <MetricCard label="RAM" value={data?.ram_percent == null ? "-" : `${data.ram_percent.toFixed(0)}%`} hint={data?.ram_label ?? "Unavailable"} icon={Monitor} />
        <MetricCard label="Disk" value={data?.disk_percent == null ? "-" : `${data.disk_percent.toFixed(0)}%`} hint={data?.disk_label ?? "Unavailable"} icon={HardDrive} />
      </div>
      <article className="panel">
        <div className="panel-head">
          <div>
            <h2>GPU</h2>
            <p>{data?.gpu_label ?? "Unavailable"}</p>
          </div>
          <Badge value={data?.gpu_label ?? "Unavailable"} kind={data?.gpu_label ? "info" : "neutral"} />
        </div>
        <div className="detail-grid">
          <MetricCard label="GPU load" value={data?.gpu_percent == null ? "-" : `${data.gpu_percent.toFixed(0)}%`} hint="Utilization" icon={Gauge} />
          <MetricCard label="GPU memory" value={data?.gpu_mem_percent == null ? "-" : `${data.gpu_mem_percent.toFixed(0)}%`} hint="Memory usage" icon={Monitor} />
        </div>
      </article>
      {(health.error || system.error) && <StateBlock kind="bad" title="System read failed" detail={health.error || system.error} />}
      <StateBlock kind="warn" warnings={[...(health.envelope?.warnings ?? []), ...(system.envelope?.warnings ?? [])]} />
    </section>
  );
}

function SettingsPage({ active, refreshJobs }: { active: boolean; refreshJobs: () => void }) {
  const settings = usePolling("settings", () => getJson<SettingsReadSnapshot>("/api/settings/effective"), 30000, active);
  const groups = settings.envelope?.data.groups ?? {};
  const revision = settings.envelope?.data.revision ?? "";
  const entries = Object.values(groups).flat();
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [message, setMessage] = useState<ActionMessage>();

  useEffect(() => {
    const next: Record<string, string> = {};
    Object.values(groups).flat().forEach((entry) => {
      next[entry.name] = String(entry.value ?? "");
    });
    setDraft(next);
  }, [revision]);

  function isInvalid(entry: SettingsReadEntry): boolean {
    const raw = draft[entry.name] ?? "";
    if (entry.value_type === "int") {
      return Number.isNaN(Number.parseInt(raw, 10));
    }
    if (entry.value_type === "float") {
      return Number.isNaN(Number.parseFloat(raw));
    }
    return false;
  }

  function parseEntry(entry: SettingsReadEntry): boolean | number | string {
    const raw = draft[entry.name] ?? "";
    if (entry.value_type === "bool") {
      return raw === "true";
    }
    if (entry.value_type === "int") {
      return Number.parseInt(raw, 10);
    }
    if (entry.value_type === "float") {
      return Number.parseFloat(raw);
    }
    return raw;
  }

  const invalidEntries = entries.filter(isInvalid);
  const changedEntries = entries.filter((entry) => !isInvalid(entry) && String(parseEntry(entry)) !== String(entry.value ?? ""));

  function saveChanges() {
    const overrides: Record<string, boolean | number | string> = {};
    changedEntries.forEach((entry) => {
      overrides[entry.name] = parseEntry(entry);
    });
    void submitMutation(
      () => sendJson<ControlJob>("PUT", "/api/settings/overrides", {
        overrides,
        expected_revision: revision
      }),
      setMessage,
      refreshJobs,
      [settings.refresh]
    );
  }

  function deleteOverride(name: string) {
    void submitMutation(
      () => sendJson<ControlJob>("DELETE", `/api/settings/overrides/${encodeURIComponent(name)}${query({ expected_revision: revision })}`),
      setMessage,
      refreshJobs,
      [settings.refresh]
    );
  }

  return (
    <section className="page-stack">
      <PageTitle title="Settings" detail="Edit only registry-backed operator-safe settings." onRefresh={settings.refresh}>
        <button className="primary-button" disabled={!revision || invalidEntries.length > 0 || changedEntries.length === 0} onClick={saveChanges}>
          <Settings size={16} aria-hidden="true" />
          Save {changedEntries.length ? `${changedEntries.length} change(s)` : "changes"}
        </button>
      </PageTitle>
      {settings.loading && <SkeletonLines count={5} />}
      {settings.error && <StateBlock kind="bad" title="Settings read failed" detail={settings.error} />}
      <StateBlock kind="warn" warnings={settings.envelope?.warnings} />
      {invalidEntries.length > 0 && <StateBlock kind="bad" title="Invalid values" detail={`${invalidEntries.length} setting(s) need numeric values before saving.`} />}
      <ActionNotice message={message} />
      <article className="panel">
        <div className="panel-head">
          <div>
            <h2>Override editor</h2>
            <p>Revision {revision ? revision.slice(0, 12) : "loading"}. Values save to the app override file.</p>
          </div>
          <Badge value={changedEntries.length ? `${changedEntries.length} dirty` : "Clean"} kind={changedEntries.length ? "warn" : "good"} />
        </div>
      </article>
      <div className="settings-grid">
        {Object.entries(groups).map(([category, groupEntries]) => (
          <article className="panel" key={category}>
            <div className="panel-head">
              <div>
                <h2>{category}</h2>
                <p>{groupEntries.length} registered values</p>
              </div>
            </div>
            <div className="settings-list">
              {groupEntries.map((entry) => (
                <div className={`setting-row editable-setting ${isInvalid(entry) ? "invalid" : ""}`} key={entry.name}>
                  <div>
                    <strong>{entry.name}</strong>
                    <span>{entry.value_type} - {entry.source}</span>
                    {(entry.minimum !== null || entry.maximum !== null) && (
                      <span>Bounds {entry.minimum ?? "-"} to {entry.maximum ?? "-"}</span>
                    )}
                  </div>
                  {entry.value_type === "bool" ? (
                    <select value={draft[entry.name] ?? "false"} onChange={(event) => setDraft((current) => ({ ...current, [entry.name]: event.target.value }))}>
                      <option value="true">true</option>
                      <option value="false">false</option>
                    </select>
                  ) : (
                    <input value={draft[entry.name] ?? ""} onChange={(event) => setDraft((current) => ({ ...current, [entry.name]: event.target.value }))} />
                  )}
                  <button className="tiny-button" disabled={entry.source !== "settings_override"} onClick={() => deleteOverride(entry.name)}>
                    Delete override
                  </button>
                </div>
              ))}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function RoutedApp() {
  const dashboard = usePolling("dashboard", () => getJson<DashboardSummary>("/api/dashboard"), 2000, true);
  const system = usePolling("system", () => getJson<SystemStats>("/api/system"), 5000, true);
  const jobs = usePolling("control-jobs", () => getJson<ControlJobPage>("/api/control/jobs?limit=12"), 2000, true);
  const summary = dashboard.envelope?.data;

  return (
    <AppShell
      summary={summary}
      system={system.envelope?.data}
    >
      <Routes>
        <Route path="/" element={<Navigate to="/operations" replace />} />
        <Route path="/operations" element={<OperationsPage summary={summary} system={system.envelope?.data} jobs={jobs.envelope?.data} loading={dashboard.loading} error={dashboard.error} warnings={dashboard.envelope?.warnings} refresh={dashboard.refresh} refreshJobs={jobs.refresh} />} />
        <Route path="/queue" element={<QueuePage refreshJobs={jobs.refresh} />} />
        <Route path="/clips" element={<ClipReviewPage active refreshJobs={jobs.refresh} />} />
        <Route path="/compliance" element={<CompliancePage active refreshJobs={jobs.refresh} />} />
        <Route path="/variations" element={<VariationsPage active />} />
        <Route path="/modules" element={<ModulesPage active refreshJobs={jobs.refresh} />} />
        <Route path="/exports" element={<ExportsPage jobs={jobs.envelope?.data} refreshJobs={jobs.refresh} />} />
        <Route path="/jobs" element={<JobsPage active />} />
        <Route path="/logs" element={<LogsPage active />} />
        <Route path="/system" element={<SystemPage active />} />
        <Route path="/settings" element={<SettingsPage active refreshJobs={jobs.refresh} />} />
        <Route path="*" element={<Navigate to="/operations" replace />} />
      </Routes>
    </AppShell>
  );
}

export function App() {
  return (
    <BrowserRouter>
      <RoutedApp />
    </BrowserRouter>
  );
}
