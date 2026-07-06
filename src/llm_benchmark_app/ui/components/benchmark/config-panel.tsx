import { useState, useCallback, useRef, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { EndpointSelector } from "./endpoint-selector";
import { ChipInput } from "./chip-input";
import {
  useStartBenchmark,
  useCancelBenchmark,
  streamBenchmark,
  type BenchmarkConfigIn,
  type ProgressEvent,
} from "@/lib/benchmark-api";
import {
  Play,
  Square,
  Zap,
  Settings2,
  FlaskConical,
  Gauge,
  AlertCircle,
  Terminal,
  Maximize2,
} from "lucide-react";

interface BenchmarkConfig {
  selectedEndpoints: string[];
  inputTokens: number[];
  outputTokens: number[];
  qpsList: number[];
  parallelWorkers: number[];
  requestsPerWorker: number;
  timeout: number;
  maxRetries: number;
  cacheHitRate: number;
}

const DEFAULT_CONFIG: BenchmarkConfig = {
  selectedEndpoints: [],
  inputTokens: [500],
  outputTokens: [200],
  qpsList: [1.0],
  parallelWorkers: [2],
  requestsPerWorker: 5,
  timeout: 300,
  maxRetries: 3,
  cacheHitRate: 0,
};

const PRESETS: Record<string, Partial<BenchmarkConfig>> = {
  quick: {
    inputTokens: [500],
    outputTokens: [200],
    qpsList: [1.0],
    parallelWorkers: [2],
    requestsPerWorker: 5,
  },
  standard: {
    inputTokens: [1000],
    outputTokens: [200, 500, 1000],
    qpsList: [0.5, 1.0],
    parallelWorkers: [4, 6],
    requestsPerWorker: 5,
  },
  stress: {
    inputTokens: [1000],
    outputTokens: [200, 500, 1000],
    qpsList: [1.0, 2.0, 5.0],
    parallelWorkers: [4, 8, 16],
    requestsPerWorker: 5,
    timeout: 600,
  },
};

interface ConfigPanelProps {
  onBenchmarkStarted: (runId: string) => void;
  activeRunId: string | null;
}

export function ConfigPanel({
  onBenchmarkStarted,
  activeRunId,
}: ConfigPanelProps) {
  const [config, setConfig] = useState<BenchmarkConfig>(DEFAULT_CONFIG);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState("");
  const [logs, setLogs] = useState<string[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [logsModalOpen, setLogsModalOpen] = useState(false);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const streamCleanupRef = useRef<(() => void) | null>(null);

  const startMutation = useStartBenchmark();
  const cancelMutation = useCancelBenchmark();

  useEffect(() => {
    return () => {
      streamCleanupRef.current?.();
      streamCleanupRef.current = null;
    };
  }, []);

  const updateConfig = useCallback(
    <K extends keyof BenchmarkConfig>(key: K, value: BenchmarkConfig[K]) => {
      setConfig((prev) => ({ ...prev, [key]: value }));
    },
    [],
  );

  const applyPreset = (preset: keyof typeof PRESETS) => {
    setConfig((prev) => ({ ...prev, ...PRESETS[preset] }));
  };

  const totalCombinations =
    config.qpsList.length *
    config.parallelWorkers.length *
    config.outputTokens.length *
    config.inputTokens.length;

  const estimatedTime =
    totalCombinations *
    config.selectedEndpoints.length *
    config.requestsPerWorker *
    2; // rough seconds estimate

  const handleStart = () => {
    if (config.selectedEndpoints.length === 0) return;

    const body: BenchmarkConfigIn = {
      endpoint_names: config.selectedEndpoints,
      input_tokens: config.inputTokens,
      output_tokens: config.outputTokens,
      qps_list: config.qpsList,
      parallel_workers: config.parallelWorkers,
      requests_per_worker: config.requestsPerWorker,
      timeout: config.timeout,
      max_retries: config.maxRetries,
      cache_hit_rate: config.cacheHitRate,
    };

    setLogs([]);
    setProgress(0);
    setStatusMsg("Starting benchmark...");

    if (Notification.permission === "default") {
      Notification.requestPermission();
    }

    startMutation.mutate(body, {
      onSuccess: (run) => {
        onBenchmarkStarted(run.id);
        setIsStreaming(true);

        streamCleanupRef.current?.();
        streamCleanupRef.current = streamBenchmark(
          run.id,
          (event: ProgressEvent) => {
            if (event.progress !== undefined) setProgress(event.progress);
            if (event.message) {
              setLogs((prev) => [...prev.slice(-100), event.message!]);
              if (event.message.startsWith("Config ") || event.message.startsWith("Benchmark ")) {
                setStatusMsg(event.message);
              }
            }
            if (event.type === "done") {
              setIsStreaming(false);
              streamCleanupRef.current = null;
              if (event.message) setStatusMsg(event.message);

              if (Notification.permission === "granted") {
                const n = new Notification("Benchmark Complete", {
                  body: event.message ?? "Your benchmark run has finished.",
                  icon: "/favicon.ico",
                });
                n.onclick = () => {
                  window.focus();
                  n.close();
                };
              }
            }
          },
        );
      },
    });
  };

  const handleCancel = () => {
    if (activeRunId) {
      cancelMutation.mutate(activeRunId);
      streamCleanupRef.current?.();
      streamCleanupRef.current = null;
      setIsStreaming(false);
    }
  };

  const isRunning = isStreaming || startMutation.isPending;

  useEffect(() => {
    if (logsModalOpen && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, logsModalOpen]);

  return (
    <div className="space-y-4">
      {/* Endpoint selection */}
      <EndpointSelector
        selected={config.selectedEndpoints}
        onSelectionChange={(sel) => updateConfig("selectedEndpoints", sel)}
      />

      {/* Presets */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Zap className="h-4 w-4" />
            Quick Presets
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => applyPreset("quick")}
              className="flex-1"
            >
              <FlaskConical className="h-3.5 w-3.5 mr-1" />
              Quick Test
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => applyPreset("standard")}
              className="flex-1"
            >
              <Settings2 className="h-3.5 w-3.5 mr-1" />
              Standard
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => applyPreset("stress")}
              className="flex-1"
            >
              <Gauge className="h-3.5 w-3.5 mr-1" />
              Stress Test
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Parameters */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Settings2 className="h-4 w-4" />
            Benchmark Parameters
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label className="text-xs font-medium">Input Tokens</Label>
            <ChipInput
              values={config.inputTokens}
              onChange={(v) => updateConfig("inputTokens", v)}
              type="int"
              placeholder="e.g. 1000"
            />
          </div>

          <div className="space-y-2">
            <Label className="text-xs font-medium">Output Tokens</Label>
            <ChipInput
              values={config.outputTokens}
              onChange={(v) => updateConfig("outputTokens", v)}
              type="int"
              placeholder="e.g. 200, 500, 1000"
            />
          </div>

          <div className="space-y-2">
            <Label className="text-xs font-medium">QPS Rates</Label>
            <ChipInput
              values={config.qpsList}
              onChange={(v) => updateConfig("qpsList", v)}
              type="float"
              placeholder="e.g. 0.5, 1.0"
            />
          </div>

          <div className="space-y-2">
            <Label className="text-xs font-medium">Parallel Workers</Label>
            <ChipInput
              values={config.parallelWorkers}
              onChange={(v) => updateConfig("parallelWorkers", v)}
              type="int"
              placeholder="e.g. 4, 6"
            />
          </div>

          <Separator />

          <div className="grid grid-cols-3 gap-3">
            <div className="space-y-1.5">
              <Label className="text-xs font-medium">Req/Worker</Label>
              <Input
                type="number"
                min={1}
                max={10}
                value={config.requestsPerWorker}
                onChange={(e) =>
                  updateConfig("requestsPerWorker", parseInt(e.target.value) || 5)
                }
                className="h-8"
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs font-medium">Timeout (s)</Label>
              <Input
                type="number"
                min={30}
                max={3600}
                value={config.timeout}
                onChange={(e) =>
                  updateConfig("timeout", parseInt(e.target.value) || 300)
                }
                className="h-8"
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs font-medium">Max Retries</Label>
              <Input
                type="number"
                min={0}
                max={10}
                value={config.maxRetries}
                onChange={(e) =>
                  updateConfig("maxRetries", parseInt(e.target.value) || 3)
                }
                className="h-8"
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs font-medium">Cache Hit %</Label>
              <Input
                type="number"
                min={0}
                max={100}
                value={config.cacheHitRate}
                onChange={(e) =>
                  updateConfig(
                    "cacheHitRate",
                    Math.min(100, Math.max(0, parseInt(e.target.value) || 0)),
                  )
                }
                className="h-8"
              />
            </div>
          </div>

          <p className="text-[11px] text-muted-foreground -mt-2">
            Cache Hit %: share of requests sent with an identical, cacheable
            prompt; the rest get a unique prefix that defeats prompt caching.
            0 = cold prefill on every request.
          </p>

          {/* Estimate */}
          <div className="text-xs text-muted-foreground bg-muted/50 rounded-md p-2.5">
            <span className="font-medium">{totalCombinations}</span> config
            combinations ×{" "}
            <span className="font-medium">
              {config.selectedEndpoints.length || "?"}
            </span>{" "}
            endpoint(s) — est.{" "}
            <span className="font-medium">
              ~{estimatedTime > 60 ? `${Math.ceil(estimatedTime / 60)}m` : `${estimatedTime}s`}
            </span>
          </div>
        </CardContent>
      </Card>

      {/* Run controls */}
      <div className="space-y-3">
        {config.selectedEndpoints.length === 0 && (
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Select at least one endpoint to run a benchmark.
            </AlertDescription>
          </Alert>
        )}

        <div className="flex gap-2">
          <Button
            onClick={handleStart}
            disabled={config.selectedEndpoints.length === 0 || isRunning}
            className="flex-1"
            size="lg"
          >
            <Play className="h-4 w-4 mr-2" />
            {isRunning ? "Running..." : "Run Benchmark"}
          </Button>
          {isRunning && (
            <Button
              onClick={handleCancel}
              variant="destructive"
              size="lg"
              className="text-white"
            >
              <Square className="h-4 w-4 mr-1" />
              Cancel
            </Button>
          )}
        </div>

        {/* Progress — always visible; click to open logs modal */}
        <Card
          className={`transition-colors ${isRunning || progress > 0 ? "cursor-pointer hover:border-primary/50 hover:bg-accent/50" : ""}`}
          onClick={isRunning || progress > 0 ? () => setLogsModalOpen(true) : undefined}
        >
          <CardContent className="pt-4 space-y-3">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Progress</span>
              <div className="flex items-center gap-2">
                <Badge variant={isRunning ? "default" : "secondary"}>
                  {Math.round(progress)}%
                </Badge>
                {(isRunning || progress > 0) && (
                  <Maximize2 className="h-3.5 w-3.5 text-muted-foreground" />
                )}
              </div>
            </div>
            <Progress value={progress} className="h-2" />
            <p className="text-xs text-muted-foreground truncate">
              {statusMsg || "Ready to run benchmark"}
            </p>
          </CardContent>
        </Card>

        {(isRunning || progress > 0) && (
          <Dialog open={logsModalOpen} onOpenChange={setLogsModalOpen}>
            <DialogContent className="max-w-none w-[80vw] h-[80vh] flex flex-col">
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <Terminal className="h-4 w-4" />
                  Benchmark Logs
                </DialogTitle>
                <DialogDescription className="flex items-center justify-between">
                  <span>{statusMsg || "Waiting for logs..."}</span>
                  <Badge variant={isRunning ? "default" : "secondary"} className="ml-2">
                    {Math.round(progress)}%
                  </Badge>
                </DialogDescription>
              </DialogHeader>
              <Progress value={progress} className="h-2" />
              <ScrollArea className="flex-1 min-h-0 rounded-md border bg-muted/30 p-3">
                <div className="text-xs font-mono space-y-0.5">
                  {logs.length === 0 ? (
                    <p className="text-muted-foreground italic">No logs yet...</p>
                  ) : (
                    logs.map((log, i) => (
                      <div key={i} className="text-muted-foreground leading-relaxed">
                        {log}
                      </div>
                    ))
                  )}
                  <div ref={logsEndRef} />
                </div>
              </ScrollArea>
            </DialogContent>
          </Dialog>
        )}
      </div>

      {startMutation.isError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Failed to start benchmark:{" "}
            {startMutation.error?.message ?? "Unknown error"}
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}
