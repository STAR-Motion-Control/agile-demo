import { computed, reactive, toRaw } from "vue";

import type {
  GoalUpdatePayload,
  HelloPayload,
  ImageUpdatePayload,
  PlannerUpdatePayload,
  PoseUpdatePayload,
  ReplayTaskDetail,
  ReplayTaskSummary,
  ReplayTimelineMessage,
  SnapshotPayload,
  TaskStatusPayload,
  ViewMode,
  VizEvent,
  VizMessage,
} from "../types";

type ConnectionState = {
  schemaVersion: string | null;
  capabilities: Record<string, boolean>;
  status: "disconnected" | "connecting" | "connected";
};

const runningNavigationStatuses = new Set(["processing", "busy"]);

function isNavigationRunningStatus(status: string) {
  return runningNavigationStatuses.has(status);
}

function createEmptyPlannerState(): SnapshotPayload["planner"] {
  return {
    mode: null,
    global_path: [],
    waypoints: [],
    local_path: [],
    local_goal: null,
    actions: [],
    action_limit: null,
    preview_actions: [],
    camera_intrinsics: null,
    camera_image_size: null,
  };
}

function normalizePlannerState(
  planner:
    | Partial<SnapshotPayload["planner"]>
    | Partial<PlannerUpdatePayload>
    | null
    | undefined,
): SnapshotPayload["planner"] {
  const emptyPlanner = createEmptyPlannerState();
  return {
    mode: planner?.mode ?? emptyPlanner.mode,
    global_path: planner?.global_path ?? emptyPlanner.global_path,
    waypoints: planner?.waypoints ?? emptyPlanner.waypoints,
    local_path: planner?.local_path ?? emptyPlanner.local_path,
    local_goal: planner?.local_goal ?? emptyPlanner.local_goal,
    actions: planner?.actions ?? emptyPlanner.actions,
    action_limit: planner?.action_limit ?? emptyPlanner.action_limit,
    preview_actions: planner?.preview_actions ?? emptyPlanner.preview_actions,
    camera_intrinsics: planner?.camera_intrinsics ?? emptyPlanner.camera_intrinsics,
    camera_image_size: planner?.camera_image_size ?? emptyPlanner.camera_image_size,
  };
}

function normalizeSnapshot(snapshot: SnapshotPayload): SnapshotPayload {
  return {
    ...snapshot,
    task: {
      ...snapshot.task,
      result_status: snapshot.task.result_status ?? null,
    },
    planner: normalizePlannerState(snapshot.planner),
  };
}

const emptySnapshot = (): SnapshotPayload => ({
  type: "snapshot",
  timestamp: 0,
  schema_version: "v1",
  task: {
    task_id: null,
    goal_text: null,
    status: "idle",
    result_status: null,
    success_flag: null,
    message: null,
    state: null,
    dry_run: false,
  },
  goal: {
    pose: null,
    source: null,
  },
  robot: {
    pose: null,
    vpr_pose: null,
  },
  planner: createEmptyPlannerState(),
  images: {},
  events: [],
});

function cloneSnapshot(snapshot: SnapshotPayload) {
  return structuredClone(toRaw(snapshot));
}

function createReplayBaseSnapshot(
  task: ReplayTaskDetail | null,
  finalSnapshot: SnapshotPayload | null,
): SnapshotPayload {
  const snapshot = emptySnapshot();
  snapshot.schema_version = finalSnapshot?.schema_version ?? snapshot.schema_version;
  snapshot.timestamp = task?.started_at ?? 0;
  if (task !== null) {
    snapshot.task.task_id = task.task_id;
    snapshot.task.goal_text = task.goal_text;
  }
  return snapshot;
}

function applyTaskStatus(snapshot: SnapshotPayload, message: TaskStatusPayload) {
  snapshot.task = {
    task_id: message.task_id,
    goal_text: message.goal_text,
    status: message.status,
    result_status: message.result_status ?? null,
    success_flag: message.success_flag,
    message: message.message,
    state: message.state,
    dry_run: message.dry_run,
  };
}

function applyPoseUpdate(snapshot: SnapshotPayload, message: PoseUpdatePayload) {
  snapshot.robot = {
    pose: message.pose,
    vpr_pose: message.vpr_pose,
  };
}

function applyGoalUpdate(snapshot: SnapshotPayload, message: GoalUpdatePayload) {
  snapshot.goal = {
    pose: message.goal.pose,
    source: message.goal.source,
  };
}

function applyPlannerUpdate(snapshot: SnapshotPayload, message: PlannerUpdatePayload) {
  snapshot.planner = normalizePlannerState(message);
}

function applyImageUpdate(snapshot: SnapshotPayload, message: ImageUpdatePayload) {
  snapshot.images = {
    ...snapshot.images,
    ...message.images,
  };
}

function applyEvent(snapshot: SnapshotPayload, message: VizEvent) {
  snapshot.events = [...snapshot.events, message];
}

function applyReplayMessage(snapshot: SnapshotPayload, message: ReplayTimelineMessage) {
  snapshot.schema_version = message.schema_version;
  snapshot.timestamp = message.timestamp;

  switch (message.type) {
    case "task_status":
      applyTaskStatus(snapshot, message);
      break;
    case "pose_update":
      applyPoseUpdate(snapshot, message);
      break;
    case "goal_update":
      applyGoalUpdate(snapshot, message);
      break;
    case "planner_update":
      applyPlannerUpdate(snapshot, message);
      break;
    case "image_update":
      applyImageUpdate(snapshot, message);
      break;
    case "event":
      applyEvent(snapshot, message);
      break;
    default:
      break;
  }
}

function describeReplayMessage(message: ReplayTimelineMessage | null) {
  if (message === null) {
    return "Ready";
  }
  switch (message.type) {
    case "task_status":
      return `Task ${message.status}`;
    case "pose_update":
      return "Robot pose updated";
    case "goal_update":
      return "Goal updated";
    case "planner_update":
      return `Planner ${message.mode ?? "updated"}`;
    case "image_update":
      return "Image frame updated";
    case "event":
      return message.event_name;
    default:
      return message.type;
  }
}

function replayMinDelay(playbackRate: number) {
  return Math.max(8, Math.round(120 / playbackRate));
}

function createInitialReplaySnapshot(replayState: {
  task: ReplayTaskDetail | null;
  finalSnapshot: SnapshotPayload | null;
  hasTimeline: boolean;
}) {
  if (replayState.finalSnapshot !== null && !replayState.hasTimeline) {
    return cloneSnapshot(replayState.finalSnapshot);
  }
  return createReplayBaseSnapshot(replayState.task, replayState.finalSnapshot);
}

export function createVizStore() {
  const connection = reactive<ConnectionState>({
    schemaVersion: null,
    capabilities: {},
    status: "disconnected",
  });
  const liveSnapshot = reactive<SnapshotPayload>(emptySnapshot());
  const replay = reactive({
    mode: "live" as ViewMode,
    tasks: [] as ReplayTaskSummary[],
    selectedTaskId: null as string | null,
    task: null as ReplayTaskDetail | null,
    finalSnapshot: null as SnapshotPayload | null,
    timeline: [] as ReplayTimelineMessage[],
    currentIndex: -1,
    isPlaying: false,
    playbackRate: 5,
    snapshot: emptySnapshot() as SnapshotPayload,
    status: "idle" as "idle" | "loading" | "ready" | "error",
    error: null as string | null,
  });
  const navigationControl = reactive({
    goalTextInput: "",
    labels: [] as string[],
    textPending: false,
    textError: null as string | null,
    dryRun: false,
    stopPending: false,
    stopError: null as string | null,
    manualPending: false,
    manualError: null as string | null,
  });
  const currentSnapshot = computed(() =>
    replay.mode === "replay" ? replay.snapshot : liveSnapshot,
  );
  const isNavigationRunning = computed(() =>
    isNavigationRunningStatus(liveSnapshot.task.status),
  );
  const canStopNavigation = computed(() => isNavigationRunning.value);
  const canSubmitTextNavigation = computed(
    () =>
      !isNavigationRunning.value &&
      !navigationControl.textPending &&
      navigationControl.goalTextInput.trim().length > 0,
  );

  function applyTaskStatusResponse(response: Partial<TaskStatusPayload>) {
    if (typeof response.status !== "string") {
      return;
    }
    applyTaskStatus(liveSnapshot, {
      type: "task_status",
      timestamp: Date.now() / 1000,
      schema_version: connection.schemaVersion ?? "v1",
      task_id: response.task_id ?? null,
      status: response.status,
      result_status: response.result_status ?? null,
      success_flag: response.success_flag ?? null,
      message: response.message ?? null,
      state: response.state ?? null,
      goal_text: response.goal_text ?? null,
      dry_run: response.dry_run ?? false,
    });
  }

  const replayPlayback = computed(() => {
    const totalSteps = replay.timeline.length;
    const currentStep = totalSteps === 0 ? 0 : Math.min(replay.currentIndex + 1, totalSteps);
    const progressPercent = totalSteps === 0 ? 0 : (currentStep / totalSteps) * 100;
    const currentMessage =
      replay.currentIndex >= 0 ? replay.timeline[replay.currentIndex] ?? null : null;

    return {
      currentMessageLabel: describeReplayMessage(currentMessage),
      currentStep,
      totalSteps,
      progressPercent,
      playbackRate: replay.playbackRate,
      canStepBackward: replay.currentIndex >= 0,
      canStepForward: replay.currentIndex < totalSteps - 1,
      hasTimeline: totalSteps > 0,
      isAtEnd: totalSteps > 0 && replay.currentIndex >= totalSteps - 1,
    };
  });

  let socket: WebSocket | null = null;
  let replayTimer: ReturnType<typeof setTimeout> | null = null;

  function syncReplaySnapshot(index: number) {
    const totalSteps = replay.timeline.length;
    if (totalSteps === 0) {
      replay.currentIndex = -1;
      replay.snapshot = createInitialReplaySnapshot({
        task: replay.task,
        finalSnapshot: replay.finalSnapshot,
        hasTimeline: false,
      });
      return;
    }

    const nextIndex = Math.max(-1, Math.min(index, totalSteps - 1));
    if (nextIndex >= totalSteps - 1 && replay.finalSnapshot !== null) {
      replay.currentIndex = nextIndex;
      replay.snapshot = cloneSnapshot(replay.finalSnapshot);
      return;
    }

    const nextSnapshot = createReplayBaseSnapshot(replay.task, replay.finalSnapshot);
    for (const message of replay.timeline.slice(0, nextIndex + 1)) {
      applyReplayMessage(nextSnapshot, message);
    }

    replay.currentIndex = nextIndex;
    replay.snapshot = nextSnapshot;
  }

  function clearReplayTimer() {
    if (replayTimer !== null) {
      clearTimeout(replayTimer);
      replayTimer = null;
    }
  }

  function pauseReplay() {
    replay.isPlaying = false;
    clearReplayTimer();
  }

  function replayDelayFor(nextIndex: number) {
    if (nextIndex <= 0) {
      return Math.max(replayMinDelay(replay.playbackRate), Math.round(300 / replay.playbackRate));
    }
    const previousTimestamp = replay.timeline[nextIndex - 1]?.timestamp ?? 0;
    const nextTimestamp = replay.timeline[nextIndex]?.timestamp ?? previousTimestamp;
    const delayMs = Math.round(((nextTimestamp - previousTimestamp) * 1000) / replay.playbackRate);
    if (!Number.isFinite(delayMs)) {
      return Math.max(replayMinDelay(replay.playbackRate), Math.round(400 / replay.playbackRate));
    }
    return Math.min(1200, Math.max(replayMinDelay(replay.playbackRate), delayMs));
  }

  function scheduleReplayStep() {
    clearReplayTimer();
    if (!replay.isPlaying) {
      return;
    }

    const nextIndex = replay.currentIndex + 1;
    if (nextIndex >= replay.timeline.length) {
      pauseReplay();
      return;
    }

    replayTimer = setTimeout(() => {
      syncReplaySnapshot(nextIndex);
      if (nextIndex >= replay.timeline.length - 1) {
        pauseReplay();
        return;
      }
      scheduleReplayStep();
    }, replayDelayFor(nextIndex));
  }

  function playReplay() {
    if (replay.timeline.length === 0) {
      return;
    }
    if (replay.currentIndex >= replay.timeline.length - 1) {
      syncReplaySnapshot(-1);
    }
    replay.mode = "replay";
    replay.isPlaying = true;
    scheduleReplayStep();
  }

  function stepReplayForward() {
    pauseReplay();
    if (replay.timeline.length === 0) {
      return;
    }
    syncReplaySnapshot(replay.currentIndex + 1);
  }

  function stepReplayBackward() {
    pauseReplay();
    if (replay.timeline.length === 0) {
      return;
    }
    syncReplaySnapshot(replay.currentIndex - 1);
  }

  function restartReplay() {
    pauseReplay();
    syncReplaySnapshot(-1);
  }

  function seekReplay(step: number) {
    pauseReplay();
    if (replay.timeline.length === 0) {
      return;
    }

    const targetStep = Math.max(0, Math.min(Math.round(step), replay.timeline.length));
    syncReplaySnapshot(targetStep - 1);
  }

  function setReplayPlaybackRate(rate: number) {
    const nextRate = Math.max(0.5, Math.min(30, rate));
    replay.playbackRate = nextRate;
    if (replay.isPlaying) {
      scheduleReplayStep();
    }
  }

  function applyHello(message: HelloPayload) {
    connection.schemaVersion = message.schema_version;
    connection.capabilities = { ...message.capabilities };
  }

  function applySnapshot(message: SnapshotPayload) {
    Object.assign(liveSnapshot, cloneSnapshot(normalizeSnapshot(message)));
    connection.schemaVersion = message.schema_version;
  }

  function applyMessage(message: VizMessage) {
    switch (message.type) {
      case "hello":
        applyHello(message);
        break;
      case "snapshot":
        applySnapshot(message);
        break;
      default:
        applyReplayMessage(liveSnapshot, message);
        break;
    }
  }

  function connect(url = buildSocketUrl()) {
    if (typeof WebSocket === "undefined") {
      return;
    }
    if (socket && connection.status === "connected") {
      return;
    }
    connection.status = "connecting";
    socket = new WebSocket(url);
    socket.onopen = () => {
      connection.status = "connected";
    };
    socket.onmessage = (event) => {
      applyMessage(JSON.parse(event.data) as VizMessage);
    };
    socket.onerror = () => {
      connection.status = "disconnected";
    };
    socket.onclose = () => {
      connection.status = "disconnected";
      socket = null;
    };
  }

  function dispose() {
    pauseReplay();
    socket?.close();
  }

  function setMode(mode: ViewMode) {
    replay.mode = mode;
    if (mode === "live") {
      pauseReplay();
      return;
    }
    void loadReplayTasks();
  }

  async function loadReplayTasks() {
    replay.status = "loading";
    replay.error = null;
    try {
      const response = await fetch("/viz/api/replay/tasks");
      if (!response.ok) {
        throw new Error(`replay task request failed: ${response.status}`);
      }
      replay.tasks = (await response.json()) as ReplayTaskSummary[];
      replay.status = "ready";
    } catch (error) {
      replay.status = "error";
      replay.error = error instanceof Error ? error.message : "unknown error";
    }
  }

  async function loadNavigationLabels() {
    navigationControl.textError = null;
    try {
      const response = await fetch("/viz/api/map/metadata");
      if (!response.ok) {
        throw new Error(`metadata request failed: ${response.status}`);
      }

      const metadata = (await response.json()) as { labels?: string[] };
      navigationControl.labels = metadata.labels ?? [];
    } catch (error) {
      navigationControl.textError = error instanceof Error ? error.message : "unknown error";
    }
  }

  async function selectReplayTask(taskId: string) {
    replay.status = "loading";
    replay.error = null;
    replay.selectedTaskId = taskId;
    pauseReplay();

    try {
      const [taskResponse, snapshotResponse, eventsResponse] = await Promise.all([
        fetch(`/viz/api/replay/tasks/${taskId}`),
        fetch(`/viz/api/replay/tasks/${taskId}/snapshot`),
        fetch(`/viz/api/replay/tasks/${taskId}/events`),
      ]);
      if (!taskResponse.ok) {
        throw new Error(`replay task request failed: ${taskResponse.status}`);
      }
      if (!snapshotResponse.ok) {
        throw new Error(`replay snapshot request failed: ${snapshotResponse.status}`);
      }
      if (!eventsResponse.ok) {
        throw new Error(`replay events request failed: ${eventsResponse.status}`);
      }

      const task = (await taskResponse.json()) as ReplayTaskDetail;
      const snapshot = normalizeSnapshot((await snapshotResponse.json()) as SnapshotPayload);
      const timeline = [...((await eventsResponse.json()) as ReplayTimelineMessage[])].sort(
        (left, right) => left.timestamp - right.timestamp,
      );

      replay.task = task;
      replay.finalSnapshot = cloneSnapshot(snapshot);
      replay.timeline = timeline;
      replay.mode = "replay";
      syncReplaySnapshot(-1);
      replay.status = "ready";
    } catch (error) {
      replay.status = "error";
      replay.error = error instanceof Error ? error.message : "unknown error";
    }
  }

  async function stopNavigation() {
    if (navigationControl.stopPending) {
      return;
    }

    navigationControl.stopPending = true;
    navigationControl.stopError = null;
    try {
      const response = await fetch("/viz/api/navigation/stop", {
        method: "POST",
      });
      if (!response.ok) {
        throw new Error(`stop navigation request failed: ${response.status}`);
      }
      await response.json();
    } catch (error) {
      navigationControl.stopError = error instanceof Error ? error.message : "unknown error";
    } finally {
      navigationControl.stopPending = false;
    }
  }

  function setGoalTextInput(goalText: string) {
    navigationControl.goalTextInput = goalText;
  }

  function setNavigationDryRun(dryRun: boolean) {
    navigationControl.dryRun = dryRun;
  }

  async function submitTextNavigation() {
    const goalText = navigationControl.goalTextInput.trim();
    if (!goalText || navigationControl.textPending) {
      return;
    }

    if (isNavigationRunning.value) {
      navigationControl.textError = "navigation already running";
      return;
    }

    navigationControl.textPending = true;
    navigationControl.textError = null;
    try {
      const response = await fetch("/viz/api/navigation/text", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ goal_text: goalText, dry_run: navigationControl.dryRun }),
      });
      if (!response.ok) {
        throw new Error(`text navigation request failed: ${response.status}`);
      }
      const result = (await response.json()) as Partial<TaskStatusPayload> & {
        accepted?: boolean;
        message?: string;
      };
      applyTaskStatusResponse(result);
      if (result.accepted === false) {
        navigationControl.textError = result.message ?? "navigation request rejected";
      }
      replay.mode = "live";
    } catch (error) {
      navigationControl.textError = error instanceof Error ? error.message : "unknown error";
    } finally {
      navigationControl.textPending = false;
    }
  }

  async function sendManualControl(action: string) {
    if (navigationControl.manualPending) return;
    navigationControl.manualPending = true;
    navigationControl.manualError = null;
    try {
      const response = await fetch("/viz/api/navigation/manual", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      });
      const result = await response.json() as { accepted?: boolean; message?: string; detail?: string };
      if (!response.ok || result.accepted === false) {
        throw new Error(result.message ?? result.detail ?? `manual control request failed: ${response.status}`);
      }
    } catch (error) {
      navigationControl.manualError = error instanceof Error ? error.message : "unknown error";
    } finally {
      navigationControl.manualPending = false;
    }
  }

  return {
    canSubmitTextNavigation,
    canStopNavigation,
    connection,
    currentSnapshot,
    liveSnapshot,
    navigationControl,
    replay,
    replayPlayback,
    applyMessage,
    connect,
    dispose,
    loadReplayTasks,
    loadNavigationLabels,
    pauseReplay,
    playReplay,
    seekReplay,
    setReplayPlaybackRate,
    restartReplay,
    selectReplayTask,
    setGoalTextInput,
    setNavigationDryRun,
    setMode,
    sendManualControl,
    stopNavigation,
    submitTextNavigation,
    stepReplayBackward,
    stepReplayForward,
  };
}

function buildSocketUrl() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/viz/ws`;
}
