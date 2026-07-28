import { afterEach, describe, expect, it, vi } from "vitest";

import { createVizStore } from "./vizStore";

describe("createVizStore", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("stores hello and snapshot payloads", () => {
    const store = createVizStore();

    expect(store.replayPlayback.value.playbackRate).toBe(5);

    store.applyMessage({
      type: "hello",
      timestamp: 1,
      schema_version: "v1",
      capabilities: {
        realtime: true,
        replay: true,
        map_overlay: true,
      },
    });
    store.applyMessage({
      type: "snapshot",
      timestamp: 2,
      schema_version: "v1",
      task: {
        task_id: "task-1",
        goal_text: "go",
        status: "processing",
        success_flag: null,
        message: "accepted",
        state: null,
        dry_run: false,
      },
      goal: {
        pose: { x: 1, y: 2, theta: 0.3 },
        source: "embedding",
      },
      robot: {
        pose: { x: 0, y: 0, theta: 0 },
        vpr_pose: { x: 0.1, y: 0.2, theta: 0.1 },
      },
      planner: {
        mode: "navdp",
        global_path: [[0, 0], [1, 1]],
        waypoints: [[1, 1]],
        local_path: [[0, 0], [0.2, 0.1]],
        local_goal: [0.5, 0.2],
        actions: [[0.1, 0.25]],
        action_limit: 2,
        preview_actions: [[0.1, 0.25], [0, 0.25]],
        camera_intrinsics: [
          [387, 0, 320],
          [0, 386, 243],
          [0, 0, 1],
        ],
        camera_image_size: [640, 480],
      },
      images: {
        rgb_latest: { url: "/viz/api/frame/rgb/latest.jpg?t=1", version: 1 },
        rgb_navdp: { url: "/viz/api/frame/rgb/navdp.jpg?t=1", version: 1 },
      },
      events: [],
    });

    expect(store.connection.schemaVersion).toBe("v1");
    expect(store.liveSnapshot.task.task_id).toBe("task-1");
    expect(store.liveSnapshot.images.rgb_latest?.version).toBe(1);
    expect(store.liveSnapshot.images.rgb_navdp?.version).toBe(1);
    expect(store.liveSnapshot.planner.preview_actions).toHaveLength(2);
  });

  it("merges incremental updates into normalized state", () => {
    const store = createVizStore();

    store.applyMessage({
      type: "snapshot",
      timestamp: 1,
      schema_version: "v1",
      task: {
        task_id: null,
        goal_text: null,
        status: "idle",
        success_flag: null,
        message: null,
        state: null,
        dry_run: false,
      },
      goal: { pose: null, source: null },
      robot: { pose: null, vpr_pose: null },
      planner: {
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
      },
      images: {},
      events: [],
    });

    store.applyMessage({
      type: "task_status",
      timestamp: 2,
      schema_version: "v1",
      task_id: "task-2",
      status: "processing",
      success_flag: null,
      message: "running",
      state: null,
      goal_text: "door",
      dry_run: false,
    });
    store.applyMessage({
      type: "pose_update",
      timestamp: 3,
      schema_version: "v1",
      pose: { x: 1.2, y: -3.4, theta: 0.7 },
      vpr_pose: { x: 1.1, y: -3.5, theta: 0.6 },
    });
    store.applyMessage({
      type: "goal_update",
      timestamp: 3.5,
      schema_version: "v1",
      goal: {
        pose: { x: 2.5, y: -1.1, theta: 1.2 },
        source: "goal_recognition",
      },
    });
    store.applyMessage({
      type: "planner_update",
      timestamp: 4,
      schema_version: "v1",
      mode: "line_segment",
      global_path: [[1, 2], [3, 4]],
      waypoints: [[3, 4]],
      local_path: [],
      local_goal: null,
      actions: [[0.2, 0.5]],
      action_limit: null,
      preview_actions: [],
      camera_intrinsics: null,
      camera_image_size: null,
    });
    store.applyMessage({
      type: "image_update",
      timestamp: 5,
      schema_version: "v1",
      images: {
        rgb_latest: { url: "/viz/api/frame/rgb/latest.jpg?t=2", version: 2 },
      },
    });
    store.applyMessage({
      type: "event",
      timestamp: 6,
      schema_version: "v1",
      event_name: "planner_selected",
      level: "info",
      task_id: "task-2",
      payload: { mode: "line_segment" },
    });

    expect(store.liveSnapshot.task.task_id).toBe("task-2");
    expect(store.liveSnapshot.robot.pose?.x).toBe(1.2);
    expect(store.liveSnapshot.goal.pose?.x).toBe(2.5);
    expect(store.liveSnapshot.planner.mode).toBe("line_segment");
    expect(store.liveSnapshot.images.rgb_latest?.version).toBe(2);
    expect(store.liveSnapshot.events.at(-1)?.event_name).toBe("planner_selected");
  });

  it("keeps navdp local preview points when planner updates include a short local path", () => {
    const store = createVizStore();

    store.applyMessage({
      type: "snapshot",
      timestamp: 1,
      schema_version: "v1",
      task: {
        task_id: "task-3",
        goal_text: "desk",
        status: "processing",
        success_flag: null,
        message: null,
        state: null,
        dry_run: false,
      },
      goal: { pose: null, source: null },
      robot: {
        pose: { x: 7.76, y: 4.01, theta: -1.8 },
        vpr_pose: null,
      },
      planner: {
        mode: "navdp",
        global_path: [],
        waypoints: [],
        local_path: [
          [7.7632, 4.016],
          [7.7632, 4.016],
          [7.6953, 3.7754],
        ],
        local_goal: [6.086, 2.711],
        actions: [[-0.2618, 0], [0, 0.25]],
        action_limit: 2,
        preview_actions: [[-0.2618, 0], [0, 0.25], [0, 0.25]],
        camera_intrinsics: [
          [387, 0, 320],
          [0, 386, 243],
          [0, 0, 1],
        ],
        camera_image_size: [640, 480],
      },
      images: {},
      events: [],
    });

    expect(store.liveSnapshot.planner.mode).toBe("navdp");
    expect(store.liveSnapshot.planner.local_path).toHaveLength(3);
    expect(store.liveSnapshot.planner.local_path.at(-1)).toEqual([7.6953, 3.7754]);
    expect(store.liveSnapshot.planner.preview_actions).toHaveLength(3);
    expect(store.liveSnapshot.planner.camera_intrinsics?.[0]?.[0]).toBe(387);
  });

  it("rebuilds replay snapshot step by step from the recorded timeline", async () => {
    const store = createVizStore();
    const payloads = {
      "/viz/api/replay/tasks": [
        {
          task_id: "task-1",
          goal_text: "door",
          status: "completed",
          started_at: 1,
          ended_at: 4,
          has_snapshot: true,
          event_count: 4,
        },
      ],
      "/viz/api/replay/tasks/task-1": {
        task_id: "task-1",
        goal_text: "door",
        status: "completed",
        started_at: 1,
        ended_at: 4,
        has_snapshot: true,
        event_count: 4,
      },
      "/viz/api/replay/tasks/task-1/snapshot": {
        type: "snapshot",
        timestamp: 4,
        schema_version: "v1",
        task: {
          task_id: "task-1",
          goal_text: "door",
          status: "completed",
          success_flag: true,
          message: "done",
          state: 1,
          dry_run: true,
        },
        goal: { pose: { x: 1, y: 2, theta: 0.3 }, source: "goal_recognition" },
        robot: { pose: { x: 1, y: 1, theta: 0.1 }, vpr_pose: { x: 1, y: 1, theta: 0.1 } },
        planner: {
          mode: "line_segment",
          global_path: [[0, 0], [1, 1], [2, 1.5]],
          waypoints: [[2, 1.5]],
          local_path: [],
          local_goal: null,
          actions: [[0.1, 0.2]],
          action_limit: null,
          preview_actions: [],
          camera_intrinsics: null,
          camera_image_size: null,
        },
        images: {},
        events: [
          {
            type: "event",
            timestamp: 4,
            schema_version: "v1",
            event_name: "task_completed",
            level: "info",
            task_id: "task-1",
            payload: { status: "completed" },
          },
        ],
      },
      "/viz/api/replay/tasks/task-1/events": [
        {
          type: "task_status",
          timestamp: 1,
          schema_version: "v1",
          task_id: "task-1",
          status: "processing",
          success_flag: null,
          message: "running",
          state: null,
          goal_text: "door",
          dry_run: true,
        },
        {
          type: "planner_update",
          timestamp: 2,
          schema_version: "v1",
          mode: "line_segment",
          global_path: [[0, 0], [1, 1]],
          waypoints: [[1, 1]],
          local_path: [],
          local_goal: null,
          actions: [[0.1, 0.2]],
          action_limit: null,
          preview_actions: [],
          camera_intrinsics: null,
          camera_image_size: null,
        },
        {
          type: "planner_update",
          timestamp: 3,
          schema_version: "v1",
          mode: "line_segment",
          global_path: [[0, 0], [1, 1], [2, 1.5]],
          waypoints: [[2, 1.5]],
          local_path: [],
          local_goal: null,
          actions: [[0.1, 0.2]],
          action_limit: null,
          preview_actions: [],
          camera_intrinsics: null,
          camera_image_size: null,
        },
        {
          type: "event",
          timestamp: 4,
          schema_version: "v1",
          event_name: "task_completed",
          level: "info",
          task_id: "task-1",
          payload: { status: "completed" },
        },
      ],
    } as const;

    globalThis.fetch = (async (input: RequestInfo | URL) => {
      const url = String(input);
      const payload = payloads[url as keyof typeof payloads];
      return {
        ok: true,
        json: async () => payload,
      } as Response;
    }) as typeof fetch;

    await store.loadReplayTasks();
    await store.selectReplayTask("task-1");

    expect(store.replay.mode).toBe("replay");
    expect(store.currentSnapshot.value.task.task_id).toBe("task-1");
    expect(store.currentSnapshot.value.planner.global_path).toEqual([]);
    expect(store.replayPlayback.value.currentStep).toBe(0);

    store.stepReplayForward();
    expect(store.currentSnapshot.value.task.status).toBe("processing");
    expect(store.currentSnapshot.value.planner.global_path).toEqual([]);

    store.stepReplayForward();
    expect(store.currentSnapshot.value.planner.global_path).toEqual([
      [0, 0],
      [1, 1],
    ]);

    store.stepReplayForward();
    expect(store.currentSnapshot.value.planner.global_path).toEqual([
      [0, 0],
      [1, 1],
      [2, 1.5],
    ]);

    store.stepReplayBackward();
    expect(store.currentSnapshot.value.planner.global_path).toEqual([
      [0, 0],
      [1, 1],
    ]);

    store.seekReplay(3);
    expect(store.replayPlayback.value.currentStep).toBe(3);
    expect(store.currentSnapshot.value.planner.global_path).toEqual([
      [0, 0],
      [1, 1],
      [2, 1.5],
    ]);

    store.seekReplay(0);
    expect(store.replayPlayback.value.currentStep).toBe(0);
    expect(store.currentSnapshot.value.planner.global_path).toEqual([]);
  });

  it("plays the replay timeline through to the final snapshot", async () => {
    vi.useFakeTimers();

    const store = createVizStore();
    const payloads = {
      "/viz/api/replay/tasks/task-1": {
        task_id: "task-1",
        goal_text: "door",
        status: "completed",
        started_at: 1,
        ended_at: 4,
        has_snapshot: true,
        event_count: 3,
      },
      "/viz/api/replay/tasks/task-1/snapshot": {
        type: "snapshot",
        timestamp: 4,
        schema_version: "v1",
        task: {
          task_id: "task-1",
          goal_text: "door",
          status: "completed",
          success_flag: true,
          message: "done",
          state: 1,
          dry_run: false,
        },
        goal: { pose: null, source: null },
        robot: { pose: null, vpr_pose: null },
        planner: {
          mode: "line_segment",
          global_path: [[0, 0], [1, 1]],
          waypoints: [[1, 1]],
          local_path: [],
          local_goal: null,
          actions: [],
          action_limit: null,
          preview_actions: [],
          camera_intrinsics: null,
          camera_image_size: null,
        },
        images: {},
        events: [
          {
            type: "event",
            timestamp: 4,
            schema_version: "v1",
            event_name: "task_completed",
            level: "info",
            task_id: "task-1",
            payload: { status: "completed" },
          },
        ],
      },
      "/viz/api/replay/tasks/task-1/events": [
        {
          type: "task_status",
          timestamp: 1,
          schema_version: "v1",
          task_id: "task-1",
          status: "processing",
          success_flag: null,
          message: "running",
          state: null,
          goal_text: "door",
          dry_run: false,
        },
        {
          type: "planner_update",
          timestamp: 1.2,
          schema_version: "v1",
          mode: "line_segment",
          global_path: [[0, 0], [1, 1]],
          waypoints: [[1, 1]],
          local_path: [],
          local_goal: null,
          actions: [],
          action_limit: null,
          preview_actions: [],
          camera_intrinsics: null,
          camera_image_size: null,
        },
        {
          type: "event",
          timestamp: 1.5,
          schema_version: "v1",
          event_name: "task_completed",
          level: "info",
          task_id: "task-1",
          payload: { status: "completed" },
        },
      ],
    } as const;

    globalThis.fetch = (async (input: RequestInfo | URL) => {
      const url = String(input);
      const payload = payloads[url as keyof typeof payloads];
      return {
        ok: true,
        json: async () => payload,
      } as Response;
    }) as typeof fetch;

    await store.selectReplayTask("task-1");

    store.setReplayPlaybackRate(30);
    store.playReplay();
    await vi.advanceTimersByTimeAsync(200);

    expect(store.replay.isPlaying).toBe(false);
    expect(store.replayPlayback.value.isAtEnd).toBe(true);
    expect(store.replayPlayback.value.playbackRate).toBe(30);
    expect(store.currentSnapshot.value.task.status).toBe("completed");
    expect(store.currentSnapshot.value.events.at(-1)?.event_name).toBe("task_completed");
  });

  it("uses different playback pacing for 1x and 30x replay", async () => {
    vi.useFakeTimers();

    const store = createVizStore();
    const payloads = {
      "/viz/api/replay/tasks/task-1": {
        task_id: "task-1",
        goal_text: "door",
        status: "completed",
        started_at: 1,
        ended_at: 2,
        has_snapshot: true,
        event_count: 2,
      },
      "/viz/api/replay/tasks/task-1/snapshot": {
        type: "snapshot",
        timestamp: 2,
        schema_version: "v1",
        task: {
          task_id: "task-1",
          goal_text: "door",
          status: "completed",
          success_flag: true,
          message: "done",
          state: 1,
          dry_run: false,
        },
        goal: { pose: null, source: null },
        robot: { pose: null, vpr_pose: null },
        planner: {
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
        },
        images: {},
        events: [],
      },
      "/viz/api/replay/tasks/task-1/events": [
        {
          type: "task_status",
          timestamp: 1,
          schema_version: "v1",
          task_id: "task-1",
          status: "processing",
          success_flag: null,
          message: "running",
          state: null,
          goal_text: "door",
          dry_run: false,
        },
        {
          type: "event",
          timestamp: 1.01,
          schema_version: "v1",
          event_name: "task_completed",
          level: "info",
          task_id: "task-1",
          payload: { status: "completed" },
        },
      ],
    } as const;

    globalThis.fetch = (async (input: RequestInfo | URL) => {
      const url = String(input);
      const payload = payloads[url as keyof typeof payloads];
      return {
        ok: true,
        json: async () => payload,
      } as Response;
    }) as typeof fetch;

    await store.selectReplayTask("task-1");

    store.setReplayPlaybackRate(1);
    store.playReplay();
    await vi.advanceTimersByTimeAsync(50);
    expect(store.replayPlayback.value.currentStep).toBe(0);

    store.seekReplay(0);
    store.setReplayPlaybackRate(30);
    store.playReplay();
    await vi.advanceTimersByTimeAsync(50);
    expect(store.replayPlayback.value.currentStep).toBeGreaterThan(0);
  });

  it("posts stop navigation requests and tracks pending state", async () => {
    const store = createVizStore();
    let stopResolved = false;

    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "/viz/api/navigation/stop") {
        expect(init?.method).toBe("POST");
        expect(store.navigationControl.stopPending).toBe(true);
        stopResolved = true;
        return {
          ok: true,
          json: async () => ({ success: true }),
        } as Response;
      }

      throw new Error(`unexpected fetch url: ${url}`);
    }) as typeof fetch;

    const pendingRequest = store.stopNavigation();

    expect(store.navigationControl.stopPending).toBe(true);
    await pendingRequest;
    expect(stopResolved).toBe(true);
    expect(store.navigationControl.stopPending).toBe(false);
    expect(store.navigationControl.stopError).toBeNull();
  });

  it("refreshes replay tasks whenever mode switches to replay", async () => {
    const store = createVizStore();
    let requestCount = 0;

    globalThis.fetch = (async (input: RequestInfo | URL) => {
      if (String(input) !== "/viz/api/replay/tasks") {
        throw new Error(`unexpected fetch url: ${String(input)}`);
      }
      requestCount += 1;
      return {
        ok: true,
        json: async () => [],
      } as Response;
    }) as typeof fetch;

    store.setMode("replay");
    await Promise.resolve();
    expect(requestCount).toBe(1);

    store.setMode("live");
    store.setMode("replay");
    await Promise.resolve();
    expect(requestCount).toBe(2);
  });

  it("loads map labels and posts text navigation requests", async () => {
    const store = createVizStore();
    const requests: Array<{ url: string; init?: RequestInit }> = [];

    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url === "/viz/api/map/metadata") {
        return {
          ok: true,
          json: async () => ({ labels: ["办公桌", "玻璃大门门前"] }),
        } as Response;
      }
      if (url === "/viz/api/navigation/text") {
        return {
          ok: true,
          json: async () => ({
            accepted: true,
            task_id: "task-1",
            goal_text: "去玻璃大门",
            status: "processing",
            result_status: null,
            success_flag: null,
            message: "Navigation task accepted",
            state: null,
            dry_run: true,
          }),
        } as Response;
      }
      throw new Error(`unexpected fetch url: ${url}`);
    }) as typeof fetch;

    await store.loadNavigationLabels();
    expect(store.navigationControl.labels).toEqual(["办公桌", "玻璃大门门前"]);

    store.setGoalTextInput("去玻璃大门");
    store.setNavigationDryRun(true);
    expect(store.canSubmitTextNavigation.value).toBe(true);
    await store.submitTextNavigation();

    expect(requests.at(-1)?.url).toBe("/viz/api/navigation/text");
    expect(requests.at(-1)?.init?.method).toBe("POST");
    expect(requests.at(-1)?.init?.body).toBe(
      JSON.stringify({ goal_text: "去玻璃大门", dry_run: true }),
    );
    expect(store.navigationControl.textPending).toBe(false);
    expect(store.navigationControl.textError).toBeNull();
    expect(store.replay.mode).toBe("live");
    expect(store.liveSnapshot.task.status).toBe("processing");
    expect(store.canStopNavigation.value).toBe(true);
    expect(store.canSubmitTextNavigation.value).toBe(false);
  });

  it("disables text navigation submission while navigation is already running", async () => {
    const store = createVizStore();
    let requestCount = 0;

    globalThis.fetch = (async () => {
      requestCount += 1;
      return {
        ok: true,
        json: async () => ({ task_id: "task-1" }),
      } as Response;
    }) as typeof fetch;

    store.applyMessage({
      type: "task_status",
      timestamp: 2,
      schema_version: "v1",
      task_id: "task-2",
      status: "processing",
      success_flag: null,
      message: "running",
      state: null,
      goal_text: "door",
      dry_run: false,
    });
    store.setGoalTextInput("去玻璃大门");

    expect(store.canSubmitTextNavigation.value).toBe(false);
    await store.submitTextNavigation();

    expect(requestCount).toBe(0);
    expect(store.navigationControl.textError).toBe("navigation already running");
  });
});
