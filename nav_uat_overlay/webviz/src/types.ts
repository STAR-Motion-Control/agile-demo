export type VizEvent = {
  type: "event";
  timestamp: number;
  schema_version: string;
  event_name: string;
  level: string;
  task_id: string | null;
  payload: Record<string, unknown>;
};

export type ImageRef = {
  url: string;
  version: number;
  updated_at?: number;
};

export type MapMetadata = {
  frame_id: string;
  resolution: number;
  origin: number[];
  width: number;
  height: number;
  image_url: string | null;
  image_path?: string;
  only_global_planner_area?: number[][][];
  labels?: string[];
};

export type ViewMode = "live" | "replay";

export type ReplayTaskSummary = {
  task_id: string;
  goal_text: string | null;
  status: string;
  started_at: number;
  ended_at: number;
  has_snapshot: boolean;
  event_count: number;
};

export type ReplayTaskDetail = ReplayTaskSummary;

export type SnapshotPayload = {
  type: "snapshot";
  timestamp: number;
  schema_version: string;
  task: {
    task_id: string | null;
    goal_text: string | null;
    status: string;
    result_status?: string | null;
    success_flag: boolean | null;
    message: string | null;
    state: number | string | null;
    dry_run: boolean;
  };
  goal: {
    pose: { x: number; y: number; theta: number } | null;
    source: string | null;
  };
  robot: {
    pose: { x: number; y: number; theta: number } | null;
    vpr_pose: { x: number; y: number; theta: number } | null;
  };
  planner: {
    mode: string | null;
    global_path: number[][];
    waypoints: number[][];
    local_path: number[][];
    local_goal: number[] | null;
    actions: number[][];
    action_limit: number | null;
    preview_actions: number[][];
    camera_intrinsics: number[][] | null;
    camera_image_size: number[] | null;
  };
  images: Record<string, ImageRef | undefined>;
  events: VizEvent[];
};

export type HelloPayload = {
  type: "hello";
  timestamp: number;
  schema_version: string;
  capabilities: Record<string, boolean>;
};

export type TaskStatusPayload = {
  type: "task_status";
  timestamp: number;
  schema_version: string;
  task_id: string | null;
  status: string;
  result_status?: string | null;
  success_flag: boolean | null;
  message: string | null;
  state: number | string | null;
  goal_text: string | null;
  dry_run: boolean;
};

export type PoseUpdatePayload = {
  type: "pose_update";
  timestamp: number;
  schema_version: string;
  pose: { x: number; y: number; theta: number } | null;
  vpr_pose: { x: number; y: number; theta: number } | null;
};

export type GoalUpdatePayload = {
  type: "goal_update";
  timestamp: number;
  schema_version: string;
  goal: {
    pose: { x: number; y: number; theta: number } | null;
    source: string | null;
  };
};

export type PlannerUpdatePayload = {
  type: "planner_update";
  timestamp: number;
  schema_version: string;
  mode: string | null;
  global_path: number[][];
  waypoints: number[][];
  local_path: number[][];
  local_goal: number[] | null;
  actions: number[][];
  action_limit?: number | null;
  preview_actions?: number[][];
  camera_intrinsics?: number[][] | null;
  camera_image_size?: number[] | null;
};

export type ImageUpdatePayload = {
  type: "image_update";
  timestamp: number;
  schema_version: string;
  images: Record<string, ImageRef | undefined>;
};

export type ReplayTimelineMessage =
  | TaskStatusPayload
  | PoseUpdatePayload
  | GoalUpdatePayload
  | PlannerUpdatePayload
  | ImageUpdatePayload
  | VizEvent;

export type VizMessage =
  | HelloPayload
  | SnapshotPayload
  | TaskStatusPayload
  | PoseUpdatePayload
  | GoalUpdatePayload
  | PlannerUpdatePayload
  | ImageUpdatePayload
  | VizEvent;
