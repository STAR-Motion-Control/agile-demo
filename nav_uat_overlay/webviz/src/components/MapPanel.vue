<script setup lang="ts">
import { computed } from "vue";

import { useMapMetadata } from "../composables/useMapMetadata";

type Pose = { x: number; y: number; theta: number };

const props = defineProps<{
  mode: string | null;
  robotPose: Pose | null;
  vprPose: Pose | null;
  goalPose: Pose | null;
  globalPath: number[][];
  localPath: number[][];
  localGoal: number[] | null;
  executedActionCount: number;
  actionLimit: number | null;
}>();

const { state: metadataState } = useMapMetadata();

const imageUrl = computed(() => metadataState.data?.image_url ?? null);
const viewBox = computed(() => {
  const width = metadataState.data?.width ?? 100;
  const height = metadataState.data?.height ?? 100;
  return `0 0 ${width} ${height}`;
});

function worldToPixel(point: { x: number; y: number } | number[] | null) {
  const metadata = metadataState.data;
  if (!metadata || point === null) {
    return null;
  }

  const x = Array.isArray(point) ? point[0] : point.x;
  const y = Array.isArray(point) ? point[1] : point.y;
  const originX = metadata.origin[0] ?? 0;
  const originY = metadata.origin[1] ?? 0;

  return {
    x: (x - originX) / metadata.resolution,
    y: metadata.height - (y - originY) / metadata.resolution,
  };
}

function toPixelPoints(points: number[][]) {
  return points
    .map((point) => worldToPixel(point))
    .filter((point): point is { x: number; y: number } => point !== null);
}

function toSvgPoints(points: { x: number; y: number }[]) {
  return points.map((point) => `${point.x},${point.y}`).join(" ");
}

const robotPixel = computed(() => worldToPixel(props.robotPose));
const vprPixel = computed(() => worldToPixel(props.vprPose));
const goalPixel = computed(() => worldToPixel(props.goalPose));
const localGoalPixel = computed(() => worldToPixel(props.localGoal));
const navdpPreview = computed(() => {
  const points = toPixelPoints(props.localPath);
  const actionCount = Math.max(props.localPath.length - 1, 0);
  const visible = props.mode === "navdp" && points.length > 1;

  if (!visible) {
    return {
      actionCount,
      endPoint: null,
      polyline: "",
      tooltip: null,
      visible,
    };
  }

  const tooltip =
    props.actionLimit !== null
      ? `NavDP preview shows ${actionCount} actions. Actual execution only uses the first ${props.executedActionCount} actions (limit: ${props.actionLimit}).`
      : `NavDP preview shows ${actionCount} actions. Actual execution uses ${props.executedActionCount} actions.`;

  return {
    actionCount,
    endPoint: points.at(-1) ?? null,
    polyline: toSvgPoints(points),
    tooltip,
    visible,
  };
});
const polygonPoints = computed(() =>
  (metadataState.data?.only_global_planner_area ?? [])
    .map((polygon) => toSvgPoints(toPixelPoints(polygon)))
    .filter((points) => points.length > 0),
);
const pathPolyline = computed(() =>
  toSvgPoints(toPixelPoints(props.globalPath)),
);

function buildHeading(pixel: { x: number; y: number } | null, pose: Pose | null, length = 12) {
  if (!pixel || !pose) {
    return null;
  }

  const tipX = pixel.x + Math.cos(pose.theta) * length;
  const tipY = pixel.y - Math.sin(pose.theta) * length;
  const arrowLength = 4.5;
  const wingAngle = Math.PI / 6;

  return {
    x1: pixel.x,
    y1: pixel.y,
    x2: tipX,
    y2: tipY,
    tipPoints: [
      `${tipX},${tipY}`,
      `${tipX - Math.cos(pose.theta - wingAngle) * arrowLength},${tipY + Math.sin(pose.theta - wingAngle) * arrowLength}`,
      `${tipX - Math.cos(pose.theta + wingAngle) * arrowLength},${tipY + Math.sin(pose.theta + wingAngle) * arrowLength}`,
    ].join(" "),
  };
}

const robotHeading = computed(() => buildHeading(robotPixel.value, props.robotPose, 12));
const goalHeading = computed(() => buildHeading(goalPixel.value, props.goalPose, 14));
</script>

<template>
  <section class="panel">
    <header class="panel-header">
      <h2>Map Panel</h2>
      <span class="mode-chip" :class="props.mode ? `mode-${props.mode}` : 'mode-none'">
        Mode: {{ props.mode ?? "n/a" }}
      </span>
    </header>
    <div class="map">
      <div class="symbol-legend">
        <div class="legend-item">
          <span class="legend-swatch robot-swatch" />
          <span>Robot</span>
        </div>
        <div class="legend-item">
          <span class="legend-heading">
            <span class="legend-heading-outline" />
            <span class="legend-heading-line" />
          </span>
          <span>Robot Heading</span>
        </div>
        <div class="legend-item">
          <span class="legend-swatch vpr-swatch" />
          <span>VPR Pose</span>
        </div>
        <div class="legend-item">
          <span class="legend-swatch goal-swatch" />
          <span>Goal</span>
        </div>
        <div class="legend-item">
          <span class="legend-heading">
            <span class="legend-heading-outline goal-heading-outline-swatch" />
            <span class="legend-heading-line goal-heading-line-swatch" />
          </span>
          <span>Goal Heading</span>
        </div>
        <div class="legend-item">
          <span class="legend-swatch local-goal-swatch" />
          <span>Local Goal</span>
        </div>
        <div v-if="navdpPreview.visible" class="legend-item">
          <span class="legend-line navdp-path-swatch" />
          <span>NavDP Preview</span>
        </div>
        <div class="legend-item">
          <span class="legend-line path-swatch" />
          <span>Global Path</span>
        </div>
      </div>

      <div v-if="metadataState.status === 'ready' && imageUrl" class="stage">
        <img class="map-image" :src="imageUrl" alt="navigation map" />
        <svg class="overlay" :viewBox="viewBox" preserveAspectRatio="xMidYMid meet">
          <polyline
            v-if="pathPolyline"
            class="path-line"
            :points="pathPolyline"
          />
          <polygon
            v-for="(points, index) in polygonPoints"
            :key="`global-area-${index}`"
            class="global-area"
            :points="points"
          />
          <polyline
            v-if="navdpPreview.visible"
            class="navdp-path-line"
            :points="navdpPreview.polyline"
          >
            <title v-if="navdpPreview.tooltip">{{ navdpPreview.tooltip }}</title>
          </polyline>
          <circle
            v-if="navdpPreview.endPoint"
            class="navdp-path-end"
            :cx="navdpPreview.endPoint.x"
            :cy="navdpPreview.endPoint.y"
            r="3.5"
          >
            <title v-if="navdpPreview.tooltip">{{ navdpPreview.tooltip }}</title>
          </circle>
          <circle
            v-if="localGoalPixel"
            class="local-goal"
            :cx="localGoalPixel.x"
            :cy="localGoalPixel.y"
            r="4"
          />
          <circle
            v-if="goalPixel"
            class="goal"
            :cx="goalPixel.x"
            :cy="goalPixel.y"
            r="5"
          />
          <line
            v-if="goalHeading"
            class="goal-heading-outline"
            :x1="goalHeading.x1"
            :y1="goalHeading.y1"
            :x2="goalHeading.x2"
            :y2="goalHeading.y2"
          />
          <line
            v-if="goalHeading"
            class="goal-heading"
            :x1="goalHeading.x1"
            :y1="goalHeading.y1"
            :x2="goalHeading.x2"
            :y2="goalHeading.y2"
          />
          <polygon
            v-if="goalHeading"
            class="goal-heading-tip"
            :points="goalHeading.tipPoints"
          />
          <circle
            v-if="vprPixel"
            class="vpr"
            :cx="vprPixel.x"
            :cy="vprPixel.y"
            r="4"
          />
          <line
            v-if="robotHeading"
            class="heading-outline"
            :x1="robotHeading.x1"
            :y1="robotHeading.y1"
            :x2="robotHeading.x2"
            :y2="robotHeading.y2"
          />
          <line
            v-if="robotHeading"
            class="heading"
            :x1="robotHeading.x1"
            :y1="robotHeading.y1"
            :x2="robotHeading.x2"
            :y2="robotHeading.y2"
          />
          <polygon
            v-if="robotHeading"
            class="heading-tip"
            :points="robotHeading.tipPoints"
          />
          <circle
            v-if="robotPixel"
            class="robot"
            :cx="robotPixel.x"
            :cy="robotPixel.y"
            r="5"
          />
        </svg>
      </div>
      <div v-else-if="metadataState.status === 'error'" class="placeholder">
        Failed to load map metadata: {{ metadataState.error }}
      </div>
      <div v-else class="placeholder">
        Loading map metadata...
      </div>

      <div class="legend">
        <span>Robot: {{ props.robotPose ? `${props.robotPose.x.toFixed(2)}, ${props.robotPose.y.toFixed(2)}` : "n/a" }}</span>
        <span>VPR: {{ props.vprPose ? `${props.vprPose.x.toFixed(2)}, ${props.vprPose.y.toFixed(2)}` : "n/a" }}</span>
        <span>Goal: {{ props.goalPose ? `${props.goalPose.x.toFixed(2)}, ${props.goalPose.y.toFixed(2)}, ${props.goalPose.theta.toFixed(2)}rad` : "n/a" }}</span>
        <span>Mode: {{ props.mode ?? "n/a" }}</span>
        <span>Path points: {{ props.globalPath.length }}</span>
        <span v-if="props.mode === 'navdp'">Preview actions: {{ navdpPreview.actionCount }}</span>
        <span v-if="props.mode === 'navdp'">Executed actions: {{ props.executedActionCount }}</span>
      </div>
    </div>
  </section>
</template>

<style scoped>
.panel {
  display: grid;
  gap: 14px;
}

.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.mode-chip {
  display: inline-flex;
  align-items: center;
  min-height: 32px;
  padding: 0 12px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  white-space: nowrap;
}

.mode-line_segment {
  background: rgba(117, 240, 160, 0.12);
  color: #b9f8cd;
}

.mode-navdp {
  background: rgba(110, 231, 255, 0.12);
  color: #aef5ff;
}

.mode-none {
  background: rgba(255, 255, 255, 0.06);
  color: #c9d4ea;
}

.map {
  display: grid;
  gap: 14px;
  border-radius: 18px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  padding: 18px;
  background:
    radial-gradient(circle at top left, rgba(117, 240, 160, 0.08), transparent 32%),
    rgba(10, 17, 29, 0.94);
}

.stage {
  position: relative;
  overflow: hidden;
  border-radius: 14px;
  background: #050a14;
  min-height: 420px;
}

.map-image,
.overlay {
  display: block;
  width: 100%;
  height: 100%;
}

.map-image {
  object-fit: contain;
}

.overlay {
  position: absolute;
  inset: 0;
}

.global-area {
  fill: rgba(111, 231, 255, 0.08);
  stroke: rgba(111, 231, 255, 0.42);
  stroke-width: 2;
}

.placeholder {
  display: grid;
  place-items: center;
  min-height: 420px;
  border-radius: 14px;
  background:
    linear-gradient(rgba(255, 255, 255, 0.06) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.06) 1px, transparent 1px),
    #10192d;
  background-size: 32px 32px;
  color: #93a6c9;
}

.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  color: #dfe7f5;
}

.symbol-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 16px;
  padding: 10px 12px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.04);
}

.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: #dfe7f5;
  font-size: 12px;
}

.legend-swatch {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  border: 1.5px solid rgba(10, 17, 29, 0.9);
}

.legend-line {
  width: 18px;
  height: 0;
  border-top: 3px solid #75f0a0;
  border-radius: 999px;
}

.legend-heading {
  position: relative;
  display: inline-flex;
  align-items: center;
  width: 18px;
  height: 10px;
}

.legend-heading-outline,
.legend-heading-line {
  position: absolute;
  left: 0;
  right: 0;
  border-radius: 999px;
}

.legend-heading-outline {
  border-top: 3px solid rgba(10, 17, 29, 0.95);
}

.legend-heading-line {
  border-top: 1.5px solid #6ee7ff;
}

.goal-heading-outline-swatch {
  border-top-color: rgba(10, 17, 29, 0.95);
}

.goal-heading-line-swatch {
  border-top-color: #ffb087;
}

.robot-swatch {
  background: #f6f8fb;
}

.vpr-swatch {
  background: #63c9ff;
}

.goal-swatch {
  background: #ff8a5b;
}

.local-goal-swatch {
  background: #ffe066;
}

.navdp-path-swatch {
  border-top-width: 2px;
  border-top-color: #ff6b6b;
}

.legend span {
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.06);
  font-size: 12px;
}

@media (max-width: 900px) {
  .stage,
  .placeholder {
    min-height: 320px;
  }
}

.path-line {
  fill: none;
  stroke: #75f0a0;
  stroke-width: 3;
  stroke-linecap: round;
  stroke-linejoin: round;
  opacity: 0.95;
}

.navdp-path-line {
  fill: none;
  stroke: #ff6b6b;
  stroke-width: 2.8;
  stroke-linecap: round;
  stroke-linejoin: round;
  opacity: 1;
}

.navdp-path-end {
  fill: #ff6b6b;
  stroke: rgba(10, 17, 29, 0.9);
  stroke-width: 1.2;
}

.local-goal {
  fill: #ffe066;
  stroke: rgba(10, 17, 29, 0.9);
  stroke-width: 1.5;
}

.goal {
  fill: #ff8a5b;
  stroke: rgba(10, 17, 29, 0.9);
  stroke-width: 1.5;
}

.vpr {
  fill: #63c9ff;
  stroke: rgba(10, 17, 29, 0.9);
  stroke-width: 1.5;
  opacity: 0.85;
}

.robot {
  fill: #f6f8fb;
  stroke: rgba(10, 17, 29, 0.9);
  stroke-width: 1.5;
}

.heading-outline {
  stroke: rgba(10, 17, 29, 0.95);
  stroke-width: 3.5;
  stroke-linecap: round;
}

.heading {
  stroke: #6ee7ff;
  stroke-width: 1.6;
  stroke-linecap: round;
}

.heading-tip {
  fill: #6ee7ff;
}

.goal-heading-outline {
  stroke: rgba(10, 17, 29, 0.95);
  stroke-width: 3.5;
  stroke-linecap: round;
}

.goal-heading {
  stroke: #ffb087;
  stroke-width: 1.8;
  stroke-linecap: round;
}

.goal-heading-tip {
  fill: #ffb087;
}
</style>
