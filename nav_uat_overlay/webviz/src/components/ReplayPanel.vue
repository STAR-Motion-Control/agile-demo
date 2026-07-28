<script setup lang="ts">
import { computed } from "vue";

import type { ReplayTaskSummary, ViewMode } from "../types";

const props = defineProps<{
  mode: ViewMode;
  tasks: ReplayTaskSummary[];
  selectedTaskId: string | null;
  status: "idle" | "loading" | "ready" | "error";
  error: string | null;
  isPlaying: boolean;
  currentStep: number;
  totalSteps: number;
  currentMessageLabel: string;
  progressPercent: number;
  playbackRate: number;
  canStepBackward: boolean;
  canStepForward: boolean;
  hasTimeline: boolean;
  isAtEnd: boolean;
}>();

const emit = defineEmits<{
  (event: "set-mode", mode: ViewMode): void;
  (event: "refresh"): void;
  (event: "select-task", taskId: string): void;
  (event: "play"): void;
  (event: "pause"): void;
  (event: "restart"): void;
  (event: "step-backward"): void;
  (event: "step-forward"): void;
  (event: "seek", step: number): void;
  (event: "set-playback-rate", rate: number): void;
}>();

const playbackRateOptions = [1, 2, 5, 10, 20, 30];

const progressStyle = computed(() => ({
  width: `${Math.max(0, Math.min(100, props.progressPercent))}%`,
}));

const replayHint = computed(() => {
  if (props.selectedTaskId === null) {
    return "Select a task to enter replay.";
  }
  if (!props.hasTimeline) {
    return "No incremental timeline was recorded for this task.";
  }
  if (props.isPlaying) {
    return props.currentMessageLabel;
  }
  if (props.isAtEnd) {
    return `Replay complete. ${props.currentMessageLabel}`;
  }
  return props.currentMessageLabel;
});

function onSeek(event: Event) {
  const target = event.target as HTMLInputElement;
  emit("seek", Number(target.value));
}

function onPlaybackRateChange(event: Event) {
  const target = event.target as HTMLSelectElement;
  emit("set-playback-rate", Number(target.value));
}
</script>

<template>
  <section class="panel">
    <header class="panel-header">
      <h2 class="panel-title">Replay</h2>
      <button class="refresh-button" type="button" @click="emit('refresh')">Refresh</button>
    </header>

    <div class="mode-switch">
      <button
        type="button"
        class="mode-button"
        :class="{ active: props.mode === 'live' }"
        @click="emit('set-mode', 'live')"
      >
        Live
      </button>
      <button
        type="button"
        class="mode-button"
        :class="{ active: props.mode === 'replay' }"
        @click="emit('set-mode', 'replay')"
      >
        Replay
      </button>
    </div>

    <div v-if="props.mode === 'replay'" class="playback-card">
      <div class="playback-meta">
        <span class="playback-label">Timeline</span>
        <strong class="playback-step">{{ props.currentStep }} / {{ props.totalSteps }}</strong>
      </div>
      <label class="slider-block">
        <span class="slider-label">Progress</span>
        <input
          class="progress-slider"
          type="range"
          min="0"
          :max="props.totalSteps"
          :value="props.currentStep"
          :disabled="!props.hasTimeline"
          @input="onSeek"
        >
      </label>
      <div class="progress-track">
        <span class="progress-fill" :style="progressStyle" />
      </div>
      <div class="speed-row">
        <label class="speed-label" for="replay-speed">Speed</label>
        <select
          id="replay-speed"
          class="speed-select"
          :value="props.playbackRate"
          @change="onPlaybackRateChange"
        >
          <option v-for="rate in playbackRateOptions" :key="rate" :value="rate">
            {{ rate }}x
          </option>
        </select>
      </div>
      <p class="hint">{{ replayHint }}</p>

      <div class="control-grid">
        <button
          type="button"
          class="control-button"
          :disabled="!props.hasTimeline"
          @click="emit('restart')"
        >
          Restart
        </button>
        <button
          type="button"
          class="control-button"
          :disabled="!props.canStepBackward"
          @click="emit('step-backward')"
        >
          Prev
        </button>
        <button
          type="button"
          class="control-button control-button-primary"
          :disabled="!props.hasTimeline"
          @click="props.isPlaying ? emit('pause') : emit('play')"
        >
          {{ props.isPlaying ? "Pause" : "Play" }}
        </button>
        <button
          type="button"
          class="control-button"
          :disabled="!props.canStepForward"
          @click="emit('step-forward')"
        >
          Next
        </button>
      </div>
    </div>

    <p v-if="props.status === 'error'" class="error">{{ props.error }}</p>
    <p v-else-if="props.status === 'loading'" class="hint">Loading replay tasks...</p>
    <p v-else-if="props.tasks.length === 0" class="hint">No saved tasks yet.</p>

    <ul v-else class="task-list">
      <li v-for="task in props.tasks" :key="task.task_id">
        <button
          type="button"
          class="task-button"
          :class="{ selected: task.task_id === props.selectedTaskId }"
          @click="emit('select-task', task.task_id)"
        >
          <strong>{{ task.goal_text ?? task.task_id }}</strong>
          <span>{{ task.status }}</span>
        </button>
      </li>
    </ul>
  </section>
</template>

<style scoped>
.panel {
  display: grid;
  gap: 12px;
}

.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.panel-title {
  margin: 0;
}

.mode-switch {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
}

.mode-button,
.refresh-button,
.task-button,
.control-button {
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 14px;
  background: rgba(17, 27, 49, 0.86);
  color: #dfe7f5;
}

.mode-button,
.refresh-button {
  padding: 10px 12px;
}

.mode-button.active {
  border-color: rgba(111, 231, 255, 0.4);
  background: rgba(23, 44, 77, 0.96);
}

.playback-card {
  display: grid;
  gap: 10px;
  padding: 14px;
  border-radius: 16px;
  background: rgba(17, 27, 49, 0.92);
  border: 1px solid rgba(117, 240, 160, 0.14);
}

.playback-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.playback-label {
  color: #93a6c9;
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.slider-block {
  display: grid;
  gap: 8px;
}

.slider-label,
.speed-label {
  color: #93a6c9;
  font-size: 12px;
}

.progress-slider,
.speed-select {
  width: 100%;
}

.progress-slider {
  margin: 0;
  accent-color: #75f0a0;
}

.playback-step {
  font-size: 14px;
}

.progress-track {
  overflow: hidden;
  height: 8px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
}

.progress-fill {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #75f0a0, #6ee7ff);
  transition: width 180ms ease;
}

.speed-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 88px;
  gap: 10px;
  align-items: center;
}

.speed-select {
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 10px;
  padding: 8px 10px;
  background: rgba(11, 18, 31, 0.9);
  color: #dfe7f5;
}

.control-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
}

.control-button {
  padding: 10px 8px;
}

.control-button:disabled {
  opacity: 0.4;
}

.control-button-primary {
  border-color: rgba(111, 231, 255, 0.42);
  background: rgba(23, 44, 77, 0.96);
}

.task-list {
  display: grid;
  gap: 8px;
  padding: 0;
  margin: 0;
  list-style: none;
}

.task-button {
  width: 100%;
  display: grid;
  gap: 6px;
  padding: 12px 14px;
  text-align: left;
}

.task-button.selected {
  border-color: rgba(117, 240, 160, 0.45);
}

.task-button span,
.hint,
.error {
  color: #93a6c9;
  font-size: 12px;
}

.error {
  color: #ff9a9a;
}
</style>
