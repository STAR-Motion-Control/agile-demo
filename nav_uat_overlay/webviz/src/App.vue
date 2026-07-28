<script setup lang="ts">
import { computed, onMounted, onUnmounted } from "vue";

import EventTimeline from "./components/EventTimeline.vue";
import ImagePanel from "./components/ImagePanel.vue";
import MapPanel from "./components/MapPanel.vue";
import ManualControlPanel from "./components/ManualControlPanel.vue";
import PlannerPanel from "./components/PlannerPanel.vue";
import ReplayPanel from "./components/ReplayPanel.vue";
import TopStatusBar from "./components/TopStatusBar.vue";
import { createVizStore } from "./stores/vizStore";

const store = createVizStore();

onMounted(() => {
  store.connect();
  void store.loadReplayTasks();
  void store.loadNavigationLabels();
});

onUnmounted(() => {
  store.dispose();
});

const snapshot = computed(() => store.currentSnapshot.value);

const summary = computed(() => ({
  schemaVersion: store.connection.schemaVersion,
  status: snapshot.value.task.status,
  taskId: snapshot.value.task.task_id,
  goalText: snapshot.value.task.goal_text,
}));
</script>

<template>
  <main class="shell">
    <section class="hero">
      <p class="eyebrow">Navigation WebViz</p>
      <h1 class="hero-title">实时导航可视化控制台</h1>
    </section>

    <TopStatusBar
      :connection-status="store.connection.status"
      :schema-version="summary.schemaVersion"
      :task-id="summary.taskId"
      :task-status="summary.status"
      :goal-text="summary.goalText"
      :mode="store.replay.mode"
      :can-stop-navigation="store.canStopNavigation.value"
      :stop-pending="store.navigationControl.stopPending"
      :stop-error="store.navigationControl.stopError"
      :navigation-labels="store.navigationControl.labels"
      :navigation-goal-text="store.navigationControl.goalTextInput"
      :navigation-dry-run="store.navigationControl.dryRun"
      :navigation-pending="store.navigationControl.textPending"
      :can-submit-navigation="store.canSubmitTextNavigation.value"
      :navigation-error="store.navigationControl.textError"
      :current-task-dry-run="snapshot.task.dry_run"
      @stop-navigation="store.stopNavigation"
      @update-goal-text="store.setGoalTextInput"
      @update-dry-run="store.setNavigationDryRun"
      @submit-navigation="store.submitTextNavigation"
    />

    <section class="layout">
      <div class="main-column">
        <ManualControlPanel
          :pending="store.navigationControl.manualPending"
          :error="store.navigationControl.manualError"
          :enabled="store.replay.mode === 'live' && !store.canStopNavigation.value"
          @move="store.sendManualControl"
          @stop="store.stopNavigation"
        />
        <section class="visual-grid">
          <MapPanel
            :mode="snapshot.planner.mode"
            :robot-pose="snapshot.robot.pose"
            :vpr-pose="snapshot.robot.vpr_pose"
            :goal-pose="snapshot.goal.pose"
            :global-path="snapshot.planner.global_path"
            :local-path="snapshot.planner.local_path"
            :local-goal="snapshot.planner.local_goal"
            :executed-action-count="snapshot.planner.actions.length"
            :action-limit="snapshot.planner.action_limit"
          />
          <ImagePanel
            :latest-image="snapshot.images.rgb_latest"
            :navdp-image="snapshot.images.rgb_navdp"
            :preview-actions="snapshot.planner.preview_actions"
            :camera-intrinsics="snapshot.planner.camera_intrinsics"
            :camera-image-size="snapshot.planner.camera_image_size"
            :mode="store.replay.mode"
          />
        </section>
      </div>
      <aside class="side-column">
        <ReplayPanel
          :mode="store.replay.mode"
          :tasks="store.replay.tasks"
          :selected-task-id="store.replay.selectedTaskId"
          :status="store.replay.status"
          :error="store.replay.error"
          :is-playing="store.replay.isPlaying"
          :current-step="store.replayPlayback.value.currentStep"
          :total-steps="store.replayPlayback.value.totalSteps"
          :current-message-label="store.replayPlayback.value.currentMessageLabel"
          :progress-percent="store.replayPlayback.value.progressPercent"
          :playback-rate="store.replayPlayback.value.playbackRate"
          :can-step-backward="store.replayPlayback.value.canStepBackward"
          :can-step-forward="store.replayPlayback.value.canStepForward"
          :has-timeline="store.replayPlayback.value.hasTimeline"
          :is-at-end="store.replayPlayback.value.isAtEnd"
          @set-mode="store.setMode"
          @refresh="store.loadReplayTasks"
          @select-task="store.selectReplayTask"
          @play="store.playReplay"
          @pause="store.pauseReplay"
          @restart="store.restartReplay"
          @seek="store.seekReplay"
          @set-playback-rate="store.setReplayPlaybackRate"
          @step-backward="store.stepReplayBackward"
          @step-forward="store.stepReplayForward"
        />
        <PlannerPanel
          :mode="snapshot.planner.mode"
          :local-goal="snapshot.planner.local_goal"
          :actions="snapshot.planner.actions"
        />
        <EventTimeline :events="snapshot.events" :mode="store.replay.mode" />
      </aside>
    </section>
  </main>
</template>

<style scoped>
.shell {
  max-width: 1680px;
  margin: 0 auto;
  padding: 24px;
}

.hero {
  margin-bottom: 18px;
}

.eyebrow {
  margin: 0 0 8px;
  color: #6ee7ff;
  font-size: 12px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.hero-title {
  margin: 0 0 12px;
  font-size: clamp(28px, 4vw, 42px);
  line-height: 1.05;
}

.subtitle {
  max-width: 840px;
  margin: 0;
  color: #9db0d0;
  line-height: 1.6;
}

.layout {
  display: grid;
  grid-template-columns: minmax(0, 1.7fr) 360px;
  gap: 18px;
  align-items: start;
}

.main-column,
.side-column {
  display: grid;
  gap: 18px;
}

.visual-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) minmax(360px, 0.8fr);
  gap: 18px;
  align-items: start;
}

.side-column {
  position: sticky;
  top: 16px;
}

@media (max-width: 1200px) {
  .layout {
    grid-template-columns: 1fr;
  }

  .visual-grid {
    grid-template-columns: 1fr;
  }

  .side-column {
    position: static;
  }
}

@media (max-width: 720px) {
  .shell {
    padding: 16px;
  }
}
</style>
