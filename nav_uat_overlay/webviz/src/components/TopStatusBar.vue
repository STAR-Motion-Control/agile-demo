<script setup lang="ts">
import { computed, onMounted, onUnmounted, shallowRef, useTemplateRef } from "vue";

const props = defineProps<{
  connectionStatus: string;
  schemaVersion: string | null;
  taskId: string | null;
  taskStatus: string;
  goalText: string | null;
  mode: "live" | "replay";
  canStopNavigation: boolean;
  stopPending: boolean;
  stopError: string | null;
  navigationLabels: string[];
  navigationGoalText: string;
  navigationDryRun: boolean;
  navigationPending: boolean;
  canSubmitNavigation: boolean;
  navigationError: string | null;
  currentTaskDryRun: boolean;
}>();

const emit = defineEmits<{
  (event: "stop-navigation"): void;
  (event: "update-goal-text", value: string): void;
  (event: "update-dry-run", value: boolean): void;
  (event: "submit-navigation"): void;
}>();

function onGoalInput(event: Event) {
  const target = event.target as HTMLInputElement;
  isLabelMenuOpen.value = true;
  emit("update-goal-text", target.value);
}

const navComboboxRef = useTemplateRef<HTMLDivElement>("navCombobox");
const navInputRef = useTemplateRef<HTMLInputElement>("navInput");
const isLabelMenuOpen = shallowRef(false);

const filteredNavigationLabels = computed(() => {
  const query = props.navigationGoalText.trim().toLocaleLowerCase();
  if (!query) {
    return props.navigationLabels;
  }

  return props.navigationLabels.filter((label) =>
    label.toLocaleLowerCase().includes(query),
  );
});

const showLabelMenu = computed(
  () => isLabelMenuOpen.value && filteredNavigationLabels.value.length > 0,
);
const submitButtonLabel = computed(() => {
  if (props.navigationPending) {
    return "Starting...";
  }
  if (props.navigationDryRun) {
    return "Start Dry Run";
  }
  return "Start Nav";
});

function openLabelMenu() {
  if (props.navigationLabels.length === 0) {
    return;
  }
  isLabelMenuOpen.value = true;
}

function closeLabelMenu() {
  isLabelMenuOpen.value = false;
}

function selectNavigationLabel(label: string) {
  emit("update-goal-text", label);
  closeLabelMenu();
  navInputRef.value?.focus();
}

function onGoalKeydown(event: KeyboardEvent) {
  if (event.key === "Escape") {
    closeLabelMenu();
    return;
  }

  if (event.key === "ArrowDown") {
    event.preventDefault();
    openLabelMenu();
    return;
  }

  if (event.key === "Enter" && props.canSubmitNavigation) {
    event.preventDefault();
    closeLabelMenu();
    emit("submit-navigation");
  }
}

function onDryRunChange(event: Event) {
  const target = event.target as HTMLInputElement;
  emit("update-dry-run", target.checked);
}

function handleDocumentPointerDown(event: PointerEvent) {
  const target = event.target;
  if (!(target instanceof Node)) {
    return;
  }

  if (navComboboxRef.value?.contains(target)) {
    return;
  }

  closeLabelMenu();
}

onMounted(() => {
  document.addEventListener("pointerdown", handleDocumentPointerDown);
});

onUnmounted(() => {
  document.removeEventListener("pointerdown", handleDocumentPointerDown);
});
</script>

<template>
  <section class="status-bar">
    <div class="item">
      <span class="label">Connection</span>
      <strong>{{ props.connectionStatus }}</strong>
    </div>
    <div class="item">
      <span class="label">Schema</span>
      <strong>{{ props.schemaVersion ?? "n/a" }}</strong>
    </div>
    <div class="item">
      <span class="label">Task</span>
      <strong>{{ props.taskId ?? "none" }}</strong>
    </div>
    <div class="item">
      <span class="label">Status</span>
      <strong>{{ props.taskStatus }}</strong>
    </div>
    <div class="item">
      <span class="label">Mode</span>
      <strong>{{ props.mode }}</strong>
    </div>
    <div class="item">
      <span class="label">Run Type</span>
      <strong>{{ props.currentTaskDryRun ? "dry-run" : "live" }}</strong>
    </div>
    <div class="item wide">
      <span class="label">Goal</span>
      <strong>{{ props.goalText ?? "no active goal" }}</strong>
    </div>
    <div class="item nav-card">
      <span class="label">Text Nav</span>
      <div class="nav-controls">
        <div ref="navCombobox" class="nav-combobox">
          <input
            ref="navInput"
            class="nav-input"
            type="text"
            placeholder="输入或选择目标地点"
            :value="props.navigationGoalText"
            @focus="openLabelMenu"
            @click="openLabelMenu"
            @input="onGoalInput"
            @keydown="onGoalKeydown"
          >
          <div v-if="showLabelMenu" class="nav-options" role="listbox" aria-label="Navigation labels">
            <button
              v-for="label in filteredNavigationLabels"
              :key="label"
              type="button"
              class="nav-option"
              @mousedown.prevent="selectNavigationLabel(label)"
            >
              {{ label }}
            </button>
          </div>
        </div>
        <button
          type="button"
          class="nav-button"
          :disabled="!props.canSubmitNavigation"
          @click="emit('submit-navigation')"
        >
          {{ submitButtonLabel }}
        </button>
      </div>
      <label class="dry-run-toggle">
        <input
          class="dry-run-checkbox"
          type="checkbox"
          :checked="props.navigationDryRun"
          :disabled="props.navigationPending"
          @change="onDryRunChange"
        >
        <span>只生成路径，不下发机器人运动</span>
      </label>
      <span v-if="props.navigationError" class="nav-error">{{ props.navigationError }}</span>
    </div>
    <div class="item stop-card">
      <span class="label">Control</span>
      <button
        type="button"
        class="stop-button"
        :disabled="!props.canStopNavigation || props.stopPending"
        @click="emit('stop-navigation')"
      >
        {{ props.stopPending ? "Stopping..." : "Stop Nav" }}
      </button>
      <span v-if="props.stopError" class="stop-error">{{ props.stopError }}</span>
    </div>
  </section>
</template>

<style scoped>
.status-bar {
  display: grid;
  grid-template-columns: repeat(10, minmax(0, 1fr));
  gap: 12px;
}

.item {
  padding: 14px 16px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 14px;
  background: rgba(17, 27, 49, 0.86);
}

.wide {
  grid-column: span 2;
}

.nav-card {
  grid-column: span 2;
  display: grid;
  gap: 8px;
}

.stop-card {
  display: grid;
  gap: 8px;
}

.nav-controls {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 8px;
  align-items: start;
}

.dry-run-toggle {
  display: flex;
  gap: 8px;
  align-items: center;
  color: #c4d3ed;
  font-size: 12px;
}

.dry-run-checkbox {
  accent-color: #6ee7ff;
}

.nav-combobox {
  position: relative;
}

.label {
  display: block;
  margin-bottom: 6px;
  font-size: 12px;
  color: #90a4c7;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

strong {
  font-size: 15px;
}

.nav-input,
.nav-button,
.stop-button {
  border: 1px solid rgba(255, 111, 111, 0.35);
  border-radius: 12px;
  padding: 10px 12px;
}

.nav-input {
  border-color: rgba(255, 255, 255, 0.08);
  background: rgba(10, 17, 29, 0.9);
  color: #dfe7f5;
  caret-color: #dfe7f5;
  -webkit-text-fill-color: #dfe7f5;
  opacity: 1;
}

.nav-input::placeholder {
  color: #7f93b7;
}

.nav-input:focus {
  outline: 1px solid rgba(110, 231, 255, 0.35);
  outline-offset: 0;
}

.nav-input:-webkit-autofill,
.nav-input:-webkit-autofill:hover,
.nav-input:-webkit-autofill:focus,
.nav-input:-webkit-autofill:active {
  -webkit-text-fill-color: #dfe7f5;
  caret-color: #dfe7f5;
  box-shadow: 0 0 0 1000px rgba(10, 17, 29, 0.9) inset;
  transition: background-color 9999s ease-out 0s;
}

.nav-options {
  position: absolute;
  top: calc(100% + 6px);
  left: 0;
  right: 0;
  z-index: 20;
  display: grid;
  gap: 4px;
  max-height: 220px;
  padding: 6px;
  overflow-y: auto;
  border: 1px solid rgba(111, 231, 255, 0.14);
  border-radius: 14px;
  background: rgba(7, 13, 25, 0.98);
  box-shadow: 0 16px 40px rgba(0, 0, 0, 0.34);
}

.nav-option {
  border: 0;
  border-radius: 10px;
  padding: 10px 12px;
  background: transparent;
  color: #dfe7f5;
  text-align: left;
  cursor: pointer;
}

.nav-option:hover,
.nav-option:focus-visible {
  background: rgba(110, 231, 255, 0.12);
  color: #f4fbff;
  outline: none;
}

.nav-button {
  border-color: rgba(111, 231, 255, 0.28);
  background: rgba(23, 44, 77, 0.96);
  color: #dff6ff;
}

.nav-button:disabled,
.stop-button:disabled {
  opacity: 0.45;
}

.stop-button {
  border-radius: 12px;
  background: linear-gradient(180deg, rgba(156, 24, 24, 0.98), rgba(109, 10, 10, 0.96));
  color: #ffe5e5;
}

.nav-error,
.stop-error {
  color: #ff9a9a;
  font-size: 12px;
}

@media (max-width: 1200px) {
  .status-bar {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .wide,
  .nav-card {
    grid-column: span 2;
  }
}

@media (max-width: 720px) {
  .status-bar {
    grid-template-columns: 1fr;
  }

  .wide,
  .nav-card {
    grid-column: span 1;
  }

  .nav-controls {
    grid-template-columns: 1fr;
  }
}
</style>
