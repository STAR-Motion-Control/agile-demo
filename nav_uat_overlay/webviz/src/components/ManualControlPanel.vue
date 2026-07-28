<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";

const props = defineProps<{
  pending: boolean;
  error: string | null;
  enabled: boolean;
}>();

const emit = defineEmits<{
  (event: "move", action: string): void;
  (event: "stop"): void;
}>();

const active = ref(false);
const keyActions: Record<string, string> = {
  w: "forward",
  s: "backward",
  a: "left",
  d: "right",
  q: "rotate_left",
  e: "rotate_right",
};
const keyLabels: Record<string, string> = {
  Q: "左转", W: "前进", E: "右转", A: "左移", S: "后退", D: "右移",
};

function setActive(next: boolean) {
  active.value = next;
  if (!next) emit("stop");
}

function onKeydown(event: KeyboardEvent) {
  if (!active.value || props.pending || event.repeat) return;
  const action = keyActions[event.key.toLowerCase()];
  if (!action) return;
  event.preventDefault();
  emit("move", action);
}

function deactivate() {
  if (active.value) setActive(false);
}

function onVisibilityChange() {
  if (document.hidden) deactivate();
}

onMounted(() => {
  window.addEventListener("keydown", onKeydown);
  window.addEventListener("blur", deactivate);
  document.addEventListener("visibilitychange", onVisibilityChange);
});

onUnmounted(() => {
  window.removeEventListener("keydown", onKeydown);
  window.removeEventListener("blur", deactivate);
  document.removeEventListener("visibilitychange", onVisibilityChange);
  if (active.value) emit("stop");
});
</script>

<template>
  <section class="manual-panel" :class="{ active }">
    <div class="manual-heading">
      <div>
        <p class="eyebrow">Manual Control</p>
        <h2>键盘手动控制</h2>
      </div>
      <button
        type="button"
        class="activation"
        :class="{ active }"
        :disabled="!props.enabled && !active"
        @click="setActive(!active)"
      >
        {{ active ? "停用手动控制" : "激活手动控制" }}
      </button>
    </div>

    <div class="keys" :aria-disabled="!active">
      <button v-for="key in ['Q', 'W', 'E', 'A', 'S', 'D']" :key="key" type="button"
        :disabled="!active || props.pending" @click="emit('move', keyActions[key.toLowerCase()])">
        <kbd>{{ key }}</kbd><span>{{ keyLabels[key] }}</span>
      </button>
    </div>
    <p class="hint">每次按键移动 0.2 m（横移 0.15 m）或旋转 15°。长按不会连续下发。</p>
    <p v-if="props.pending" class="status">命令执行中…</p>
    <p v-if="props.error" class="error">{{ props.error }}</p>
  </section>
</template>

<style scoped>
.manual-panel { padding: 18px; border: 1px solid rgba(110,231,255,.18); border-radius: 16px; background: rgba(17,27,49,.86); }
.manual-panel.active { border-color: #6ee7ff; box-shadow: 0 0 0 1px rgba(110,231,255,.2); }
.manual-heading { display:flex; justify-content:space-between; gap:16px; align-items:center; }
.eyebrow { margin:0 0 5px; color:#6ee7ff; font-size:11px; letter-spacing:.12em; text-transform:uppercase; }
h2 { margin:0; font-size:18px; }
.activation { padding:10px 14px; border:1px solid #526782; border-radius:10px; color:#d9e7ff; background:#1b2940; cursor:pointer; }
.activation.active { color:#081421; border-color:#6ee7ff; background:#6ee7ff; }
.activation:disabled { opacity:.45; cursor:not-allowed; }
.keys { display:grid; grid-template-columns:repeat(3, minmax(80px, 1fr)); gap:9px; margin-top:16px; }
.keys button { display:flex; align-items:center; gap:9px; padding:10px; border:1px solid rgba(255,255,255,.1); border-radius:10px; color:#c9d8ef; background:#111d30; cursor:pointer; }
.keys button:disabled { opacity:.42; cursor:not-allowed; }
kbd { display:grid; width:28px; height:28px; place-items:center; border:1px solid #58708d; border-radius:6px; color:#fff; background:#253650; font:700 13px/1 monospace; }
.hint,.status,.error { margin:11px 0 0; font-size:12px; color:#91a6c5; }
.status { color:#6ee7ff; }.error { color:#ff9c9c; }
@media (max-width: 600px) { .manual-heading { align-items:stretch; flex-direction:column; } }
</style>
