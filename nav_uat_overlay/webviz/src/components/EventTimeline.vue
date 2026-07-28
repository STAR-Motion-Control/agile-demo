<script setup lang="ts">
import type { VizEvent } from "../types";

const props = defineProps<{
  events: VizEvent[];
  mode: "live" | "replay";
}>();
</script>

<template>
  <section class="panel">
    <header>
      <h2>{{ props.mode === "replay" ? "Replay Events" : "Recent Events" }}</h2>
    </header>
    <div class="timeline">
      <article v-for="(event, index) in props.events.slice(-12).reverse()" :key="`${event.event_name}-${index}`" class="event">
        <div class="event-head">
          <strong>{{ event.event_name }}</strong>
          <span>{{ event.level }}</span>
        </div>
        <code>{{ JSON.stringify(event.payload) }}</code>
      </article>
      <div v-if="props.events.length === 0" class="empty">No events yet</div>
    </div>
  </section>
</template>

<style scoped>
.panel {
  display: grid;
  gap: 12px;
}

.timeline {
  display: grid;
  gap: 10px;
  max-height: 480px;
  overflow: auto;
  padding-right: 4px;
}

.event,
.empty {
  padding: 14px 16px;
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(17, 27, 49, 0.86);
}

.event-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
}

.event span {
  color: #93a6c9;
  text-transform: uppercase;
  font-size: 11px;
  letter-spacing: 0.08em;
}

.event code {
  display: block;
  white-space: pre-wrap;
  color: #dce4f5;
  font-size: 12px;
  line-height: 1.5;
}

.empty {
  color: #93a6c9;
}
</style>
