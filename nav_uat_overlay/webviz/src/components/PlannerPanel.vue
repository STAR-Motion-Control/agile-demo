<script setup lang="ts">
const props = defineProps<{
  mode: string | null;
  localGoal: number[] | null;
  actions: number[][];
}>();
</script>

<template>
  <section class="panel">
    <header>
      <h2>Planner</h2>
    </header>
    <div class="card">
      <div class="row"><span>Mode</span><strong>{{ props.mode ?? "n/a" }}</strong></div>
      <div class="row"><span>Local goal</span><strong>{{ props.localGoal ? props.localGoal.join(", ") : "n/a" }}</strong></div>
      <div class="row"><span>Actions</span><strong>{{ props.actions.length }}</strong></div>
    </div>
    <ol v-if="props.actions.length > 0" class="actions">
      <li v-for="(action, index) in props.actions" :key="index">
        rotate={{ action[0] ?? 0 }}, forward={{ action[1] ?? 0 }}
      </li>
    </ol>
  </section>
</template>

<style scoped>
.panel {
  display: grid;
  gap: 12px;
}

.card,
.actions {
  margin: 0;
  padding: 14px 16px;
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(17, 27, 49, 0.86);
}

.row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
}

.row:last-child {
  margin-bottom: 0;
}

.row span {
  color: #93a6c9;
}

.actions {
  padding-left: 32px;
  max-height: 180px;
  overflow: auto;
}
</style>
