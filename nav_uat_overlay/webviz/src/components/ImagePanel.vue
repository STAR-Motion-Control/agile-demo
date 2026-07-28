<script setup lang="ts">
import LiveImage from "./LiveImage.vue";
import NavdpProjectedImage from "./NavdpProjectedImage.vue";

type ImageRef = { url: string; version: number; updated_at?: number } | undefined;

const props = defineProps<{
  latestImage: ImageRef;
  navdpImage: ImageRef;
  previewActions: number[][];
  cameraIntrinsics: number[][] | null;
  cameraImageSize: number[] | null;
  mode: "live" | "replay";
}>();
</script>

<template>
  <section class="panel">
    <header>
      <h2>Vision Panel</h2>
    </header>
    <div class="hero-card">
      <div class="card-header">
        <h3>Latest RGB</h3>
        <span class="tag">{{ props.mode }}</span>
      </div>
      <LiveImage
        class="primary-frame"
        :image="props.latestImage"
        alt="Latest RGB"
      />
      <div class="meta">
        <span>Source: rgb_latest</span>
        <span>Version: {{ props.latestImage?.version ?? "n/a" }}</span>
        <span>
          Updated:
          {{ props.latestImage?.updated_at ? new Date(props.latestImage.updated_at * 1000).toLocaleTimeString() : "n/a" }}
        </span>
      </div>
    </div>
    <div class="hero-card">
      <div class="card-header">
        <h3>navdp input</h3>
        <span class="tag tag-navdp">projected</span>
      </div>
      <NavdpProjectedImage
        class="primary-frame"
        :image="props.navdpImage"
        :actions="props.previewActions"
        :camera-intrinsics="props.cameraIntrinsics"
        :camera-image-size="props.cameraImageSize"
        alt="navdp input"
      />
      <div class="meta">
        <span>Source: rgb_navdp</span>
        <span>Version: {{ props.navdpImage?.version ?? "n/a" }}</span>
        <span>
          Updated:
          {{ props.navdpImage?.updated_at ? new Date(props.navdpImage.updated_at * 1000).toLocaleTimeString() : "n/a" }}
        </span>
        <span>Preview actions: {{ props.previewActions.length }}</span>
        <span>Intrinsics: {{ props.cameraIntrinsics ? "ready" : "n/a" }}</span>
      </div>
    </div>
  </section>
</template>

<style scoped>
.panel {
  display: grid;
  gap: 14px;
}

.hero-card {
  border-radius: 18px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background:
    radial-gradient(circle at top left, rgba(111, 231, 255, 0.12), transparent 42%),
    rgba(17, 27, 49, 0.9);
  padding: 16px;
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 12px;
}

.hero-card h3 {
  margin: 0;
  font-size: 14px;
}

.primary-frame {
  aspect-ratio: 4 / 3;
}

.tag {
  padding: 4px 8px;
  border-radius: 999px;
  background: rgba(117, 240, 160, 0.12);
  color: #9fe870;
  font-size: 12px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.tag-navdp {
  background: rgba(111, 231, 255, 0.12);
  color: #6ee7ff;
}

.meta {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 12px;
  color: #93a6c9;
  font-size: 12px;
}
</style>
