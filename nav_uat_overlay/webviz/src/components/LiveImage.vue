<script setup lang="ts">
import { onBeforeUnmount, shallowRef, watch } from "vue";

type ImageRef = { url: string; version: number } | undefined;

const props = defineProps<{
  image: ImageRef;
  alt: string;
}>();

const displayUrl = shallowRef<string | null>(null);
const isLoading = shallowRef(false);

let preloadImage: HTMLImageElement | null = null;
let pendingImage: ImageRef;

function resetImageState() {
  displayUrl.value = null;
  isLoading.value = false;
  preloadImage = null;
  pendingImage = undefined;
}

function loadPendingImage() {
  const nextImageRef = pendingImage;
  if (!nextImageRef?.url) {
    preloadImage = null;
    isLoading.value = false;
    return;
  }

  pendingImage = undefined;
  isLoading.value = true;
  const nextImage = new Image();
  const nextUrl = nextImageRef.url;
  preloadImage = nextImage;
  nextImage.decoding = "async";
  nextImage.onload = () => {
    if (preloadImage !== nextImage) {
      return;
    }
    displayUrl.value = nextUrl;
    preloadImage = null;
    if (pendingImage?.url && pendingImage.url !== nextUrl) {
      loadPendingImage();
      return;
    }
    isLoading.value = false;
  };
  nextImage.onerror = () => {
    if (preloadImage !== nextImage) {
      return;
    }
    preloadImage = null;
    if (pendingImage?.url && pendingImage.url !== nextUrl) {
      loadPendingImage();
      return;
    }
    isLoading.value = false;
  };
  nextImage.src = nextUrl;
}

watch(
  () => props.image?.version,
  () => {
    if (!props.image?.url) {
      resetImageState();
      return;
    }

    pendingImage = {
      url: props.image.url,
      version: props.image.version,
    };
    if (preloadImage !== null) {
      isLoading.value = true;
      return;
    }
    loadPendingImage();
  },
  { immediate: true },
);

onBeforeUnmount(() => {
  resetImageState();
});
</script>

<template>
  <div class="frame">
    <img v-if="displayUrl" :src="displayUrl" :alt="props.alt" />
    <div v-else class="placeholder">No frame yet</div>
    <div v-if="isLoading && displayUrl" class="loading-bar" />
  </div>
</template>

<style scoped>
.frame {
  position: relative;
}

img,
.placeholder {
  width: 100%;
  aspect-ratio: 4 / 3;
  border-radius: 12px;
  object-fit: cover;
  background: #10192d;
}

.placeholder {
  display: grid;
  place-items: center;
  color: #8092b3;
}

.loading-bar {
  position: absolute;
  left: 12px;
  right: 12px;
  bottom: 12px;
  height: 3px;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(111, 231, 255, 0.92), rgba(117, 240, 160, 0.92));
  opacity: 0.75;
}
</style>
