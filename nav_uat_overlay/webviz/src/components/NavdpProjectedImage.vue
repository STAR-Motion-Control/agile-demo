<script setup lang="ts">
import { computed, onBeforeUnmount, shallowRef, watch } from "vue";

type ImageRef = { url: string; version: number } | undefined;
type PixelPoint = { x: number; y: number };
type OffscreenDirection = "left" | "right" | "behind" | null;
type TrajectorySample = {
  localX: number;
  projected: PixelPoint | null;
};

const CAMERA_HEIGHT_METERS = 0.2;

const props = defineProps<{
  image: ImageRef;
  actions: number[][];
  cameraIntrinsics: number[][] | null;
  cameraImageSize: number[] | null;
  alt: string;
}>();

const displayUrl = shallowRef<string | null>(null);
const isLoading = shallowRef(false);
const imageWidth = shallowRef(320);
const imageHeight = shallowRef(240);

let preloadImage: HTMLImageElement | null = null;
let pendingImage: ImageRef;

function resetImageState() {
  displayUrl.value = null;
  isLoading.value = false;
  preloadImage = null;
  pendingImage = undefined;
}

function updateLoadedImageDimensions(image: HTMLImageElement) {
  if (image.naturalWidth > 0) {
    imageWidth.value = image.naturalWidth;
  }
  if (image.naturalHeight > 0) {
    imageHeight.value = image.naturalHeight;
  }
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
    updateLoadedImageDimensions(nextImage);
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

function normalizeAngle(angle: number) {
  return ((angle + Math.PI) % (2 * Math.PI)) - Math.PI;
}

const scaledIntrinsics = computed(() => {
  if (
    props.cameraIntrinsics === null ||
    props.cameraIntrinsics.length !== 3 ||
    props.cameraImageSize === null ||
    props.cameraImageSize.length !== 2
  ) {
    return null;
  }

  const sourceWidth = Number(props.cameraImageSize[0] ?? Number.NaN);
  const sourceHeight = Number(props.cameraImageSize[1] ?? Number.NaN);
  if (
    !Number.isFinite(sourceWidth) ||
    !Number.isFinite(sourceHeight) ||
    sourceWidth <= 0 ||
    sourceHeight <= 0
  ) {
    return null;
  }

  const scaleX = imageWidth.value / sourceWidth;
  const scaleY = imageHeight.value / sourceHeight;
  const fx = Number(props.cameraIntrinsics[0]?.[0] ?? Number.NaN) * scaleX;
  const fy = Number(props.cameraIntrinsics[1]?.[1] ?? Number.NaN) * scaleY;
  const cx = Number(props.cameraIntrinsics[0]?.[2] ?? Number.NaN) * scaleX;
  const cy = Number(props.cameraIntrinsics[1]?.[2] ?? Number.NaN) * scaleY;

  if (![fx, fy, cx, cy].every((value) => Number.isFinite(value))) {
    return null;
  }

  return { fx, fy, cx, cy };
});

const trajectorySamples = computed<TrajectorySample[]>(() => {
  if (scaledIntrinsics.value === null || props.actions.length === 0) {
    return [];
  }

  const { fx, fy, cx, cy } = scaledIntrinsics.value;
  const samples: TrajectorySample[] = [];
  let x = 0;
  let y = 0;
  let theta = 0;

  for (const action of props.actions) {
    const deltaTheta = Number(action?.[0] ?? 0);
    const distance = Number(action?.[1] ?? 0);
    theta = normalizeAngle(theta + deltaTheta);
    x += distance * Math.cos(theta);
    y += distance * Math.sin(theta);

    if (x <= 1e-4) {
      samples.push({ localX: x, projected: null });
      continue;
    }

    const projectedX = (fx * -y) / x + cx;
    const projectedY = imageHeight.value - 1 + (fy * CAMERA_HEIGHT_METERS) / x - cy;
    if (!Number.isFinite(projectedX) || !Number.isFinite(projectedY)) {
      samples.push({ localX: x, projected: null });
      continue;
    }

    samples.push({
      localX: x,
      projected: { x: projectedX, y: projectedY },
    });
  }

  return samples;
});

const projectedPoints = computed<PixelPoint[]>(() =>
  trajectorySamples.value
    .map((sample) => sample.projected)
    .filter((point): point is PixelPoint => point !== null),
);

const hasForwardSamples = computed(() =>
  trajectorySamples.value.some((sample) => sample.localX > 1e-4),
);

const lastTrajectorySample = computed(() =>
  trajectorySamples.value.length > 0 ? trajectorySamples.value.at(-1) ?? null : null,
);

const trajectoryFullyBehind = computed(() => {
  if (trajectorySamples.value.length === 0) {
    return false;
  }
  if (!hasForwardSamples.value) {
    return true;
  }
  return projectedPoints.value.length === 0 && (lastTrajectorySample.value?.localX ?? 1) <= 1e-4;
});

const visibleProjectedPoints = computed(() =>
  projectedPoints.value.filter(
    (point) =>
      point.x >= 0 &&
      point.x <= imageWidth.value &&
      point.y >= 0 &&
      point.y <= imageHeight.value,
  ),
);

const projectedPolyline = computed(() =>
  projectedPoints.value.map((point) => `${point.x},${point.y}`).join(" "),
);

const projectedEndPoint = computed(() =>
  visibleProjectedPoints.value.length > 0 ? visibleProjectedPoints.value.at(-1) ?? null : null,
);

const offscreenDirection = computed<OffscreenDirection>(() => {
  if (projectedPoints.value.length === 0 || visibleProjectedPoints.value.length > 0) {
    return trajectoryFullyBehind.value ? "behind" : null;
  }
  if (projectedPoints.value.every((point) => point.x < 0)) {
    return "left";
  }
  if (projectedPoints.value.every((point) => point.x > imageWidth.value)) {
    return "right";
  }
  if ((lastTrajectorySample.value?.localX ?? 1) <= 1e-4) {
    return "behind";
  }
  return null;
});

const arrowPolygon = computed(() => {
  const direction = offscreenDirection.value;
  if (direction === null) {
    return null;
  }

  const width = imageWidth.value;
  const height = imageHeight.value;
  const tipInset = 4;
  const shaftLength = 42;
  const arrowHalfSpan = 22;
  const shaftHalfThickness = 9;

  if (direction === "left") {
    const centerY = Math.round(height / 2);
    const tipX = tipInset;
    const innerX = tipX + shaftLength;
    return [
      `${tipX},${centerY}`,
      `${innerX},${centerY - arrowHalfSpan}`,
      `${innerX},${centerY - shaftHalfThickness}`,
      `${width * 0.18},${centerY - shaftHalfThickness}`,
      `${width * 0.18},${centerY + shaftHalfThickness}`,
      `${innerX},${centerY + shaftHalfThickness}`,
      `${innerX},${centerY + arrowHalfSpan}`,
    ].join(" ");
  }

  if (direction === "right") {
    const centerY = Math.round(height / 2);
    const tipX = width - tipInset;
    const innerX = tipX - shaftLength;
    return [
      `${tipX},${centerY}`,
      `${innerX},${centerY - arrowHalfSpan}`,
      `${innerX},${centerY - shaftHalfThickness}`,
      `${width * 0.82},${centerY - shaftHalfThickness}`,
      `${width * 0.82},${centerY + shaftHalfThickness}`,
      `${innerX},${centerY + shaftHalfThickness}`,
      `${innerX},${centerY + arrowHalfSpan}`,
    ].join(" ");
  }

  const centerX = Math.round(width / 2);
  const tipY = height - tipInset;
  const innerY = tipY - shaftLength;
  return [
    `${centerX},${tipY}`,
    `${centerX - arrowHalfSpan},${innerY}`,
    `${centerX - shaftHalfThickness},${innerY}`,
    `${centerX - shaftHalfThickness},${height * 0.72}`,
    `${centerX + shaftHalfThickness},${height * 0.72}`,
    `${centerX + shaftHalfThickness},${innerY}`,
    `${centerX + arrowHalfSpan},${innerY}`,
  ].join(" ");
});

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
    <img v-if="displayUrl" class="image" :src="displayUrl" :alt="props.alt" />
    <div v-else class="placeholder">No frame yet</div>
    <svg
      v-if="displayUrl && (projectedPoints.length > 0 || arrowPolygon)"
      class="overlay"
      :viewBox="`0 0 ${imageWidth} ${imageHeight}`"
      preserveAspectRatio="none"
    >
      <polyline
        v-if="projectedPoints.length > 0"
        class="path-outline"
        :points="projectedPolyline"
      />
      <polyline
        v-if="projectedPoints.length > 0"
        class="path-line"
        :points="projectedPolyline"
      />
      <circle
        v-if="projectedEndPoint"
        class="path-end"
        :cx="projectedEndPoint.x"
        :cy="projectedEndPoint.y"
        r="4"
      />
      <polygon
        v-if="arrowPolygon"
        class="offscreen-arrow-outline"
        :points="arrowPolygon"
      />
      <polygon
        v-if="arrowPolygon"
        class="offscreen-arrow"
        :points="arrowPolygon"
      />
    </svg>
    <div v-if="isLoading && displayUrl" class="loading-bar" />
  </div>
</template>

<style scoped>
.frame {
  position: relative;
}

.image,
.placeholder,
.overlay {
  width: 100%;
  aspect-ratio: 4 / 3;
  border-radius: 12px;
}

.image {
  object-fit: cover;
  background: #10192d;
}

.placeholder {
  display: grid;
  place-items: center;
  background: #10192d;
  color: #8092b3;
}

.overlay {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

.path-outline {
  fill: none;
  stroke: rgba(10, 17, 29, 0.92);
  stroke-linecap: round;
  stroke-linejoin: round;
  stroke-width: 6;
}

.path-line {
  fill: none;
  stroke: #6ee7ff;
  stroke-linecap: round;
  stroke-linejoin: round;
  stroke-width: 3;
}

.path-end {
  fill: #9fe870;
  stroke: rgba(10, 17, 29, 0.92);
  stroke-width: 2;
}

.offscreen-arrow-outline {
  fill: rgba(10, 17, 29, 0.92);
}

.offscreen-arrow {
  fill: #ff8c42;
  filter: drop-shadow(0 0 10px rgba(255, 140, 66, 0.55));
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
