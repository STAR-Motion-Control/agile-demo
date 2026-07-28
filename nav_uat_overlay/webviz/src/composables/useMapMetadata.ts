import { onMounted, reactive } from "vue";

import type { MapMetadata } from "../types";

type MapMetadataState = {
  data: MapMetadata | null;
  status: "idle" | "loading" | "ready" | "error";
  error: string | null;
};

export function useMapMetadata() {
  const state = reactive<MapMetadataState>({
    data: null,
    status: "idle",
    error: null,
  });

  async function load() {
    state.status = "loading";
    state.error = null;

    try {
      const response = await fetch("/viz/api/map/metadata");
      if (!response.ok) {
        throw new Error(`metadata request failed: ${response.status}`);
      }

      state.data = (await response.json()) as MapMetadata;
      state.status = "ready";
    } catch (error) {
      state.status = "error";
      state.error = error instanceof Error ? error.message : "unknown error";
    }
  }

  onMounted(() => {
    void load();
  });

  return {
    state,
    load,
  };
}
