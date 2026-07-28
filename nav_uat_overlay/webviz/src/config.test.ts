import { describe, expect, it } from "vitest";

import viteConfig from "../vite.config";

describe("vite config", () => {
  it("uses /viz/ as the build base path", () => {
    expect(viteConfig.base).toBe("/viz/");
  });
});
