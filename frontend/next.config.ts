import path from "path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Keep Next from tracing an unrelated parent workspace on this Windows machine.
  outputFileTracingRoot: path.join(__dirname),
};

export default nextConfig;
