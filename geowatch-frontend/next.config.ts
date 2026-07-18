import type { NextConfig } from "next"

const backendOrigin =
  process.env.GEOWATCH_API_ORIGIN
  ?? "http://127.0.0.1:8007"

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/backend/:path*",
        destination: `${backendOrigin}/:path*`,
      },
    ]
  },
}

export default nextConfig
