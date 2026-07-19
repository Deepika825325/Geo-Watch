import type { NextConfig } from "next"

const rawBackendOrigin =
  process.env.GEOWATCH_API_ORIGIN?.trim()

const explicitBackendOrigin =
  rawBackendOrigin
    ? (
        /^[a-z][a-z\d+\-.]*:\/\//i.test(
          rawBackendOrigin
        )
          ? rawBackendOrigin
          : `https://${rawBackendOrigin}`
      )
    : undefined

const backendHostPort =
  process.env.GEOWATCH_API_HOSTPORT?.trim()

const backendOrigin =
  explicitBackendOrigin
  || (
    backendHostPort
      ? `http://${backendHostPort}`
      : "http://127.0.0.1:8007"
  )

const nextConfig: NextConfig = {
  output: "standalone",

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
