import type { Metadata } from "next"
import type { ReactNode } from "react"
import {
  IBM_Plex_Mono,
  Inter,
  Space_Grotesk,
} from "next/font/google"
import "leaflet/dist/leaflet.css"
import "./globals.css"

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  weight: ["500", "600"],
  variable: "--font-space-grotesk",
  display: "swap",
})

const inter = Inter({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-inter",
  display: "swap",
})

const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-ibm-plex-mono",
  display: "swap",
})

export const metadata: Metadata = {
  title: "GeoWatch — Change Detection Console",
  description:
    "Bi-temporal Sentinel-2 satellite change detection and geospatial analysis",
}

interface RootLayoutProps {
  children: ReactNode
}

export default function RootLayout({
  children,
}: RootLayoutProps) {
  return (
    <html
      lang="en"
      className={`${spaceGrotesk.variable} ${inter.variable} ${plexMono.variable}`}
    >
      <body className="bg-bg font-body text-ink antialiased">
        {children}
      </body>
    </html>
  )
}
