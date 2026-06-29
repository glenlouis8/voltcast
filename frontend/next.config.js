/** @type {import('next').NextConfig} */
// Minimal config. The app is a static dashboard that fetches forecast JSON
// from S3 at runtime, so no special build settings are needed.
const nextConfig = {
  reactStrictMode: true,
};

module.exports = nextConfig;
