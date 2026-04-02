/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false  // Strict Mode causes double-mount in dev, which opens two WebSocket connections
};

module.exports = nextConfig;
