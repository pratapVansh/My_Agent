/** @type {import('next').NextConfig} */
const nextConfig = {
  // Strict Mode double-invokes effects in development to surface missing
  // cleanup. It previously had to be disabled because connectLiveKit() would
  // open two rooms; that is now guarded by an in-flight ref in ChatShell, so
  // the check is back on and doing its job.
  reactStrictMode: true
};

module.exports = nextConfig;
