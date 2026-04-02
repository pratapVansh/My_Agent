/**
 * AudioWorklet processor — replaces the deprecated ScriptProcessorNode.
 * Runs on the dedicated audio rendering thread for glitch-free capture.
 *
 * Buffers 4096 samples before posting to the main thread so downstream
 * behaviour (downsampling, WebSocket send cadence) matches the old
 * ScriptProcessorNode(4096) setup.
 */
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Float32Array(4096);
    this._pos = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;

    let i = 0;
    while (i < channel.length) {
      const space = 4096 - this._pos;
      const toCopy = Math.min(space, channel.length - i);
      this._buf.set(channel.subarray(i, i + toCopy), this._pos);
      this._pos += toCopy;
      i += toCopy;

      if (this._pos === 4096) {
        // postMessage copies the typed array — no transfer needed
        this.port.postMessage(this._buf.slice());
        this._pos = 0;
      }
    }
    return true;
  }
}

registerProcessor("pcm-processor", PCMProcessor);
