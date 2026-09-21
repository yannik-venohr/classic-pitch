// Microphone capture for the live app, as an AudioWorklet.
//
// All it does is batch the render quantum (128 samples, ~6ms) up into
// blocks big enough to be worth a message, and post them to the main
// thread. Everything else — the rolling context window, when to ask the
// server for a prediction — lives in live.js, which is easier to read
// and debug on the main thread and costs nothing here: this processor
// must not block the audio thread, and copying a few thousand floats
// per block is the most it should ever do.

// ~93ms at 22050 Hz. Small enough that the level meter still feels
// immediate, large enough to keep the message rate around 10/s.
const BLOCK_SAMPLES = 2048;

class RecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.block = new Float32Array(BLOCK_SAMPLES);
    this.filled = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    // An input with no connected source yields an empty array rather
    // than silence; returning true keeps the processor alive until the
    // node is disconnected from the main thread.
    if (!channel) return true;

    for (let i = 0; i < channel.length; i++) {
      this.block[this.filled++] = channel[i];
      if (this.filled === BLOCK_SAMPLES) {
        // A copy, not the buffer itself: this.block is reused for the
        // next block immediately after posting.
        this.port.postMessage(this.block.slice());
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor('recorder', RecorderProcessor);
