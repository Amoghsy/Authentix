/**
 * compressor.js — Client-side video compression using FFmpeg WASM
 *
 * Uses @ffmpeg/ffmpeg (WebAssembly) to compress videos in the browser
 * before uploading. This prevents OOM issues on free-tier backends.
 *
 * Requirements:
 *   - The page must be served with COOP + COEP headers (for SharedArrayBuffer)
 *   - See vite.config.js for dev-server header configuration
 */

import { FFmpeg } from '@ffmpeg/ffmpeg'
import { fetchFile, toBlobURL } from '@ffmpeg/util'

// Singleton FFmpeg instance — loaded once, reused
let ffmpegInstance = null
let loadPromise = null

/**
 * Loads the FFmpeg WASM core (singleton).
 * Uses unpkg CDN for the WASM binary so no local copy needed.
 */
async function getFFmpeg() {
  if (ffmpegInstance) return ffmpegInstance

  if (loadPromise) return loadPromise

  loadPromise = (async () => {
    const ff = new FFmpeg()

    // Use CDN-hosted WASM to avoid bundling the large binary
    const baseURL = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/esm'

    await ff.load({
      coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
      wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
    })

    ffmpegInstance = ff
    return ff
  })()

  return loadPromise
}

/**
 * Terminates the FFmpeg instance (frees WASM memory).
 * Call this if you want a clean teardown.
 */
export function terminateFFmpeg() {
  if (ffmpegInstance) {
    try { ffmpegInstance.terminate() } catch { /* ignore */ }
    ffmpegInstance = null
    loadPromise = null
  }
}

/**
 * Compresses a video File using FFmpeg WASM.
 *
 * @param {File}     file          - The original video file
 * @param {Object}   options
 * @param {number}   options.targetSizeMB  - Target output size in MB (default: 30)
 * @param {number}   options.maxWidth      - Max video width (default: 1280)
 * @param {number}   options.crf           - Constant Rate Factor: lower = better quality (default: 28)
 * @param {number}   options.audioBitrate  - Audio bitrate kbps (default: 64)
 * @param {Function} options.onProgress    - Progress callback (0–100)
 * @param {Function} options.onLog         - FFmpeg log callback
 * @returns {Promise<File>} Compressed File object
 */
export async function compressVideo(file, {
  targetSizeMB = 30,
  maxWidth = 1280,
  crf = 28,
  audioBitrate = 64,
  onProgress = null,
  onLog = null,
} = {}) {
  const ff = await getFFmpeg()

  // Wire progress and log callbacks
  const progressHandler = ({ progress }) => {
    if (onProgress) onProgress(Math.round(progress * 100))
  }
  const logHandler = ({ message }) => {
    if (onLog) onLog(message)
  }

  ff.on('progress', progressHandler)
  ff.on('log', logHandler)

  const inputName = 'input_' + Date.now() + getExtension(file.name)
  const outputName = 'output_' + Date.now() + '.mp4'

  try {
    // Write input file to FFmpeg's virtual filesystem
    await ff.writeFile(inputName, await fetchFile(file))

    // Build FFmpeg args:
    // -vf scale: scale down to maxWidth, keeping aspect ratio
    // -c:v libx264: H.264 codec (universally supported)
    // -crf: quality factor
    // -preset fast: speed/quality tradeoff
    // -movflags +faststart: progressive download optimisation
    // -c:a aac: AAC audio
    // -b:a: audio bitrate
    const args = [
      '-i', inputName,
      '-vf', `scale='min(${maxWidth},iw)':-2`,
      '-c:v', 'libx264',
      '-crf', String(crf),
      '-preset', 'fast',
      '-movflags', '+faststart',
      '-c:a', 'aac',
      '-b:a', `${audioBitrate}k`,
      '-y', // overwrite if exists
      outputName,
    ]

    await ff.exec(args)

    // Read the compressed output
    const data = await ff.readFile(outputName)
    const blob = new Blob([data.buffer], { type: 'video/mp4' })

    // Clean up virtual FS
    try {
      await ff.deleteFile(inputName)
      await ff.deleteFile(outputName)
    } catch { /* ignore cleanup errors */ }

    const compressedFile = new File(
      [blob],
      file.name.replace(/\.[^.]+$/, '_compressed.mp4'),
      { type: 'video/mp4' }
    )

    return compressedFile
  } finally {
    ff.off('progress', progressHandler)
    ff.off('log', logHandler)
  }
}

function getExtension(filename) {
  const match = filename.match(/\.[^.]+$/)
  return match ? match[0] : '.mp4'
}

/**
 * Checks if compression is worth doing:
 * Returns true if file is larger than thresholdMB.
 */
export function shouldCompress(file, thresholdMB = 50) {
  return file.size > thresholdMB * 1024 * 1024
}

/**
 * Returns a human-readable file size string.
 */
export function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
