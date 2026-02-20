package com.armfinal.translator.audio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack

class TtsAudioPlayer {
    private var track: AudioTrack? = null
    private var activeSampleRate: Int = -1

    fun playChunk(pcm16: ByteArray, sampleRateHz: Int) {
        if (pcm16.isEmpty()) {
            return
        }

        ensureTrack(sampleRateHz)
        val activeTrack = track ?: return
        activeTrack.play()
        activeTrack.write(pcm16, 0, pcm16.size, AudioTrack.WRITE_BLOCKING)
    }

    fun stopAndRelease() {
        runCatching {
            track?.pause()
            track?.flush()
            track?.stop()
            track?.release()
        }
        track = null
        activeSampleRate = -1
    }

    private fun ensureTrack(sampleRateHz: Int) {
        if (track != null && activeSampleRate == sampleRateHz) {
            return
        }
        stopAndRelease()

        val queriedMinBuffer = AudioTrack.getMinBufferSize(
            sampleRateHz,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        val baseBuffer = if (queriedMinBuffer > 0) queriedMinBuffer else sampleRateHz
        val minBuffer = alignToFrame(baseBuffer.coerceAtLeast(sampleRateHz / 2))

        track = AudioTrack(
            AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build(),
            AudioFormat.Builder()
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .setSampleRate(sampleRateHz)
                .build(),
            minBuffer,
            AudioTrack.MODE_STREAM,
            AudioManager.AUDIO_SESSION_ID_GENERATE,
        )
        activeSampleRate = sampleRateHz
    }

    private fun alignToFrame(sizeBytes: Int): Int {
        val frameBytes = 2 // mono PCM16
        val remainder = sizeBytes % frameBytes
        return if (remainder == 0) sizeBytes else sizeBytes + (frameBytes - remainder)
    }
}
