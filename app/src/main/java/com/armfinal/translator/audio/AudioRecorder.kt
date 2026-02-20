package com.armfinal.translator.audio

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

class AudioRecorder(
    private val sampleRateHz: Int = 16_000,
) {
    private var audioRecord: AudioRecord? = null
    private var recordJob: Job? = null
    private var isRecording: Boolean = false

    private val guard = Mutex()
    private val capturedSamples = ArrayList<Short>(sampleRateHz * 20)

    fun start(scope: CoroutineScope, onChunk: (ShortArray) -> Unit) {
        if (isRecording) {
            return
        }

        val minBuffer = AudioRecord.getMinBufferSize(
            sampleRateHz,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        check(minBuffer > 0) {
            "AudioRecord min buffer query failed: $minBuffer"
        }

        val record = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            sampleRateHz,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            minBuffer.coerceAtLeast(sampleRateHz),
        )

        record.startRecording()
        audioRecord = record
        isRecording = true

        recordJob = scope.launch(Dispatchers.IO) {
            val chunk = ShortArray(512)
            while (isActive && isRecording) {
                val read = record.read(chunk, 0, chunk.size)
                if (read <= 0) {
                    continue
                }

                val out = chunk.copyOf(read)
                guard.withLock {
                    for (i in 0 until read) {
                        capturedSamples.add(out[i])
                    }
                }
                onChunk(out)
            }
        }
    }

    suspend fun stop(): ShortArray {
        if (!isRecording) {
            return ShortArray(0)
        }

        isRecording = false

        withContext(Dispatchers.IO) {
            recordJob?.cancel()
            runCatching { recordJob?.join() }
            recordJob = null

            runCatching {
                audioRecord?.stop()
                audioRecord?.release()
            }
            audioRecord = null
        }

        return guard.withLock {
            val out = ShortArray(capturedSamples.size)
            for (i in capturedSamples.indices) {
                out[i] = capturedSamples[i]
            }
            capturedSamples.clear()
            out
        }
    }
}
