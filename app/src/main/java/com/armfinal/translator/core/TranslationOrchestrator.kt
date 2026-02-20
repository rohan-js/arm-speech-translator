package com.armfinal.translator.core

import android.content.ComponentCallbacks2
import android.content.Context
import com.armfinal.translator.asr.VoskEngine
import com.armfinal.translator.audio.AudioRecorder
import com.armfinal.translator.audio.TtsAudioPlayer
import com.armfinal.translator.nativebridge.NativeBridge
import com.armfinal.translator.storage.AssetModelManager
import com.armfinal.translator.storage.getExternalNllbDir
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.cancel
import kotlinx.coroutines.coroutineScope
import android.util.Log
import java.io.File

data class PipelineResult(
    val transcript: String,
    val translation: String,
    val runtimeStats: String,
    val latencyMs: Long,
)

class TranslationOrchestrator(
    private val context: Context,
) {
    companion object {
        private val REQUIRED_NLLB_FILES = listOf(
            "en_hi/encoder_model_int8.onnx",
            "en_hi/decoder_model_int8.onnx",
            "en_hi/decoder_with_past_model_int8.onnx",
            "en_hi/source.spm",
            "en_hi/target.spm",
            "en_hi/vocab.json",
            "en_hi/nllb_config.json",
            "hi_en/encoder_model_int8.onnx",
            "hi_en/decoder_model_int8.onnx",
            "hi_en/decoder_with_past_model_int8.onnx",
            "hi_en/source.spm",
            "hi_en/target.spm",
            "hi_en/vocab.json",
            "hi_en/nllb_config.json",
        )
    }

    private val modelManager = AssetModelManager(context)
    private val recorder = AudioRecorder(sampleRateHz = 16_000)
    private val vosk = VoskEngine(sampleRateHz = 16_000)
    private val player = TtsAudioPlayer()
    private val memLogger = MemThermalLogger(context)
    private val ioScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private lateinit var modelsDir: File
    private var initialized = false
    private var activeDirection = LanguageDirection.EN_TO_HI
    private var thermalMode = ThermalMode.NORMAL
    private var lastRawPartial = ""
    private var lastEmittedPartial = ""
    private var stableHitCount = 0

    suspend fun initialize() {
        modelsDir = modelManager.ensureExtracted()

        val externalDir = getExternalNllbDir(context)
        check(externalDir.exists() && externalDir.isDirectory) {
            "External NLLB directory missing: ${externalDir.absolutePath}"
        }
        val missing = REQUIRED_NLLB_FILES.filter { name ->
            val file = File(externalDir, name)
            !file.exists() || !file.isFile || file.length() <= 0L
        }
        check(missing.isEmpty()) { "Missing external NLLB files: ${missing.joinToString()}" }

        // Native code reads this file to locate external NLLB dir.
        File(modelsDir, "nllb_external_dir.txt").writeText(externalDir.absolutePath)

        val ok = NativeBridge.init(modelsDir.absolutePath)
        check(ok) { "Native init failed" }
        initialized = true
    }

    fun setThermalMode(mode: ThermalMode) {
        thermalMode = mode
        NativeBridge.setThermalMode(mode.nativeValue)
        if (mode == ThermalMode.CRITICAL) {
            NativeBridge.piperUnloadVoice()
        }
    }

    fun setDirection(direction: LanguageDirection) {
        ensureInitialized()
        if (activeDirection == direction) {
            return
        }

        activeDirection = direction
        NativeBridge.piperUnloadVoice()
        vosk.unload()
    }

    fun startRecording(onPartial: (String) -> Unit) {
        ensureInitialized()
        resetPartialFilter()
        vosk.ensureSourceLanguage(activeDirection.source, modelsDir)
        vosk.startUtterance()
        recorder.start(scope = ioScope) { chunk ->
            val partial = vosk.acceptChunk(chunk)
            filterAndEmitPartial(partial, onPartial)
        }
    }

    suspend fun stopRecording(mode: ThermalMode): PipelineResult = coroutineScope {
        ensureInitialized()
        val startTime = System.currentTimeMillis()

        val rawAudio = recorder.stop()
        val finalVoskTranscript = vosk.finalResult()
        if (rawAudio.isEmpty()) {
            val runtime = NativeBridge.runtimeStats()
            return@coroutineScope PipelineResult(
                transcript = finalVoskTranscript,
                translation = "",
                runtimeStats = runtime,
                latencyMs = 0,
            )
        }

        val pcmF32 = PcmUtils.shortToFloat(rawAudio)
        val splitBounds = NativeBridge.vadTrimAndSplit(pcmF32, 16_000, 700)
        val segments = PcmUtils.segmentsFromBounds(pcmF32, splitBounds)

        val warmup = if (mode.useNllb) {
            async(Dispatchers.Default) {
                val prewarmText = if (activeDirection.source == Language.EN) "hello" else "नमस्ते"
                runCatching {
                    NativeBridge.nllbTranslate(
                        text = prewarmText,
                        srcLang = activeDirection.source.nativeId,
                        tgtLang = activeDirection.target.nativeId,
                        mode = mode.nativeValue,
                    )
                }
            }
        } else {
            null
        }

        val transcript = if (mode.useWhisper) {
            val whisperText = segments
                .map {
                    NativeBridge.whisperTranscribeOnce(
                        pcmF32 = it,
                        sampleRate = 16_000,
                        lang = activeDirection.source.nativeId,
                    )
                }
                .joinToString(separator = " ")
                .trim()
            if (whisperText.isBlank()) finalVoskTranscript else whisperText
        } else {
            finalVoskTranscript
        }

        warmup?.await()

        val translation = if (transcript.isBlank()) {
            ""
        } else {
            NativeBridge.nllbTranslate(
                text = transcript,
                srcLang = activeDirection.source.nativeId,
                tgtLang = activeDirection.target.nativeId,
                mode = mode.nativeValue,
            )
        }

        val clauses = PcmUtils.splitClauses(translation)
        val voice = if (activeDirection.target == Language.EN) Voice.EN else Voice.HI
        var ttsFailed = false
        for (clause in clauses) {
            if (ttsFailed) {
                continue
            }
            runCatching {
                val pcm = NativeBridge.piperSynthesize(voice.nativeId, clause)
                player.playChunk(pcm, voice.sampleRate)
            }.onFailure { err ->
                Log.w("SpeechTranslation", "TTS playback skipped: ${err.message}")
                ttsFailed = true
            }
        }

        val latency = System.currentTimeMillis() - startTime
        val runtime = NativeBridge.runtimeStats()
        memLogger.log("utterance_complete", mode, runtime, latency)

        return@coroutineScope PipelineResult(
            transcript = transcript,
            translation = translation,
            runtimeStats = runtime,
            latencyMs = latency,
        )
    }

    fun onTrimMemory(level: Int) {
        NativeBridge.onTrimMemory(level)
        if (level >= ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL) {
            player.stopAndRelease()
        }
        if (level >= ComponentCallbacks2.TRIM_MEMORY_COMPLETE) {
            vosk.unload()
        }
    }

    fun onAppBackground() {
        NativeBridge.onTrimMemory(ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL)
        NativeBridge.piperUnloadVoice()
        player.stopAndRelease()
    }

    fun close() {
        player.stopAndRelease()
        vosk.close()
        NativeBridge.piperUnloadVoice()
        ioScope.coroutineContext.cancel()
    }

    private fun ensureInitialized() {
        check(initialized) { "Orchestrator not initialized" }
    }

    private fun resetPartialFilter() {
        lastRawPartial = ""
        lastEmittedPartial = ""
        stableHitCount = 0
    }

    private fun wordCount(text: String): Int {
        return text.trim()
            .split(Regex("\\s+"))
            .count { it.isNotBlank() }
    }

    private fun filterAndEmitPartial(raw: String, onPartial: (String) -> Unit) {
        val partial = raw.trim()
        if (partial.isBlank()) return

        val words = wordCount(partial)
        if (words < 2) {
            lastRawPartial = partial
            return
        }

        stableHitCount = if (partial == lastRawPartial) stableHitCount + 1 else 0
        lastRawPartial = partial

        if (words < 3 && stableHitCount < 1) {
            return
        }

        if (lastEmittedPartial.isNotBlank()) {
            if (partial == lastEmittedPartial) return
            if (partial.length + 2 < lastEmittedPartial.length) return
            if (!partial.startsWith(lastEmittedPartial) && words < 4) return
        }

        lastEmittedPartial = partial
        onPartial(partial)
    }
}
