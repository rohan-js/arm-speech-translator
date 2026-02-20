package com.armfinal.translator.asr

import com.armfinal.translator.core.Language
import org.json.JSONObject
import org.vosk.Model
import org.vosk.Recognizer
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class VoskEngine(
    private val sampleRateHz: Int = 16_000,
) : Closeable {
    private var loadedLanguage: Language? = null
    private var model: Model? = null
    private var recognizer: Recognizer? = null

    fun ensureSourceLanguage(language: Language, modelsDir: File) {
        if (loadedLanguage == language && model != null) {
            return
        }
        unload()

        val relative = if (language == Language.EN) "vosk/en" else "vosk/hi"
        val modelPath = File(modelsDir, relative)
        if (!modelPath.exists()) {
            throw IllegalStateException("Missing Vosk model directory: ${modelPath.absolutePath}")
        }

        model = Model(modelPath.absolutePath)
        loadedLanguage = language
    }

    fun startUtterance() {
        val loadedModel = model ?: throw IllegalStateException("Vosk model not loaded")
        recognizer?.close()
        recognizer = Recognizer(loadedModel, sampleRateHz.toFloat())
        recognizer?.setWords(true)
    }

    fun acceptChunk(chunk: ShortArray): String {
        val activeRecognizer = recognizer ?: return ""
        val bytes = ByteArray(chunk.size * 2)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().put(chunk)

        return if (activeRecognizer.acceptWaveForm(bytes, bytes.size)) {
            extractText(activeRecognizer.result)
        } else {
            extractPartial(activeRecognizer.partialResult)
        }
    }

    fun finalResult(): String {
        val activeRecognizer = recognizer ?: return ""
        val result = extractText(activeRecognizer.finalResult)
        recognizer?.close()
        recognizer = null
        return result
    }

    fun unload() {
        recognizer?.close()
        recognizer = null
        model?.close()
        model = null
        loadedLanguage = null
    }

    override fun close() {
        unload()
    }

    private fun extractPartial(json: String): String {
        return runCatching { JSONObject(json).optString("partial", "") }
            .getOrDefault("")
            .trim()
    }

    private fun extractText(json: String): String {
        return runCatching { JSONObject(json).optString("text", "") }
            .getOrDefault("")
            .trim()
    }
}
