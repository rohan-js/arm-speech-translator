package com.armfinal.translator.nativebridge

object NativeBridge {
    init {
        System.loadLibrary("speech_translation_native")
    }

    external fun init(modelsDir: String): Boolean
    external fun setThermalMode(mode: Int)
    external fun onTrimMemory(level: Int)
    external fun vadTrimAndSplit(pcmF32: FloatArray, sampleRate: Int, splitGapMs: Int): IntArray
    external fun whisperTranscribeOnce(pcmF32: FloatArray, sampleRate: Int, lang: Int): String
    external fun nllbEnsureLoaded(mode: Int): Boolean
    external fun nllbTranslate(text: String, srcLang: Int, tgtLang: Int, mode: Int): String
    external fun piperUnloadVoice()
    external fun piperSynthesize(voice: Int, text: String): ByteArray
    external fun runtimeStats(): String
}
