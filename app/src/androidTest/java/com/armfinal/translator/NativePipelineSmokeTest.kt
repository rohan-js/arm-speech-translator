package com.armfinal.translator

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.armfinal.translator.nativebridge.NativeBridge
import com.armfinal.translator.storage.AssetModelManager
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class NativePipelineSmokeTest {
    @Before
    fun setup() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val modelsDir = runBlocking { AssetModelManager(context).ensureExtracted() }
        val ok = NativeBridge.init(modelsDir.absolutePath)
        assertTrue(ok)
    }

    @Test
    fun criticalModeUsesPhraseFallback() {
        val output = NativeBridge.nllbTranslate(
            text = "hello",
            srcLang = 0,
            tgtLang = 1,
            mode = 3,
        )
        assertEquals("नमस्ते", output)
    }

    @Test
    fun throttledModeCapsMaxTokensAt48() {
        val input = (1..120).joinToString(" ") { "word$it" }
        NativeBridge.nllbTranslate(
            text = input,
            srcLang = 0,
            tgtLang = 1,
            mode = 1,
        )
        val stats = NativeBridge.runtimeStats()
        assertTrue(stats.contains("nllb_max_new_tokens=48"))
    }

    @Test
    fun vadSplitsOnlyWhenGapExceedsThreshold() {
        val sr = 16_000
        val frame = FloatArray(sr * 3)

        fillSpeech(frame, sr * 0 / 10, sr * 3 / 10)
        fillSpeech(frame, sr * 8 / 10, sr * 11 / 10)
        fillSpeech(frame, sr * 20 / 10, sr * 23 / 10)

        val bounds = NativeBridge.vadTrimAndSplit(frame, sr, 700)
        assertEquals(4, bounds.size)
    }

    private fun fillSpeech(buf: FloatArray, start: Int, end: Int) {
        val s = start.coerceAtLeast(0)
        val e = end.coerceAtMost(buf.size)
        for (i in s until e) {
            buf[i] = 0.2f
        }
    }
}
