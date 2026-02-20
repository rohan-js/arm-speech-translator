package com.armfinal.translator.storage

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

class AssetModelManager(
    private val context: Context,
) {
    companion object {
        private val REQUIRED_ASSET_FILES = listOf(
            "models/sanity/mul_1.onnx",
            "models/silero/silero_vad.onnx",
            "models/whisper/ggml-tiny.bin",
            "models/fallback/phrases_en_hi.tsv",
            "models/fallback/phrases_hi_en.tsv",
        )
    }

    suspend fun ensureExtracted(): File = withContext(Dispatchers.IO) {
        val targetRoot = File(context.filesDir, "models")
        check(targetRoot.mkdirs() || targetRoot.exists()) {
            "Unable to create model directory: ${targetRoot.absolutePath}"
        }

        REQUIRED_ASSET_FILES.forEach { assetPath ->
            copyAssetIfMissing(assetPath, targetRoot)
        }
        targetRoot
    }

    private fun copyAssetIfMissing(assetPath: String, targetRoot: File) {
        val relativePath = assetPath.removePrefix("models/")
        val outputFile = File(targetRoot, relativePath)
        if (outputFile.exists() && outputFile.isFile && outputFile.length() > 0L) {
            return
        }
        outputFile.parentFile?.mkdirs()
        val tempFile = File(outputFile.parentFile, "${outputFile.name}.tmp")
        if (tempFile.exists()) tempFile.delete()

        context.assets.open(assetPath).use { input ->
            FileOutputStream(tempFile, false).use { output ->
                input.copyTo(output)
                output.fd.sync()
            }
        }
        if (outputFile.exists() && !outputFile.delete()) {
            throw IllegalStateException("Unable to overwrite file: ${outputFile.absolutePath}")
        }
        check(tempFile.renameTo(outputFile)) {
            "Unable to move temp model file to ${outputFile.absolutePath}"
        }
        if (outputFile.length() <= 0L) {
            throw IllegalStateException("Copied file is invalid: ${outputFile.absolutePath}")
        }
    }
}

fun getExternalNllbDir(context: Context): File =
    File(context.getExternalFilesDir(null), "models/nllb")
