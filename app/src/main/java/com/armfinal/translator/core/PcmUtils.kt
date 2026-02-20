package com.armfinal.translator.core

import kotlin.math.max
import kotlin.math.min

object PcmUtils {
    fun shortToFloat(input: ShortArray): FloatArray {
        val out = FloatArray(input.size)
        for (i in input.indices) {
            out[i] = input[i] / 32768f
        }
        return out
    }

    fun sliceSegment(input: FloatArray, start: Int, end: Int): FloatArray {
        val safeStart = max(0, min(start, input.size))
        val safeEnd = max(safeStart, min(end, input.size))
        return input.copyOfRange(safeStart, safeEnd)
    }

    fun segmentsFromBounds(input: FloatArray, bounds: IntArray): List<FloatArray> {
        if (bounds.size < 2 || bounds.size % 2 != 0) {
            return listOf(input)
        }

        val segments = mutableListOf<FloatArray>()
        var i = 0
        while (i < bounds.size - 1) {
            val start = bounds[i]
            val end = bounds[i + 1]
            val segment = sliceSegment(input, start, end)
            if (segment.isNotEmpty()) {
                segments += segment
            }
            i += 2
        }

        return if (segments.isEmpty()) listOf(input) else segments
    }

    fun splitClauses(text: String): List<String> {
        val cleaned = text.trim()
        if (cleaned.isEmpty()) {
            return emptyList()
        }

        val parts = cleaned
            .split(Regex("(?<=[.!?,;।])\\s+"))
            .map { it.trim() }
            .filter { it.isNotEmpty() }

        return if (parts.isEmpty()) listOf(cleaned) else parts
    }
}
