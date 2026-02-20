package com.armfinal.translator.core

import android.content.Context
import android.os.Debug
import android.os.SystemClock
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

class MemThermalLogger(
    private val context: Context,
) {
    private val csvFile: File by lazy { File(context.filesDir, "runtime_metrics.csv") }

    suspend fun log(event: String, mode: ThermalMode, runtimeStats: String, latencyMs: Long) {
        withContext(Dispatchers.IO) {
            if (!csvFile.exists()) {
                csvFile.writeText("uptime_ms,event,thermal_mode,latency_ms,total_pss_kb,runtime_stats\n")
            }

            val memInfo = Debug.MemoryInfo()
            Debug.getMemoryInfo(memInfo)

            val line = listOf(
                SystemClock.elapsedRealtime().toString(),
                sanitize(event),
                mode.name,
                latencyMs.toString(),
                memInfo.totalPss.toString(),
                sanitize(runtimeStats),
            ).joinToString(",")

            csvFile.appendText("$line\n")
        }
    }

    private fun sanitize(value: String): String {
        return '"' + value.replace('"', '\'').replace('\n', ' ') + '"'
    }
}
