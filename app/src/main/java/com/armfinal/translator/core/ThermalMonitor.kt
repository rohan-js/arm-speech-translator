package com.armfinal.translator.core

import android.content.Context
import android.os.Build
import android.os.PowerManager

class ThermalMonitor(
    private val context: Context,
    private val onModeChanged: (ThermalMode) -> Unit,
) {
    private var listener: PowerManager.OnThermalStatusChangedListener? = null

    fun start() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            onModeChanged(ThermalMode.NORMAL)
            return
        }

        val powerManager = context.getSystemService(PowerManager::class.java) ?: run {
            onModeChanged(ThermalMode.NORMAL)
            return
        }

        val thermalListener = PowerManager.OnThermalStatusChangedListener { status ->
            onModeChanged(ThermalMode.fromPowerManager(status))
        }
        listener = thermalListener
        powerManager.addThermalStatusListener(thermalListener)
        onModeChanged(ThermalMode.fromPowerManager(powerManager.currentThermalStatus))
    }

    fun stop() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            return
        }

        val thermalListener = listener ?: return
        val powerManager = context.getSystemService(PowerManager::class.java) ?: return
        powerManager.removeThermalStatusListener(thermalListener)
        listener = null
    }
}
