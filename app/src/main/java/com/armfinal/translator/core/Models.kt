package com.armfinal.translator.core

import android.os.PowerManager

enum class Language(val nativeId: Int) {
    EN(0),
    HI(1),
}

enum class Voice(val nativeId: Int, val sampleRate: Int) {
    EN(0, 22050),
    HI(1, 22050),
}

enum class LanguageDirection(val source: Language, val target: Language) {
    EN_TO_HI(Language.EN, Language.HI),
    HI_TO_EN(Language.HI, Language.EN),
}

enum class ThermalMode(
    val nativeValue: Int,
    val nllbThreadCount: Int,
    val nllbTokenCap: Int,
    val useWhisper: Boolean,
    val useNllb: Boolean,
) {
    NORMAL(0, nllbThreadCount = 2, nllbTokenCap = 64, useWhisper = true, useNllb = true),
    THROTTLED(1, nllbThreadCount = 1, nllbTokenCap = 48, useWhisper = true, useNllb = true),
    EMERGENCY(2, nllbThreadCount = 1, nllbTokenCap = 48, useWhisper = false, useNllb = true),
    CRITICAL(3, nllbThreadCount = 0, nllbTokenCap = 0, useWhisper = false, useNllb = false),
    ;

    companion object {
        fun fromPowerManager(status: Int): ThermalMode {
            return when (status) {
                PowerManager.THERMAL_STATUS_NONE,
                PowerManager.THERMAL_STATUS_LIGHT,
                PowerManager.THERMAL_STATUS_MODERATE,
                -> NORMAL

                PowerManager.THERMAL_STATUS_SEVERE -> THROTTLED
                PowerManager.THERMAL_STATUS_CRITICAL -> EMERGENCY
                PowerManager.THERMAL_STATUS_EMERGENCY,
                PowerManager.THERMAL_STATUS_SHUTDOWN,
                -> CRITICAL

                else -> NORMAL
            }
        }
    }
}
