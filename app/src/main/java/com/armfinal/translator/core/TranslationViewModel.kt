package com.armfinal.translator.core

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.armfinal.translator.audio.AudioRecorder
import com.armfinal.translator.nativebridge.NativeBridge
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class UiState(
    val direction: LanguageDirection = LanguageDirection.EN_TO_HI,
    val thermalMode: ThermalMode = ThermalMode.NORMAL,
    val debugOverrideMode: ThermalMode? = null,
    val effectiveMode: ThermalMode = ThermalMode.NORMAL,
    val isInitialized: Boolean = false,
    val isRecording: Boolean = false,
    val partialTranscript: String = "",
    val finalTranscript: String = "",
    val translation: String = "",
    val status: String = "Initializing...",
    val runtimeStats: String = "",
    val lastLatencyMs: Long = 0,
)

class TranslationViewModel(app: Application) : AndroidViewModel(app) {
    private val orchestrator = TranslationOrchestrator(app.applicationContext)
    private val thermalMonitor = ThermalMonitor(app.applicationContext) { mode ->
        _uiState.update { current ->
            val nextMode = current.debugOverrideMode ?: mode
            orchestrator.setThermalMode(nextMode)
            current.copy(thermalMode = mode, effectiveMode = nextMode)
        }
    }

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            runCatching {
                orchestrator.initialize()
                orchestrator.setThermalMode(ThermalMode.NORMAL)
                thermalMonitor.start()
                orchestrator.setDirection(_uiState.value.direction)
            }.onSuccess {
                _uiState.update {
                    it.copy(
                        isInitialized = true,
                        status = "Ready",
                    )
                }
            }.onFailure { err ->
                _uiState.update {
                    it.copy(
                        status = "Initialization failed: ${err.message}",
                    )
                }
            }
        }
    }

    fun setDirection(direction: LanguageDirection) {
        viewModelScope.launch {
            if (!_uiState.value.isInitialized) {
                return@launch
            }

            orchestrator.setDirection(direction)
            _uiState.update {
                it.copy(
                    direction = direction,
                    partialTranscript = "",
                    finalTranscript = "",
                    translation = "",
                    status = "Direction switched",
                )
            }
        }
    }

    fun setDebugModeOverride(mode: ThermalMode?) {
        _uiState.update { state ->
            val effective = mode ?: state.thermalMode
            orchestrator.setThermalMode(effective)
            state.copy(
                debugOverrideMode = mode,
                effectiveMode = effective,
                status = if (mode == null) "Debug override off" else "Debug mode: ${mode.name}",
            )
        }
    }

    fun startRecording() {
        if (!_uiState.value.isInitialized || _uiState.value.isRecording) {
            return
        }

        _uiState.update {
            it.copy(
                isRecording = true,
                partialTranscript = "",
                status = "Listening...",
            )
        }

        viewModelScope.launch {
            runCatching {
                orchestrator.startRecording { partial ->
                    _uiState.update { current -> current.copy(partialTranscript = partial) }
                }
            }.onFailure { err ->
                _uiState.update {
                    it.copy(
                        isRecording = false,
                        status = "Record start failed: ${err.message}",
                    )
                }
            }
        }
    }

    fun stopRecording() {
        if (!_uiState.value.isInitialized || !_uiState.value.isRecording) {
            return
        }

        _uiState.update {
            it.copy(
                isRecording = false,
                status = "Processing...",
            )
        }

        viewModelScope.launch {
            val mode = _uiState.value.effectiveMode
            runCatching {
                orchestrator.stopRecording(mode)
            }.onSuccess { output ->
                _uiState.update {
                    it.copy(
                        finalTranscript = output.transcript,
                        translation = output.translation,
                        runtimeStats = output.runtimeStats,
                        lastLatencyMs = output.latencyMs,
                        status = "Done",
                    )
                }
            }.onFailure { err ->
                _uiState.update {
                    it.copy(
                        status = "Processing failed: ${err.message}",
                    )
                }
            }
        }
    }

    fun onTrimMemory(level: Int) {
        viewModelScope.launch {
            orchestrator.onTrimMemory(level)
        }
    }

    fun onAppBackground() {
        viewModelScope.launch {
            orchestrator.onAppBackground()
        }
    }

    fun runVadDebugCapture() {
        if (!_uiState.value.isInitialized || _uiState.value.isRecording) {
            return
        }

        _uiState.update { it.copy(status = "VAD test: recording 5s...") }
        viewModelScope.launch {
            runCatching {
                val recorder = AudioRecorder(sampleRateHz = 16_000)
                recorder.start(viewModelScope) { }
                delay(5_000)
                val rawAudio = recorder.stop()
                val pcm = PcmUtils.shortToFloat(rawAudio)
                val segments = NativeBridge.vadTrimAndSplit(
                    pcmF32 = pcm,
                    sampleRate = 16_000,
                    splitGapMs = 700,
                )
                val segmentsLog = segments.joinToString(prefix = "[", postfix = "]")
                Log.i(
                    "SpeechTranslation",
                    "VAD debug capture: samples=${pcm.size}, segments=$segmentsLog",
                )
            }.onSuccess {
                _uiState.update { it.copy(status = "VAD test complete (see logcat)") }
            }.onFailure { err ->
                _uiState.update { it.copy(status = "VAD test failed: ${err.message}") }
            }
        }
    }

    fun runAsrDebugCapture() {
        if (!_uiState.value.isInitialized || _uiState.value.isRecording) {
            return
        }

        _uiState.update { it.copy(status = "ASR test: recording 5s...") }
        viewModelScope.launch {
            runCatching {
                val recorder = AudioRecorder(sampleRateHz = 16_000)
                recorder.start(viewModelScope) { }
                delay(5_000)

                val rawAudio = recorder.stop()
                val pcm = PcmUtils.shortToFloat(rawAudio)
                val bounds = NativeBridge.vadTrimAndSplit(
                    pcmF32 = pcm,
                    sampleRate = 16_000,
                    splitGapMs = 700,
                )

                if (bounds.isEmpty() || bounds.size % 2 != 0) {
                    Log.i("SpeechTranslation", "ASR debug: no valid VAD segments")
                } else {
                    var segmentIndex = 0
                    var i = 0
                    while (i < bounds.size - 1) {
                        val start = bounds[i]
                        val end = bounds[i + 1]
                        val segmentPcm = PcmUtils.sliceSegment(pcm, start, end)
                        val text = NativeBridge.whisperTranscribeOnce(
                            pcmF32 = segmentPcm,
                            sampleRate = 16_000,
                            lang = _uiState.value.direction.source.nativeId,
                        )
                        Log.i(
                            "SpeechTranslation",
                            "ASR debug segment[$segmentIndex] bounds=[$start,$end] text=$text",
                        )
                        i += 2
                        segmentIndex += 1
                    }
                }
            }.onSuccess {
                _uiState.update { it.copy(status = "ASR test complete (see logcat)") }
            }.onFailure { err ->
                _uiState.update { it.copy(status = "ASR test failed: ${err.message}") }
            }
        }
    }

    override fun onCleared() {
        thermalMonitor.stop()
        orchestrator.close()
        super.onCleared()
    }
}
