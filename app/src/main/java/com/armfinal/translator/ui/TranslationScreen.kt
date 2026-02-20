package com.armfinal.translator.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.armfinal.translator.core.LanguageDirection
import com.armfinal.translator.core.ThermalMode
import com.armfinal.translator.core.TranslationViewModel
import com.armfinal.translator.core.UiState
import kotlin.math.roundToInt

private val Background = Color(0xFF071124)
private val HeaderBackground = Color(0xFF0C172B)
private val Panel = Color(0xFF15233F)
private val PanelStroke = Color(0xFF1B2F50)
private val Muted = Color(0xFF93A2BC)
private val Accent = Color(0xFF1F64FF)
private val AccentSoft = Color(0x33256BFF)
private val Green = Color(0xFF18BE68)

@Composable
fun TranslationScreen(viewModel: TranslationViewModel) {
    val state by viewModel.uiState.collectAsState()
    val runtimeMap = remember(state.runtimeStats) { parseRuntimeStats(state.runtimeStats) }
    val rssMb = remember(runtimeMap) {
        runtimeMap["rss_mb"]
            ?: runtimeMap["rss_kb"]?.toDoubleOrNull()?.let { (it / 1024.0).roundToInt().toString() }
            ?: "850"
    }

    DashboardThemeScreen(
        state = state,
        rssMb = rssMb,
        onSetDirection = viewModel::setDirection,
        onSetDebugMode = viewModel::setDebugModeOverride,
        onStart = viewModel::startRecording,
        onStop = viewModel::stopRecording,
        onVadDebug = viewModel::runVadDebugCapture,
        onAsrDebug = viewModel::runAsrDebugCapture,
    )
}

@Composable
private fun DashboardThemeScreen(
    state: UiState,
    rssMb: String,
    onSetDirection: (LanguageDirection) -> Unit,
    onSetDebugMode: (ThermalMode?) -> Unit,
    onStart: () -> Unit,
    onStop: () -> Unit,
    onVadDebug: () -> Unit,
    onAsrDebug: () -> Unit,
) {
    Scaffold(
        containerColor = Background,
        bottomBar = {
            BottomBar(
                direction = state.direction,
                enabled = state.isInitialized,
                recording = state.isRecording,
                onToggleDirection = {
                    onSetDirection(
                        if (state.direction == LanguageDirection.EN_TO_HI) LanguageDirection.HI_TO_EN
                        else LanguageDirection.EN_TO_HI,
                    )
                },
                onStart = onStart,
                onStop = onStop,
            )
        },
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .verticalScroll(rememberScrollState())
                .background(Background),
        ) {
            TopSection(thermal = state.effectiveMode.name, rssMb = rssMb)

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 14.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                ThermalModeSelector(selected = state.debugOverrideMode, onSelected = onSetDebugMode)

                SectionHeader(
                    label = "Source (${if (state.direction == LanguageDirection.EN_TO_HI) "English" else "Hindi"})",
                    live = state.isRecording,
                )

                MessageBubble(
                    text = (if (state.isRecording) state.partialTranscript else state.finalTranscript)
                        .ifBlank { "Tap and hold mic to speak…" },
                    background = Panel,
                    color = Color(0xFFDCE5F6),
                )

                if (state.status.contains("processing", ignoreCase = true)) {
                    ProcessingChip()
                }

                SectionHeader(
                    label = "Target (${if (state.direction == LanguageDirection.EN_TO_HI) "Hindi" else "English"})",
                    live = false,
                )

                TranslationBubble(text = state.translation.ifBlank { "-" })

                StatusRuntimeCard(state)

                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    SmallDebugButton("VAD Test (5s)", onVadDebug)
                    SmallDebugButton("ASR Test (5s)", onAsrDebug)
                }

                Spacer(Modifier.height(12.dp))
            }
        }
    }
}

@Composable
private fun TopSection(thermal: String, rssMb: String) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(HeaderBackground)
            .padding(horizontal = 16.dp, vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(
                "On-Device Speech-to-Speech Translation",
                color = Color(0xFFE7EDF8),
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
            )
            Text("⚙", color = Muted, fontSize = 22.sp)
        }

        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            StatCard(
                title = "Thermal Mode",
                value = thermal.lowercase().replaceFirstChar { it.uppercase() },
                trailing = "●",
                trailingColor = Green,
                modifier = Modifier.weight(1f),
            )
            StatCard(
                title = "RAM Usage",
                value = "$rssMb MB / 1.2GB",
                trailing = "▣",
                trailingColor = Accent,
                modifier = Modifier.weight(1f),
                showProgress = true,
            )
        }

    }
}

@Composable
private fun StatCard(
    title: String,
    value: String,
    trailing: String,
    trailingColor: Color,
    modifier: Modifier = Modifier,
    showProgress: Boolean = false,
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(containerColor = Panel),
        shape = RoundedCornerShape(16.dp),
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(title, color = Muted, fontSize = 15.sp)
                Text(trailing, color = trailingColor, fontSize = 16.sp)
            }
            Text(value, color = Color(0xFFF2F6FD), fontSize = 18.sp, fontWeight = FontWeight.Bold)
            if (showProgress) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(8.dp)
                        .clip(RoundedCornerShape(999.dp))
                        .background(Color(0xFF33486A)),
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth(0.7f)
                            .height(8.dp)
                            .background(Accent, RoundedCornerShape(999.dp)),
                    )
                }
            }
        }
    }
}

@Composable
private fun ThermalModeSelector(selected: ThermalMode?, onSelected: (ThermalMode?) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text("Debug Thermal Override", color = Muted)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            ModeChip("Auto", selected == null) { onSelected(null) }
            ModeChip("NORMAL", selected == ThermalMode.NORMAL) { onSelected(ThermalMode.NORMAL) }
            ModeChip("THROTTLED", selected == ThermalMode.THROTTLED) { onSelected(ThermalMode.THROTTLED) }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            ModeChip("EMERGENCY", selected == ThermalMode.EMERGENCY) { onSelected(ThermalMode.EMERGENCY) }
            ModeChip("CRITICAL", selected == ThermalMode.CRITICAL) { onSelected(ThermalMode.CRITICAL) }
        }
    }
}

@Composable
private fun ModeChip(text: String, active: Boolean, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        colors = ButtonDefaults.buttonColors(
            containerColor = if (active) Accent else Color(0xFF2A3D60),
            contentColor = Color.White,
        ),
        shape = RoundedCornerShape(24.dp),
    ) { Text(text, fontSize = 12.sp, fontWeight = FontWeight.Bold) }
}

@Composable
private fun SectionHeader(label: String, live: Boolean) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, color = Muted, fontSize = 17.sp)
        if (live) {
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(999.dp))
                    .background(AccentSoft)
                    .padding(horizontal = 14.dp, vertical = 6.dp),
            ) {
                Text("LIVE", color = Accent, fontWeight = FontWeight.Bold)
            }
        }
    }
}

@Composable
private fun MessageBubble(text: String, background: Color, color: Color) {
    Card(
        colors = CardDefaults.cardColors(containerColor = background),
        shape = RoundedCornerShape(28.dp),
    ) {
        Text(
            text = text,
            color = color,
            fontSize = 22.sp,
            lineHeight = 38.sp,
            modifier = Modifier.padding(horizontal = 20.dp, vertical = 24.dp),
        )
    }
}

@Composable
private fun ProcessingChip() {
    Box(modifier = Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
        Card(
            colors = CardDefaults.cardColors(containerColor = AccentSoft),
            shape = RoundedCornerShape(999.dp),
        ) {
            Row(modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("↻", color = Accent, fontWeight = FontWeight.Bold)
                Spacer(Modifier.width(8.dp))
                Text("NLLB-200 Quantized Running…", color = Accent, fontSize = 14.sp)
            }
        }
    }
}

@Composable
private fun TranslationBubble(text: String) {
    Card(
        colors = CardDefaults.cardColors(containerColor = Accent),
        shape = RoundedCornerShape(28.dp),
    ) {
        Column(modifier = Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(20.dp)) {
            Text(text = text, color = Color.White, fontSize = 21.sp, lineHeight = 35.sp, fontWeight = FontWeight.Medium)
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                CircleIcon("🔊")
                Spacer(Modifier.width(10.dp))
                CircleIcon("⤴")
            }
        }
    }
}

@Composable
private fun CircleIcon(text: String) {
    Box(
        modifier = Modifier
            .size(48.dp)
            .background(Color(0x44FFFFFF), CircleShape),
        contentAlignment = Alignment.Center,
    ) {
        Text(text, color = Color.White, fontSize = 20.sp)
    }
}

@Composable
private fun StatusRuntimeCard(state: UiState) {
    Card(colors = CardDefaults.cardColors(containerColor = Panel), shape = RoundedCornerShape(18.dp)) {
        Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(5.dp)) {
            Text("Status: ${state.status}", color = Color.White)
            Text("Mode: ${state.effectiveMode.name}", color = Muted)
            Text("Latency: ${state.lastLatencyMs}ms", color = Muted)
            HorizontalDivider(color = PanelStroke, modifier = Modifier.padding(vertical = 2.dp))
            Text(state.runtimeStats.ifBlank { "-" }, color = Muted)
        }
    }
}

@Composable
private fun SmallDebugButton(text: String, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF5B46B8), contentColor = Color.White),
        shape = RoundedCornerShape(24.dp),
    ) { Text(text, fontWeight = FontWeight.SemiBold) }
}

@Composable
private fun BottomBar(
    direction: LanguageDirection,
    enabled: Boolean,
    recording: Boolean,
    onToggleDirection: () -> Unit,
    onStart: () -> Unit,
    onStop: () -> Unit,
) {
    Surface(color = HeaderBackground) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 24.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Button(
                onClick = onToggleDirection,
                enabled = enabled && !recording,
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color.Transparent,
                    contentColor = Color.White,
                    disabledContainerColor = Color.Transparent,
                    disabledContentColor = Muted,
                ),
            ) {
                val lang = if (direction == LanguageDirection.EN_TO_HI) "EN\n↕\nHI" else "HI\n↕\nEN"
                Text(lang, fontWeight = FontWeight.Bold)
            }

            Box(
                modifier = Modifier
                    .size(120.dp)
                    .background(if (recording) Color(0xFF2E7BFF) else Accent, CircleShape)
                    .pointerInput(enabled) {
                        if (!enabled) return@pointerInput
                        detectTapGestures(
                            onPress = {
                                onStart()
                                try {
                                    tryAwaitRelease()
                                } finally {
                                    onStop()
                                }
                            },
                        )
                    },
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text = if (!enabled) "Init" else if (recording) "Release to stop" else "🎤",
                    color = Color.White,
                    fontWeight = FontWeight.Bold,
                    fontSize = if (recording) 18.sp else 32.sp,
                    maxLines = 1,
                )
            }

            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text("🔈", color = Muted, fontSize = 24.sp)
                Text("Replay", color = Muted)
            }
        }
    }
}

@Composable
private fun SplitThemeRecordingScreen(
    state: UiState,
    rssMb: String,
) {
    val blueTop = Color(0xFF225DDD)
    val orangeBottom = Color(0xFFE65A04)

    Column(modifier = Modifier.fillMaxSize().background(Background)) {
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .background(blueTop, RoundedCornerShape(bottomStart = 38.dp, bottomEnd = 38.dp)),
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 18.dp, vertical = 16.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.SpaceBetween,
            ) {
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text("OFFLINE • WHISPER TINY", color = Color(0xA8FFFFFF), fontWeight = FontWeight.Bold)
                    Text("📶", color = Color(0xA8FFFFFF))
                }

                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("ENGLISH (US)", color = Color(0x88FFFFFF), letterSpacing = 3.sp, fontWeight = FontWeight.Bold)
                    Spacer(Modifier.height(10.dp))
                    Text(
                        text = "\"${state.partialTranscript.ifBlank { "Listening…" }}\"",
                        color = Color.White,
                        fontSize = 52.sp * 0.5f,
                        lineHeight = 62.sp * 0.5f,
                        fontWeight = FontWeight.ExtraBold,
                    )
                }

                Button(
                    onClick = { },
                    enabled = false,
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(24.dp),
                    colors = ButtonDefaults.buttonColors(
                        disabledContainerColor = Color(0x36FFFFFF),
                        disabledContentColor = Color.White,
                    ),
                ) {
                    Text("🎤   Hold to Speak", fontWeight = FontWeight.Bold, fontSize = 36.sp * 0.5f)
                }
            }

            FloatingStatsBar(rssMb = rssMb, modifier = Modifier.align(Alignment.BottomCenter).padding(horizontal = 14.dp, vertical = 10.dp))
        }

        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .background(brush = Brush.linearGradient(listOf(orangeBottom, Color(0xFFE25703)))),
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(18.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.SpaceBetween,
            ) {
                Spacer(Modifier.height(38.dp))
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "\"${state.translation.ifBlank { "..." }}\"",
                        color = Color.White,
                        fontSize = 56.sp * 0.5f,
                        lineHeight = 64.sp * 0.5f,
                        fontWeight = FontWeight.ExtraBold,
                    )
                    Spacer(Modifier.height(8.dp))
                    Text("HINDI (INDIA)", color = Color(0xB0FFFFFF), letterSpacing = 3.sp, fontWeight = FontWeight.Bold)
                }

                Card(colors = CardDefaults.cardColors(containerColor = Color.White), shape = RoundedCornerShape(26.dp)) {
                    Text(
                        text = "🎤   Listening…",
                        color = Color(0xFFE17A00),
                        fontSize = 38.sp * 0.5f,
                        fontWeight = FontWeight.ExtraBold,
                        modifier = Modifier.padding(horizontal = 30.dp, vertical = 18.dp),
                    )
                }

                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text("MODEL: NLLB INT8", color = Color(0xB0FFFFFF), fontWeight = FontWeight.Bold)
                    Text("⚡ OFFLINE", color = Color(0xB0FFFFFF), fontWeight = FontWeight.Bold)
                }
            }

        }
    }
}

@Composable
private fun FloatingStatsBar(rssMb: String, modifier: Modifier = Modifier) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF1D2C47)),
        shape = RoundedCornerShape(40.dp),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 18.dp, vertical = 12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("DEVICE TEMP\n42°C  High", color = Color(0xFFF9A147), fontWeight = FontWeight.Bold)
            Text("MEMORY RSS\n${rssMb}MB", color = Color(0xFF3DE3AB), fontWeight = FontWeight.Bold)
        }
    }
}

private fun parseRuntimeStats(runtime: String): Map<String, String> {
    if (runtime.isBlank()) return emptyMap()
    return runtime.split(",")
        .map { it.trim() }
        .mapNotNull { token ->
            val idx = token.indexOf('=')
            if (idx <= 0 || idx >= token.lastIndex) null else token.substring(0, idx).trim() to token.substring(idx + 1).trim()
        }
        .toMap()
}
