package com.gizzai.senset

import android.app.Activity
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

private val Porcelain = Color(0xFFF7F5F2) // --gz-bg
private val Ink = Color(0xFF29251F)       // --gz-fg
private val Purple = Color(0xFF6B4D91)    // --gz-brand
private val PurpleSoft = Color(0xFFF4EFFF) // --gz-brand-soft
private val Success = Color(0xFF16704A)   // --gz-success

private enum class ModelFamily(val wireName: String, val title: String, val timeSeries: Boolean) {
    SENSE_E2B("GizzAI-Sense-E2B", "Sense E2B / 判断", false),
    SENSE_E4B("GizzAI-Sense-E4B", "Sense E4B / 判断", false),
    SENSE_T("GizzAI-Sense-T-E2B", "Sense-T E2B / 预测", true)
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) { super.onCreate(savedInstanceState); setContent { SenseFamilyScreen() } }
}

@Composable fun SenseFamilyScreen() {
    val activity = LocalContext.current as? Activity
    var endpoint by remember { mutableStateOf("http://10.0.2.2:8770") }
    var model by remember { mutableStateOf(ModelFamily.SENSE_T) }
    var state by remember { mutableStateOf("未来第 7 天有促销。 / Promotion on day 7.") }
    var question by remember { mutableStateOf("是否需要立即行动？ / Is immediate action needed?") }
    var status by remember { mutableStateOf("准备连接 / Ready to connect") }
    var values by remember { mutableStateOf(listOf<Double>()) }
    var answer by remember { mutableStateOf<String?>(null) }
    var confidence by remember { mutableStateOf<Double?>(null) }
    var loading by remember { mutableStateOf(false) }
    Surface(color = Porcelain, modifier = Modifier.fillMaxSize()) {
        Column(modifier = Modifier.padding(24.dp), verticalArrangement = Arrangement.spacedBy(18.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(if (model.timeSeries) "◈" else "✦", color = Purple, fontSize = 30.sp, fontWeight = FontWeight.Bold)
                Column { Text("叽喳 Sense 系列", color = Ink, fontSize = 21.sp, fontWeight = FontWeight.SemiBold); Text("三模型设备试用 / Three-model device test", fontSize = 12.sp, color = Color.Gray) }
                Spacer(Modifier.weight(1f)); Text("试用包 / TEST", color = Purple, fontSize = 11.sp, fontWeight = FontWeight.Bold, modifier = Modifier.background(PurpleSoft, RoundedCornerShape(20.dp)).padding(horizontal = 10.dp, vertical = 7.dp))
            }
            ElevatedCard { Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("模型 / Model", color = Ink, fontWeight = FontWeight.SemiBold)
                ModelFamily.values().forEach { option ->
                    OutlinedButton(onClick = { model = option; values = emptyList(); answer = null; confidence = null }, modifier = Modifier.fillMaxWidth(), colors = ButtonDefaults.outlinedButtonColors(contentColor = if (model == option) Purple else Ink)) { Text(option.title) }
                }
                Text(model.wireName, fontSize = 12.sp, color = Color.Gray)
                OutlinedTextField(endpoint, { endpoint = it }, label = { Text("API endpoint") }, singleLine = true, modifier = Modifier.fillMaxWidth())
                Text(if (model.timeSeries) "Sense-T 服务端口通常为 8770。 / Sense-T server usually uses port 8770." else "Sense 服务端口通常为 8765。 / Sense server usually uses port 8765.", fontSize = 12.sp, color = Color.Gray)
            } }
            ElevatedCard { Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text(if (model.timeSeries) "实时预测 / Live forecast" else "实时判断 / Live decision", color = Ink, fontWeight = FontWeight.SemiBold)
                OutlinedTextField(state, { state = it }, label = { Text("状态 / State") }, modifier = Modifier.fillMaxWidth())
                if (!model.timeSeries) OutlinedTextField(question, { question = it }, label = { Text("问题 / Question") }, modifier = Modifier.fillMaxWidth())
                Button(onClick = { loading = true; status = "请求中… / Loading…"; Thread {
                    if (model.timeSeries) {
                        val out = requestForecast(endpoint, state)
                        activity?.runOnUiThread { loading = false; if (out != null) { values = out; answer = null; status = "已收到真实模型响应 / Real model response received" } else status = "请求失败 / Request failed" }
                    } else {
                        val out = requestDecision(endpoint, state, question)
                        activity?.runOnUiThread { loading = false; if (out != null) { answer = out.first; confidence = out.second; values = emptyList(); status = "已收到真实模型响应 / Real model response received" } else status = "请求失败 / Request failed" }
                    }
                }.start() }, enabled = !loading, colors = ButtonDefaults.buttonColors(containerColor = Purple)) { Text(if (loading) "请求中… / Loading…" else if (model.timeSeries) "预测 / Forecast" else "判断 / Decide") }
            } }
            if (values.size == 3) Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) { Metric("P10", values[0], Purple, Modifier.weight(1f)); Metric("P50", values[1], Ink, Modifier.weight(1f)); Metric("P90", values[2], Success, Modifier.weight(1f)) }
            if (answer != null) Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) { TextMetric("答案 / Answer", answer!!, Purple, Modifier.weight(1f)); Metric("置信度 / Confidence", confidence ?: 0.0, Success, Modifier.weight(1f)) }
            Text(status, fontSize = 12.sp, color = Color.Gray)
        }
    }
}

@Composable private fun Metric(label: String, value: Double, color: Color, modifier: Modifier = Modifier) { ElevatedCard(modifier) { Column(Modifier.padding(14.dp)) { Text(label, color = color, fontWeight = FontWeight.Bold, fontSize = 12.sp); Text("%.1f".format(value), color = Ink, fontSize = 20.sp, fontWeight = FontWeight.SemiBold) } } }
@Composable private fun TextMetric(label: String, value: String, color: Color, modifier: Modifier = Modifier) { ElevatedCard(modifier) { Column(Modifier.padding(14.dp)) { Text(label, color = color, fontWeight = FontWeight.Bold, fontSize = 12.sp); Text(value, color = Ink, fontSize = 18.sp, fontWeight = FontWeight.SemiBold) } } }

private fun requestForecast(endpoint: String, state: String): List<Double>? = try {
    val json = JSONObject().apply { put("state", state); put("step", 2); put("horizon", 7); put("series", JSONObject().apply { put("names", listOf("Store A daily sales")); put("values", listOf(listOf(950, 968, 988, 980, 1004, 995, 980))); put("unit", "件 / units"); put("step", "天 / day") }) }
    val c = post(endpoint, "/v1/forecast", json); val out = JSONObject(c); listOf(out.getDouble("q10"), out.getDouble("q50"), out.getDouble("q90"))
} catch (_: Exception) { null }

private fun requestDecision(endpoint: String, state: String, question: String): Pair<String, Double>? = try {
    val json = JSONObject().apply { put("state", state); put("question", JSONObject().apply { put("type", "noul"); put("instructions", question) }) }
    val out = JSONObject(post(endpoint, "/v1/decide", json)); Pair(out.optString("answer", "—"), out.optDouble("confidence", 0.0))
} catch (_: Exception) { null }

private fun post(endpoint: String, path: String, json: JSONObject): String {
    val c = (URL(endpoint.trimEnd('/') + path).openConnection() as HttpURLConnection)
    c.requestMethod = "POST"; c.doOutput = true; c.connectTimeout = 10000; c.readTimeout = 30000; c.setRequestProperty("Content-Type", "application/json")
    c.outputStream.use { it.write(json.toString().toByteArray()) }
    if (c.responseCode !in 200..299) error("HTTP ${c.responseCode}")
    return c.inputStream.bufferedReader().readText()
}
