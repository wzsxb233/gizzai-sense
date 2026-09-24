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

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) { super.onCreate(savedInstanceState); setContent { SenseTScreen() } }
}

@Composable fun SenseTScreen() {
    val activity = LocalContext.current as? Activity
    var endpoint by remember { mutableStateOf("http://10.0.2.2:8770") }
    var state by remember { mutableStateOf("未来第 7 天有促销。 / Promotion on day 7.") }
    var status by remember { mutableStateOf("准备连接 / Ready to connect") }
    var values by remember { mutableStateOf(listOf<Double>()) }
    var loading by remember { mutableStateOf(false) }
    Surface(color = Porcelain, modifier = Modifier.fillMaxSize()) {
        Column(modifier = Modifier.padding(24.dp), verticalArrangement = Arrangement.spacedBy(18.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("◈", color = Purple, fontSize = 30.sp, fontWeight = FontWeight.Bold)
                Column { Text("叽喳 Sense-T", color = Ink, fontSize = 21.sp, fontWeight = FontWeight.SemiBold); Text("时间序列判断与预测 / Time-series decisions and forecasts", fontSize = 12.sp, color = Color.Gray) }
                Spacer(Modifier.weight(1f)); Text("试用包 / TEST", color = Purple, fontSize = 11.sp, fontWeight = FontWeight.Bold, modifier = Modifier.background(PurpleSoft, RoundedCornerShape(20.dp)).padding(horizontal = 10.dp, vertical = 7.dp))
            }
            ElevatedCard { Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("连接 / Connection", color = Ink, fontWeight = FontWeight.SemiBold)
                OutlinedTextField(endpoint, { endpoint = it }, label = { Text("Sense-T API endpoint") }, singleLine = true, modifier = Modifier.fillMaxWidth())
                Text("模拟器请用 10.0.2.2；真机请填电脑局域网地址。 / Use 10.0.2.2 in the emulator or your computer's LAN address on a device.", fontSize = 12.sp, color = Color.Gray)
            } }
            ElevatedCard { Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("实时预测 / Live forecast", color = Ink, fontWeight = FontWeight.SemiBold)
                OutlinedTextField(state, { state = it }, label = { Text("状态 / State") }, modifier = Modifier.fillMaxWidth())
                Button(onClick = { loading = true; status = "请求中… / Loading…"; Thread { val out = requestForecast(endpoint, state); activity?.runOnUiThread { loading = false; if (out != null) { values = out; status = "已收到真实模型响应 / Real model response received" } else status = "请求失败 / Request failed" } } }, enabled = !loading, colors = ButtonDefaults.buttonColors(containerColor = Purple)) { Text(if (loading) "请求中… / Loading…" else "预测 / Forecast") }
            } }
            if (values.size == 3) Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) { Metric("P10", values[0], Purple, Modifier.weight(1f)); Metric("P50", values[1], Ink, Modifier.weight(1f)); Metric("P90", values[2], Success, Modifier.weight(1f)) }
            Text(status, fontSize = 12.sp, color = Color.Gray)
        }
    }
}

@Composable private fun Metric(label: String, value: Double, color: Color, modifier: Modifier = Modifier) { ElevatedCard(modifier) { Column(Modifier.padding(14.dp)) { Text(label, color = color, fontWeight = FontWeight.Bold, fontSize = 12.sp); Text("%.1f".format(value), color = Ink, fontSize = 20.sp, fontWeight = FontWeight.SemiBold) } } }

private fun requestForecast(endpoint: String, state: String): List<Double>? = try {
    val json = JSONObject().apply { put("state", state); put("step", 2); put("horizon", 7); put("series", JSONObject().apply { put("names", listOf("Store A daily sales")); put("values", listOf(listOf(950, 968, 988, 980, 1004, 995, 980))); put("unit", "件 / units"); put("step", "天 / day") }) }
    val c = (URL(endpoint.trimEnd('/') + "/v1/forecast").openConnection() as HttpURLConnection); c.requestMethod = "POST"; c.doOutput = true; c.setRequestProperty("Content-Type", "application/json"); c.outputStream.use { it.write(json.toString().toByteArray()) }; val out = JSONObject(c.inputStream.bufferedReader().readText()); listOf(out.getDouble("q10"), out.getDouble("q50"), out.getDouble("q90"))
} catch (_: Exception) { null }
