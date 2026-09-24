import Foundation
import SwiftUI

private enum GizzAI {
    static let porcelain = Color(red: 0.969, green: 0.961, blue: 0.949) // --gz-bg #f7f5f2
    static let ink = Color(red: 0.161, green: 0.145, blue: 0.122)       // --gz-fg #29251f
    static let purple = Color(red: 0.420, green: 0.302, blue: 0.569)    // --gz-brand #6b4d91
    static let purpleSoft = Color(red: 0.957, green: 0.937, blue: 1.0)  // --gz-brand-soft #f4efff
    static let success = Color(red: 0.086, green: 0.439, blue: 0.290)   // --gz-success #16704a
}

struct Forecast: Decodable { let q10: Double; let q50: Double; let q90: Double; let unit: String?; let ms: Double? }

@main struct SenseTMacApp: App {
    var body: some Scene { WindowGroup { ContentView() }.windowStyle(.hiddenTitleBar) }
}

struct ContentView: View {
    @State private var endpoint = "http://127.0.0.1:8770"
    @State private var state = "未来第 7 天有促销。 / Promotion on day 7."
    @State private var value = ""
    @State private var result: Forecast?
    @State private var message = "准备连接 / Ready to connect"
    @State private var busy = false

    var body: some View {
        VStack(alignment: .leading, spacing: 24) {
            HStack(spacing: 14) {
                Image(systemName: "chart.xyaxis.line").font(.system(size: 28, weight: .semibold)).foregroundStyle(GizzAI.purple)
                VStack(alignment: .leading, spacing: 3) {
                    Text("叽喳 Sense-T").font(.title2.weight(.semibold))
                    Text("时间序列判断与预测 / Time-series decisions and forecasts").font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Text("试用包 / TEST").font(.caption.weight(.semibold)).padding(.horizontal, 10).padding(.vertical, 6).background(GizzAI.purpleSoft).clipShape(Capsule()).foregroundStyle(GizzAI.purple)
            }
            GroupBox {
                VStack(alignment: .leading, spacing: 14) {
                    Text("连接 / Connection").font(.headline)
                    TextField("Sense-T API endpoint", text: $endpoint).textFieldStyle(.roundedBorder)
                    Text("启动服务：python demos/sense_t_server.py --model GizzAI-Sense-T-E2B --port 8770\nStart the server, then run a forecast below.").font(.caption).foregroundStyle(.secondary)
                }.padding(4)
            }
            GroupBox {
                VStack(alignment: .leading, spacing: 14) {
                    Text("实时预测 / Live forecast").font(.headline)
                    TextField("状态 / State", text: $state).textFieldStyle(.roundedBorder)
                    HStack {
                        Text("最后观测值 / Last value")
                        TextField("980", text: $value).textFieldStyle(.roundedBorder).frame(width: 120)
                        Spacer()
                        Button(busy ? "请求中… / Loading…" : "预测 / Forecast") { Task { await forecast() } }.buttonStyle(.borderedProminent).tint(GizzAI.purple).disabled(busy)
                    }
                }.padding(4)
            }
            if let r = result {
                HStack(spacing: 12) {
                    Metric(title: "P10", value: r.q10, color: GizzAI.purple)
                    Metric(title: "P50", value: r.q50, color: GizzAI.ink)
                    Metric(title: "P90", value: r.q90, color: GizzAI.success)
                }
                Text("延迟 / Latency: \(r.ms.map { String(format: "%.0f ms", $0) } ?? "—") · 单位 / Unit: \(r.unit ?? "—")").font(.caption).foregroundStyle(.secondary)
            }
            Text(message).font(.caption).foregroundStyle(.secondary)
        }.padding(30).frame(minWidth: 620, minHeight: 480).background(GizzAI.porcelain).foregroundStyle(GizzAI.ink)
    }

    private func forecast() async {
        busy = true; defer { busy = false }
        let last = Double(value) ?? 980
        let series: [String: Any] = ["names": ["Store A daily sales"], "values": [[last - 30, last - 12, last + 8, last, last + 24, last + 15, last]], "unit": "件 / units", "step": "天 / day"]
        let body: [String: Any] = ["series": series, "state": state, "step": 2, "horizon": 7]
        guard let url = URL(string: endpoint + "/v1/forecast"), let data = try? JSONSerialization.data(withJSONObject: body) else { message = "地址无效 / Invalid endpoint"; return }
        var request = URLRequest(url: url); request.httpMethod = "POST"; request.setValue("application/json", forHTTPHeaderField: "Content-Type"); request.httpBody = data
        do { let (response, _) = try await URLSession.shared.data(for: request); result = try JSONDecoder().decode(Forecast.self, from: response); message = "已收到真实模型响应 / Real model response received" }
        catch { message = "请求失败：\(error.localizedDescription) / Request failed" }
    }
}

struct Metric: View { let title: String; let value: Double; let color: Color; var body: some View { VStack(alignment: .leading, spacing: 4) { Text(title).font(.caption.weight(.semibold)).foregroundStyle(color); Text(String(format: "%.1f", value)).font(.title3.monospacedDigit().weight(.semibold)) }.frame(maxWidth: .infinity, alignment: .leading).padding(16).background(.white.opacity(0.78)).clipShape(RoundedRectangle(cornerRadius: 12)) } }
