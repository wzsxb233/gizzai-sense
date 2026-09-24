import Foundation
import SwiftUI

private enum GizzAI {
    static let porcelain = Color(red: 0.969, green: 0.961, blue: 0.949) // --gz-bg #f7f5f2
    static let ink = Color(red: 0.161, green: 0.145, blue: 0.122)       // --gz-fg #29251f
    static let purple = Color(red: 0.420, green: 0.302, blue: 0.569)    // --gz-brand #6b4d91
    static let purpleSoft = Color(red: 0.957, green: 0.937, blue: 1.0)  // --gz-brand-soft #f4efff
    static let success = Color(red: 0.086, green: 0.439, blue: 0.290)   // --gz-success #16704a
}

private enum ModelFamily: String, CaseIterable, Identifiable {
    case senseE2B = "GizzAI-Sense-E2B"
    case senseE4B = "GizzAI-Sense-E4B"
    case senseT = "GizzAI-Sense-T-E2B"
    var id: String { rawValue }
    var title: String {
        switch self { case .senseE2B: return "Sense E2B / 判断"; case .senseE4B: return "Sense E4B / 判断"; case .senseT: return "Sense-T E2B / 预测" }
    }
    var isTimeSeries: Bool { self == .senseT }
}

private struct Forecast: Decodable { let q10: Double; let q50: Double; let q90: Double; let unit: String?; let ms: Double? }
private struct Decision: Decodable {
    let answer: String?; let confidence: Double?; let probabilities: [String: Double]?; let latencyMS: Double?
    enum CodingKeys: String, CodingKey { case answer, confidence, probabilities; case latencyMS = "latency_ms" }
}

@main struct SenseTMacApp: App {
    var body: some Scene {
        #if os(macOS)
        WindowGroup { ContentView() }.windowStyle(.hiddenTitleBar)
        #else
        WindowGroup { ContentView() }
        #endif
    }
}

struct ContentView: View {
    @State private var endpoint = "http://127.0.0.1:8770"
    @State private var model: ModelFamily = .senseT
    @State private var state = "未来第 7 天有促销。 / Promotion on day 7."
    @State private var question = "是否需要立即行动？ / Is immediate action needed?"
    @State private var value = ""
    @State private var forecast: Forecast?
    @State private var decision: Decision?
    @State private var message = "准备连接 / Ready to connect"
    @State private var busy = false

    var body: some View {
        VStack(alignment: .leading, spacing: 24) {
            HStack(spacing: 14) {
                Image(systemName: model.isTimeSeries ? "chart.xyaxis.line" : "sparkles").font(.system(size: 28, weight: .semibold)).foregroundStyle(GizzAI.purple)
                VStack(alignment: .leading, spacing: 3) { Text("叽喳 Sense 系列").font(.title2.weight(.semibold)); Text("三模型设备试用 / Three-model device test").font(.caption).foregroundStyle(.secondary) }
                Spacer(); Text("试用包 / TEST").font(.caption.weight(.semibold)).padding(.horizontal, 10).padding(.vertical, 6).background(GizzAI.purpleSoft).clipShape(Capsule()).foregroundStyle(GizzAI.purple)
            }
            GroupBox { VStack(alignment: .leading, spacing: 14) {
                Text("模型 / Model").font(.headline)
                Picker("模型 / Model", selection: $model) { ForEach(ModelFamily.allCases) { option in Text(option.title).tag(option) } }.pickerStyle(.segmented)
                Text(model.rawValue).font(.caption.monospaced()).foregroundStyle(.secondary)
                TextField("API endpoint", text: $endpoint).textFieldStyle(.roundedBorder)
                Text(model.isTimeSeries ? "Sense-T 服务：python demos/sense_t_server.py --model \(model.rawValue) --port 8770" : "Sense 服务：python demos/live_server.py --model \(model.rawValue) --port 8765").font(.caption).foregroundStyle(.secondary)
            }.padding(4) }
            GroupBox { VStack(alignment: .leading, spacing: 14) {
                Text(model.isTimeSeries ? "实时预测 / Live forecast" : "实时判断 / Live decision").font(.headline)
                TextField("状态 / State", text: $state).textFieldStyle(.roundedBorder)
                if model.isTimeSeries {
                    HStack { Text("最后观测值 / Last value"); TextField("980", text: $value).textFieldStyle(.roundedBorder).frame(width: 120); Spacer(); Button(busy ? "请求中… / Loading…" : "预测 / Forecast") { Task { await runModel() } }.buttonStyle(.borderedProminent).tint(GizzAI.purple).disabled(busy) }
                } else {
                    TextField("问题 / Question", text: $question).textFieldStyle(.roundedBorder)
                    Button(busy ? "请求中… / Loading…" : "判断 / Decide") { Task { await runModel() } }.buttonStyle(.borderedProminent).tint(GizzAI.purple).disabled(busy)
                }
            }.padding(4) }
            if let r = forecast {
                HStack(spacing: 12) { Metric(title: "P10", value: r.q10, color: GizzAI.purple); Metric(title: "P50", value: r.q50, color: GizzAI.ink); Metric(title: "P90", value: r.q90, color: GizzAI.success) }
                Text("延迟 / Latency: \(r.ms.map { String(format: "%.0f ms", $0) } ?? "—") · 单位 / Unit: \(r.unit ?? "—")").font(.caption).foregroundStyle(.secondary)
            }
            if let r = decision {
                HStack(spacing: 12) { MetricText(title: "答案 / Answer", value: r.answer ?? "—", color: GizzAI.purple); Metric(title: "置信度 / Confidence", value: r.confidence ?? 0, color: GizzAI.success) }
                Text("延迟 / Latency: \(r.latencyMS.map { String(format: "%.0f ms", $0) } ?? "—")").font(.caption).foregroundStyle(.secondary)
            }
            Text(message).font(.caption).foregroundStyle(.secondary)
        }.padding(30).frame(minWidth: 620, minHeight: 520).background(GizzAI.porcelain).foregroundStyle(GizzAI.ink)
    }

    private func runModel() async { busy = true; forecast = nil; decision = nil; defer { busy = false }; if model.isTimeSeries { await runForecast() } else { await runDecision() } }
    private func runForecast() async {
        let last = Double(value) ?? 980
        let series: [String: Any] = ["names": ["Store A daily sales"], "values": [[last - 30, last - 12, last + 8, last, last + 24, last + 15, last]], "unit": "件 / units", "step": "天 / day"]
        await post(path: "/v1/forecast", body: ["series": series, "state": state, "step": 2, "horizon": 7]) { data in forecast = try? JSONDecoder().decode(Forecast.self, from: data) }
    }
    private func runDecision() async { await post(path: "/v1/decide", body: ["state": state, "question": ["type": "noul", "instructions": question]]) { data in decision = try? JSONDecoder().decode(Decision.self, from: data) } }
    private func post(path: String, body: [String: Any], decode: @escaping (Data) -> Void) async {
        guard let url = URL(string: endpoint.trimmingCharacters(in: CharacterSet(charactersIn: "/")) + path), let data = try? JSONSerialization.data(withJSONObject: body) else { message = "地址无效 / Invalid endpoint"; return }
        var request = URLRequest(url: url); request.httpMethod = "POST"; request.setValue("application/json", forHTTPHeaderField: "Content-Type"); request.httpBody = data
        do { let (response, _) = try await URLSession.shared.data(for: request); decode(response); message = "已收到真实模型响应 / Real model response received" } catch { message = "请求失败：\(error.localizedDescription) / Request failed" }
    }
}

struct Metric: View { let title: String; let value: Double; let color: Color; var body: some View { VStack(alignment: .leading, spacing: 4) { Text(title).font(.caption.weight(.semibold)).foregroundStyle(color); Text(String(format: "%.1f", value)).font(.title3.monospacedDigit().weight(.semibold)) }.frame(maxWidth: .infinity, alignment: .leading).padding(16).background(.white.opacity(0.78)).clipShape(RoundedRectangle(cornerRadius: 12)) } }
struct MetricText: View { let title: String; let value: String; let color: Color; var body: some View { VStack(alignment: .leading, spacing: 4) { Text(title).font(.caption.weight(.semibold)).foregroundStyle(color); Text(value).font(.title3.weight(.semibold)) }.frame(maxWidth: .infinity, alignment: .leading).padding(16).background(.white.opacity(0.78)).clipShape(RoundedRectangle(cornerRadius: 12)) } }
