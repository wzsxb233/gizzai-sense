// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SenseTMac",
    platforms: [.macOS(.v13), .iOS(.v17)],
    products: [.executable(name: "SenseTMac", targets: ["SenseTMac"])],
    targets: [.executableTarget(name: "SenseTMac")]
)
