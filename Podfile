# Uncomment the next line to define a global platform for your project
platform :ios, '13.0'

target 'RTMDetSwift' do
  # Comment the next line if you don't want to use dynamic frameworks
  use_frameworks!

  # Pods for RTMDetSwift
  # Use full ONNX Runtime (not mobile) for complete operator support
  pod 'onnxruntime-objc', '~> 1.18.0'

end

target 'RTMDetSwiftTests' do
  use_frameworks!
  # Pods for testing
  pod 'onnxruntime-objc', '~> 1.18.0'
end

post_install do |installer|
  installer.pods_project.targets.each do |target|
    target.build_configurations.each do |config|
      # Disable Mac Catalyst
      config.build_settings['SUPPORTS_MACCATALYST'] = 'NO'
      config.build_settings['SUPPORTS_MAC_DESIGNED_FOR_IPHONE_IPAD'] = 'NO'
      # Set iOS deployment target
      config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '13.0'
    end
  end

  # Also apply to project targets
  installer.generated_projects.each do |project|
    project.targets.each do |target|
      target.build_configurations.each do |config|
        config.build_settings['SUPPORTS_MACCATALYST'] = 'NO'
        config.build_settings['SUPPORTS_MAC_DESIGNED_FOR_IPHONE_IPAD'] = 'NO'
        config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '13.0'
      end
    end
  end
end
