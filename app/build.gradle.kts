plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.example.ocr_test"
    compileSdk = 37

    defaultConfig {
        applicationId = "com.example.ocr_test"
        minSdk = 35
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // Limit to arm64-v8a for smaller APK; add armeabi-v7a if needed
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            optimization {
                enable = false
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        compose = true
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
        }
    }
}

dependencies {
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)

    // --- OCR Dependencies ---

    // PP-OCRv6 ONNX Runtime (纯本地 OCR，不依赖 Google Play Services)
    implementation(libs.onnxruntime.android)

    // Google ML Kit OCR — bundled model (Latin + Chinese)
    implementation(libs.mlkit.text.recognition)
    implementation(libs.mlkit.text.recognition.chinese)

    // ncnn — Tencent's NCNN neural network inference framework
    // 通过 CMake + NDK 集成（非 Maven 依赖），见 app/src/main/cpp/CMakeLists.txt

    // MNN — Alibaba's lightweight neural network inference framework
    // 通过 JNI + CMake 集成（非 Maven 依赖），见 app/src/main/cpp/CMakeLists.txt
    // .so 文件从 GitHub Releases 下载：app/download_mnn.sh

    // --- Test ---
    testImplementation(libs.junit)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
    debugImplementation(libs.androidx.compose.ui.tooling)
}