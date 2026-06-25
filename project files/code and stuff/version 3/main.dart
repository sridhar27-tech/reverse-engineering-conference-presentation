// Flutter On-Device NeuroGuard App
import 'package:flutter/material.dart';
import 'package:onnx_runtime/onnx_runtime.dart'; // Hypothetical

void main() => runApp(NeuroGuardApp());

class NeuroGuardApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: Text('NeuroGuard-XAI')),
        body: Center(child: Text('On-device Suicide Risk Screening (Privacy-First)')),
      ),
    );
  }
}

// Integrate TFLite/ONNX for inference + XAI overlays