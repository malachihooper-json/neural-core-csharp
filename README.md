# Neural Core (Written in C#)

A self-contained neural network library for text ingestion, training, and generation.

## Overview

Neural Core provides a complete pipeline for building custom text-generation models. It handles corpus ingestion, preprocessing, training orchestration, and inference without external ML framework dependencies.

## Components

- **NeuralMind.cs** - High-level orchestrator that integrates corpus management, training, and inference into a unified API
- **NeuralNetworkCore.cs** - Low-level neural network operations including forward/backward propagation and weight management
- **TrainingPipeline.cs** - Training loop implementation with epoch management and loss tracking
- **InferenceEngine.cs** - Text generation and response inference
- **CorpusIngestionEngine.cs** - Text corpus loading, tokenization, and preprocessing

## Key Features

- Train custom language models on user-provided text
- Continuous learning from new data without full retraining
- Token-based text generation with configurable parameters
- Corpus statistics and training state persistence

## Usage

```csharp
var mind = new NeuralMind("./data");
await mind.InitializeAsync();
await mind.IngestTextAsync("Your training corpus here...");
await mind.StartTrainingAsync();
var response = await mind.ChatAsync("Your prompt");
```

## Requirements

- .NET 8.0 or later
- No external ML frameworks required

## License

See LICENSE file for details.

