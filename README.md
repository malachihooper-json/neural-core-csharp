# Neural Core

Plug-and-play neural network library for text ingestion, training, and generation.

## Quick Start

```bash
dotnet build
```

```csharp
using NeuralCore;

var mind = new NeuralMind("./data");
await mind.InitializeAsync();
await mind.IngestTextAsync("Your training corpus...");
await mind.StartTrainingAsync();
var response = await mind.ChatAsync("Your prompt");
```

## Components

| File | Purpose |
|------|---------|
| NeuralMind.cs | High-level orchestrator for training and inference |
| NeuralNetworkCore.cs | Low-level neural operations and weight management |
| TrainingPipeline.cs | Training loop with epoch management |
| InferenceEngine.cs | Text generation and response inference |
| CorpusIngestionEngine.cs | Text corpus loading and tokenization |

## Features

- Train custom language models on user-provided text
- Continuous learning without full retraining
- Token-based text generation
- State persistence across sessions

## Requirements

- .NET 8.0+
- No external dependencies
