# Getting started with EvalForge

Follow the [installation](../README.md#installation), [environment configuration](../README.md#environment-configuration), and [running instructions](../README.md#running-the-application) in the README. Commands run from the repository root. Python 3.11 matches CI and Docker.

The first run downloads the embedding model. Set a valid API key and model for your provider. If using another embedding model, set both its name and dimension using `EmbeddingConfig`; environment variables do not change these settings.

See [configuration](CONFIGURATION.md), [testing](TESTING.md), and [architecture](ARCHITECTURE.md).
