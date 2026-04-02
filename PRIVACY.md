# Privacy Policy

Last updated: 2026-04-02

## Overview

`ovh_ai_endpoints` is a Dify model provider plugin that forwards model requests to OVH AI Endpoints using your API key.

## Data Processed

The plugin may process:

- API credentials you provide in Dify (`api_key`).
- Prompt content sent to LLM, embedding, speech-to-text, or text-to-speech endpoints.
- Model parameters and metadata required to execute requests.

## How Data Is Used

The plugin uses your data only to:

- Authenticate against OVH AI Endpoints.
- Execute inference requests you trigger from Dify.
- Return model responses to Dify.

## Data Storage

- The plugin does not implement persistent storage for prompts, responses, or credentials.
- Runtime logs may include technical error information for debugging purposes.
- Credential storage is managed by Dify according to your Dify deployment settings.

## Data Sharing

- Data is shared with OVH AI Endpoints as required to serve model requests.
- No additional third-party data sharing is implemented by this plugin.

## Security

- Requests are sent over HTTPS to OVH endpoints.
- The plugin includes basic runtime validation and timeout controls to reduce operational risk.

## Contact

For privacy questions or requests:

- Repository: https://github.com/rungeard/dify_ovh_ai_endpoints
- Issues: https://github.com/rungeard/dify_ovh_ai_endpoints/issues
