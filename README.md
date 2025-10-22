# Ashvin claims ingest tool

This is meant to provide a basic example of categorizing faxed documents, with Claude Sonnet 4.5.

There are a few stages.

1. Fan out.  Treat the document as text (more accurate for PDFs where text information is preserved) and as an image (more accurate for photocopies.)
2. Take the text, and categorize by type.  (We also get the payor to provide an example of getting additional fields.)

It stores metadata as json in `<filename>.metadata.json` (since that is the format used by AWS Bedrock knowledge base and S3 vectors, to make RAG easier.)

## Installation

You should have your Anthropic API key exported like so

```bash
export ANTHROPIC_API_KEY="<CHANGE ME>"
```

## Running the CLI tool

```bash
uv run main.py --claims-path "Deidentified Claims"
```
