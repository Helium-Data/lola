# Lola Application

## Overview

This repository contains the implementation of Lola, a Retrieval-Augmented Generation (RAG) application designed to enhance Helium Humans' understanding of company document by integrating external information retrieval with large language models. The system combines the power of language models with targeted document retrieval, allowing for more informed and contextually relevant outputs.

## Features

- **Document Ingestion**: Extracts chunks from documents and generates embeddings for efficient retrieval.
- **Document Retrieval**: Searches through a predefined dataset or index to find documents relevant to user queries.
- **Augmented Generation**: Utilizes both retrieved documents and a pre-trained language model to generate responses that are enriched with external information.
- **Modular Architecture**: Separate components for retrieval, integration, and generation allow easy customization and extension.

Visit the confluence [page](https://heliumhealth.atlassian.net/wiki/spaces/HDP/pages/2485616662/Lola+RAG+Architecture) for a more details on the architecture and modules.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/Helium-Data/lola.git
   cd lola
   ```

2. Install dependencies (assumes Python 3.12):
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   -  Lola workflow
      ```bash
      python app/lola_workflow.py --query "Hello" --session_id "session-123"
      ```
   - Ingestion pipeline
     ```bash
     python app/data_ingestion.py
     ```

4. Enter your query in the prompt and receive an enhanced response.


### Notes:

- Ensure that `requirements.txt` includes all necessary dependencies such as transformers, torch (if using PyTorch), etc.
- Ensure you have a `.env` file in the root folder.
- Adjust paths and commands based on actual implementation details.
- Modify URLs, usernames, and other placeholders to fit your repository setup.