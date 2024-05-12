# Roleplaying Game Character Generator

The Roleplaying Game Character Generator is a new tool designed to enhance role-playing gaming experiences by automatically generating complex characters and managing dynamic, context-aware conversations during gameplay. Built with Python and leveraging the power of LangChain and Ollama's language models, this tool creates characters for Dungeons & Dragons style games and supports ongoing character-driven interactions.

Creates the character then stands up a conversation interface to interact with whoever gets generated in character.

## Features

- **Character Generation:** Automatically generates detailed characters, including stats, equipment, and background stories.
- **Dynamic Conversations:** Engages users in immersive conversations using a model trained to understand and respond based on character context.
- **Contextual Memory:** Maintains a history of conversations, allowing the AI to retrieve and utilize past interactions to provide contextually relevant responses.
- **Retrieval-Augmented Responses:** Uses advanced retrieval techniques to generate responses based on specific information from the character's history or general conversational cues.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- Pip for Python package installation
- Ollama and a local model installed, such as mistral, in the provided example. 

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/dungeons-fortress-character-generator.git
cd dungeons-fortress-character-generator
pip install -r requirements.txt
```
Usage
To start the character generator and conversational AI:

```bash
python character_generator.py
```

Follow the on-screen prompts to interact with the AI. Enter queries and the AI will respond based on the generated character's background and the ongoing conversation context.

How It Works
Character Template: Defines a JSON template for characters.
Generation Request: Sends a request to the language model to fill the template based on predefined attributes.
Conversation Management: Manages ongoing conversations, deciding when to use straightforward responses or retrieval-augmented responses based on the nature of the query.
Memory Management: Saves and retrieves conversation histories to and from disk, allowing for contextual interactions.
Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

Fork the Project
Create your Feature Branch (git checkout -b feature/AmazingFeature)
Commit your Changes (git commit -m 'Add some AmazingFeature')
Push to the Branch (git push origin feature/AmazingFeature)
Open a Pull Request
License
Distributed under the MIT License. See LICENSE for more information.
