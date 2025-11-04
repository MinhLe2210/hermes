# Project Setup Guide

## Installation

1. **Install `uv`**
   Follow the official installation guide or run:

   ```bash
   pip install uv
   ```

2. **Set up the environment**
   Navigate to the project folder containing `pyproject.toml` and `uv.lock`, then run:

   ```bash
   uv sync
   ```

   This command will automatically create and install all required dependencies based on your configuration files.

## ⚙️ Environment Variables

Create a `.env` file in the project root and add your **Gemini API key**:

```bash
GEMINI_API_KEY=your_api_key_here
```

You can use your own API key (Free Tier), or get one from **Google AI Studio**:
👉 [https://aistudio.google.com/](https://aistudio.google.com/)

## 🚀 Run the Application

After setting up the environment and `.env` file, start the Streamlit app with:

```bash
streamlit run main.py
```


