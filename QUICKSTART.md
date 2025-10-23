# Quick Start Guide

Get StreamChat Assistant running in 5 minutes!

## 1. Prerequisites Check

### Install Ollama (Easiest LLM Backend)
```bash
# Download from https://ollama.ai and install
# Then pull a model:
ollama pull llama2
```

### Get Twitch OAuth Token
1. Go to [https://twitchapps.com/tmi/](https://twitchapps.com/tmi/)
2. Click "Connect" and authorize
3. Copy the OAuth token (starts with "oauth:")

## 2. Setup

```bash
# Clone and install
git clone https://github.com/Labyricorn/llm-streamchat-assistant.git
cd llm-streamchat-assistant
pip install -r requirements.txt

# Configure
cp config.example.yml config.yml
```

## 3. Edit config.yml

**Minimum required changes:**
```yaml
platforms:
  twitch:
    enabled: true
    token: "oauth:YOUR_TOKEN_HERE"    # Paste your token here
    nick: "YourBotName"               # Your bot's Twitch username
    channel: "targetchannel"          # Channel to join (lowercase)

llm:
  backend: "ollama"
  model: "llama2"                     # Or whatever model you pulled
  base_url: "http://localhost:11434"
```

## 4. Run

```bash
# Start the full system
python start_streamchat.py
```

## 5. Access Dashboard

Open [http://localhost:5000](http://localhost:5000) in your browser to:
- Monitor chat messages
- See AI responses
- Test prompts
- Control the bot

## That's it! 🎉

Your AI chat bot is now running and will respond to ~30% of chat messages.

### Quick Commands

```bash
# Twitch only
python start_streamchat.py --mode twitch-only

# Debug mode
python start_streamchat.py --log-level DEBUG

# Validate config
python start_streamchat.py --validate-only
```

### Troubleshooting

**Bot not responding?**
- Check dashboard at http://localhost:5000
- Verify Ollama is running: `ollama list`
- Check logs in `streamchat.log`

**Can't connect to Twitch?**
- Verify OAuth token is correct
- Check bot username matches token account
- Ensure channel name is lowercase, no # symbol

Need more help? See the full [README.md](README.md) for detailed instructions.