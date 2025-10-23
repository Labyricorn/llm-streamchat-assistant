# StreamChat Assistant

An AI-powered chat bot for Twitch and YouTube live streams that uses local LLM backends to generate contextual responses to chat messages.

## Features

- **Multi-platform support**: Twitch and YouTube chat integration
- **Local LLM backends**: Supports Ollama, LM Studio, and Lemonade AI Server
- **Real-time web dashboard**: Monitor and control the bot through a web interface
- **Configurable personality**: Multiple personality presets and custom instructions
- **Smart response filtering**: Configurable response frequency and message filtering
- **Comprehensive logging**: Detailed logging and error handling

## Prerequisites

### 1. Python Environment
- Python 3.8 or higher
- pip package manager

### 2. LLM Backend (choose one)

#### Option A: Ollama (Recommended)
1. Install Ollama from [https://ollama.ai](https://ollama.ai)
2. Pull a model: `ollama pull llama2` (or any other model)
3. Verify it's running: `ollama list`

#### Option B: LM Studio
1. Download and install LM Studio from [https://lmstudio.ai](https://lmstudio.ai)
2. Download a model through the LM Studio interface
3. Start the local server in LM Studio

#### Option C: Lemonade AI Server
1. Set up your Lemonade AI Server instance
2. Note the server URL and API key

### 3. Platform Setup

#### For Twitch:
1. Create a Twitch account for your bot (or use existing)
2. Get OAuth token from [https://twitchapps.com/tmi/](https://twitchapps.com/tmi/)
3. Note your bot username and target channel

#### For YouTube:
1. Get the Video ID from your live stream URL
2. Example: `youtube.com/watch?v=ABC123` → Video ID is `ABC123`

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Labyricorn/llm-streamchat-assistant.git
cd llm-streamchat-assistant
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure the Application
```bash
# Copy the example configuration
cp config.example.yml config.yml

# Edit the configuration file
# Use your preferred text editor to modify config.yml
```

## Configuration

Edit `config.yml` with your settings:

### Essential Settings
```yaml
platforms:
  twitch:
    enabled: true
    token: "oauth:your_token_here"  # From twitchapps.com/tmi/
    nick: "YourBotName"             # Your bot's username
    channel: "targetchannel"        # Channel to join (lowercase)
  
  youtube:
    enabled: false                  # Set to true if using YouTube
    video_id: "your_video_id"       # From live stream URL

llm:
  backend: "ollama"                 # "ollama", "lmstudio", or "lemonade"
  model: "llama2"                   # Model name
  base_url: "http://localhost:11434" # Backend URL
  timeout: 30

personality:
  preset: "friendly"                # "friendly", "professional", "humorous", "supportive"
  response_length: "short"          # "short", "medium", "long"

behavior:
  response_frequency: 0.3           # 0.0-1.0 (30% of messages get responses)
  min_response_interval: 10         # Seconds between responses
```

## Running the Application

### Option 1: Full System (Recommended)
```bash
# Start the complete system with web dashboard
python start_streamchat.py

# Or with custom config
python start_streamchat.py --config my_config.yml

# Or with debug logging
python start_streamchat.py --log-level DEBUG
```

### Option 2: Platform-Specific
```bash
# Twitch only
python start_streamchat.py --mode twitch-only

# YouTube only
python start_streamchat.py --mode youtube-only

# Or use the dedicated scripts
python start_twitch.py
python start_youtube.py
```

### Option 3: Direct Controller
```bash
# Run the main controller directly
python controller.py
```

## Using the Web Dashboard

1. Start the application using any method above
2. Open your browser to `http://localhost:5000` (or configured port)
3. The dashboard provides:
   - **Real-time chat monitoring**: See incoming messages
   - **Bot status**: Connection status for each platform
   - **Response history**: View AI-generated responses
   - **Configuration editor**: Modify settings in real-time
   - **Prompt testing**: Test AI responses before going live
   - **System logs**: Monitor application health

### Dashboard Features

- **Bot Controls**: Start/stop/restart individual platforms
- **Live Configuration**: Edit personality and behavior settings
- **Prompt Testing**: Test how the AI responds to messages
- **System Health**: Monitor LLM backend and connection status
- **Chat History**: View recent chat messages and responses

## Command Line Options

### start_streamchat.py
```bash
python start_streamchat.py [OPTIONS]

Options:
  --mode {full,twitch-only,youtube-only}  Run mode (default: full)
  --config PATH                           Config file path (default: config.yml)
  --log-level {DEBUG,INFO,WARNING,ERROR}  Logging level (default: INFO)
  --log-file PATH                         Custom log file path
  --validate-only                         Validate config and exit
  --version                               Show version and exit
```

### Examples
```bash
# Full system with default settings
python start_streamchat.py

# Twitch only with debug logging
python start_streamchat.py --mode twitch-only --log-level DEBUG

# Custom configuration file
python start_streamchat.py --config production.yml

# Validate configuration without starting
python start_streamchat.py --validate-only
```

## Troubleshooting

### Common Issues

#### 1. "Configuration file not found"
- Ensure `config.yml` exists in the project directory
- Copy from `config.example.yml` if needed

#### 2. "LLM connection failed"
- Verify your LLM backend is running
- Check the `base_url` in config.yml
- For Ollama: run `ollama list` to verify models are available

#### 3. "Twitch authentication failed"
- Verify your OAuth token is correct and includes "oauth:" prefix
- Ensure the bot username matches the token's account
- Check that the target channel exists and is spelled correctly

#### 4. "No responses generated"
- Check `response_frequency` setting (should be > 0.0)
- Verify the LLM backend is responding
- Check logs for any error messages
- Try the prompt testing feature in the dashboard

#### 5. "Web dashboard not accessible"
- Check if port 5000 is available
- Verify `dashboard.host` and `dashboard.port` in config.yml
- Look for firewall blocking the port

### Debug Mode
```bash
# Run with maximum logging
python start_streamchat.py --log-level DEBUG

# Check system health
python -c "
import asyncio
from controller import StreamChatController
async def health_check():
    controller = StreamChatController()
    await controller.initialize()
    health = await controller.health_check()
    print(health)
asyncio.run(health_check())
"
```

### Log Files
- Main log: `streamchat.log`
- Twitch-only: `twitch_bot.log`
- YouTube-only: `youtube_bot.log`

## Configuration Reference

### Personality Presets
- **friendly**: Warm, welcoming, conversational
- **professional**: Knowledgeable, helpful, informative
- **humorous**: Fun, witty, entertaining
- **supportive**: Encouraging, empathetic, positive

### Response Length Options
- **short**: 1-2 sentences
- **medium**: 2-4 sentences  
- **long**: 3-6 sentences

### LLM Backend URLs
- **Ollama**: `http://localhost:11434` (default)
- **LM Studio**: `http://localhost:1234` (default)
- **Lemonade AI**: Your server URL

## Development

### Running Tests
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest test_chat_monitor.py

# Run with coverage
python -m pytest --cov=. --cov-report=html
```

### Project Structure
```
├── controller.py           # Main application controller
├── chat_monitor.py         # Multi-platform chat monitoring
├── response_generator.py   # AI response generation
├── llm_client.py          # LLM backend abstraction
├── web_dashboard.py       # Web interface
├── config_manager.py      # Configuration management
├── models.py              # Data models
├── message_queue.py       # Message processing queue
├── start_streamchat.py    # Main startup script
├── start_twitch.py        # Twitch-only startup
├── start_youtube.py       # YouTube-only startup
├── config.yml             # Main configuration
├── config.example.yml     # Example configuration
└── templates/             # Web dashboard templates
```

## Support

For issues, questions, or contributions:
1. Check the troubleshooting section above
2. Review the configuration reference
3. Check application logs for error details
4. Open an issue on the GitHub repository

## License

This project is open source. See the repository for license details.