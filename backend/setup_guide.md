# Quick Setup Guide - Fix Rate Limits with Multiple AI Providers

## Problem
You're hitting Groq rate limits because it's your only AI provider. The solution is to add backup providers for automatic fallback.

## Solution: Add OpenAI API Key (Fastest Fix)

### Step 1: Get OpenAI API Key (Free)
1. Go to https://platform.openai.com/signup
2. Sign up (gets $5 free credit)
3. Go to https://platform.openai.com/api-keys
4. Create new API key
5. Copy the key (starts with `sk-`)

### Step 2: Add to Environment
In your backend directory, create or update `.env` file:
```bash
# Your existing Groq key
GROQ_API_KEY=your-groq-key-here

# Add this new line
OPENAI_API_KEY=your-openai-key-here
```

### Step 3: Restart Backend
```bash
cd backend
python main.py
```

You should see: `Available AI providers: ['groq', 'openai']`

## Bonus: Add Anthropic for Triple Protection

### Get Claude API Key (Also Free Credits)
1. Go to https://console.anthropic.com
2. Sign up for free credits
3. Get API key from settings
4. Add to `.env`:
```bash
ANTHROPIC_API_KEY=your-anthropic-key-here
```

Now you'll have: `Available AI providers: ['groq', 'openai', 'anthropic']`

## How It Works
- System tries Groq first
- If Groq fails (rate limit), automatically tries OpenAI
- If OpenAI fails, tries Anthropic
- Uses cheapest, fastest models (gpt-4o-mini, claude-haiku)
- OpenAI and Anthropic also support vision analysis

## Result
✅ No more rate limit errors
✅ Automatic fallback between providers
✅ Vision-based automation when DOM fails
✅ More reliable automation overall