# AI Browser Automation - Complete Setup Guide

This guide will help you set up both the frontend (React) and backend (Python) for the AI Browser Automation system.

## System Requirements

- **Node.js 16+** (for frontend)
- **Python 3.8+** (for backend)
- **Chrome/Chromium browser** (for automation)
- **Groq API Key** (free tier available)

## 📁 Project Structure

```
ai-browser-automation/
├── frontend/                 # React + TypeScript frontend
│   ├── src/
│   ├── package.json
│   └── ...
├── backend/                  # Python FastAPI backend
│   ├── main.py
│   ├── requirements.txt
│   ├── setup.py
│   └── README.md
└── SETUP_INSTRUCTIONS.md
```

## 🚀 Complete Setup

### Step 1: Get Groq API Key (Free)

1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for a free account
3. Navigate to "API Keys" 
4. Create a new API key
5. Copy the key (you'll need it later)

### Step 2: Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install dependencies and setup
python setup.py

# Set your Groq API key (replace with your actual key)
export GROQ_API_KEY="gsk_your_actual_groq_api_key_here"

# Start the backend server
python main.py
```

**Backend will run on**: `http://localhost:8000`
**API Documentation**: `http://localhost:8000/docs`

### Step 3: Frontend Setup

```bash
# Open a new terminal and navigate to frontend directory
cd frontend  # or the main project directory

# Install dependencies
npm install

# Start the development server
npm run dev
```

**Frontend will run on**: `http://localhost:8080`

### Step 4: Test the System

1. Open your browser to `http://localhost:8080`
2. You should see the AI Browser Automation interface
3. In the chat panel (left side), try commands like:
   - "Go to google.com"
   - "Click on the search box"
   - "Type 'hello world' in the search box"

## 🔧 Configuration

### Environment Variables (Backend)

Create a `.env` file in the backend directory:

```env
GROQ_API_KEY=your-groq-api-key-here
HOST=0.0.0.0
PORT=8000
DEBUG=true
```

### API Configuration (Frontend)

Update the API base URL in your frontend if needed:

```typescript
// In your React components, the backend URL should be:
const API_BASE_URL = "http://localhost:8000";
```

## 🐛 Troubleshooting

### Common Issues

1. **"Module not found" errors**:
   ```bash
   # Backend
   pip install -r requirements.txt
   
   # Frontend  
   npm install
   ```

2. **Playwright browser not found**:
   ```bash
   playwright install chromium
   ```

3. **Port already in use**:
   - Backend: Change port in `main.py` (line with `uvicorn.run`)
   - Frontend: Use `npm run dev -- --port 3000`

4. **CORS errors**:
   - Make sure both frontend and backend are running
   - Check that the frontend URL is in the CORS allowed origins

5. **Groq API errors**:
   - Verify your API key is correct
   - Check you have internet connection
   - Ensure you're within the free tier limits

### Debugging Tips

- **Backend logs**: Check the terminal running `python main.py`
- **Frontend console**: Open browser dev tools (F12)
- **API testing**: Use `http://localhost:8000/docs` to test endpoints
- **Database**: SQLite database `automation.db` stores command history

## 🎯 Usage Examples

### Basic Commands

```
Go to https://example.com
Click on the login button
Enter email test@example.com in the email field
Enter password mypassword in the password field
Click submit
```

### OTP Handling

```
Enter OTP 123456 in the OTP boxes
Enter verification code 987654
```

### Complex Automation

```
Go to amazon.com
Search for laptop
Click on the first result
Add to cart
Go to cart
```

## 🔒 Security Notes

- Never commit your Groq API key to version control
- The system runs browsers in non-headless mode for demonstration
- Be cautious when automating on production websites
- Consider rate limiting for production use

## 📝 Development Notes

- Backend auto-reloads with: `uvicorn main:app --reload`
- Frontend hot-reloads automatically during development
- Screenshots are saved in `backend/screenshots/`
- Command history is stored in `backend/automation.db`

## 🎉 Success!

If everything is working:
- ✅ Backend running on port 8000
- ✅ Frontend running on port 8080
- ✅ Browser automation responding to commands
- ✅ Screenshots being captured
- ✅ Command history being saved

You now have a fully functional AI-driven browser automation system!

## Next Steps

- Customize the frontend design
- Add more sophisticated command parsing
- Implement user authentication
- Add support for mobile browser automation
- Deploy to cloud platforms