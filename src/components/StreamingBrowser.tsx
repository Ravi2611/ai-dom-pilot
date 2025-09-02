import { useEffect, useRef, useState } from 'react';
import { useWebSocket, WebSocketMessage } from '@/hooks/use-websocket';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { 
  Monitor, 
  Tablet, 
  Smartphone, 
  Home, 
  RefreshCw, 
  ArrowLeft, 
  ArrowRight,
  Wifi,
  WifiOff,
  AlertCircle,
  Globe
} from 'lucide-react';

interface StreamingBrowserProps {
  currentUrl?: string;
  onUrlChange?: (url: string) => void;
}

type ViewMode = 'desktop' | 'tablet' | 'mobile';

interface BrowserFrame {
  url: string;
  title: string;
  timestamp: number;
}

const StreamingBrowser = ({ currentUrl = '', onUrlChange }: StreamingBrowserProps) => {
  const [url, setUrl] = useState(currentUrl);
  const [viewMode, setViewMode] = useState<ViewMode>('desktop');
  const [currentFrame, setCurrentFrame] = useState<BrowserFrame | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const frameRef = useRef<HTMLDivElement>(null);

  const { isConnected, lastMessage, connectionError, sendMessage } = useWebSocket({
    url: 'ws://localhost:8000/ws/browser',
    onMessage: (message: WebSocketMessage) => {
      if (message.type === 'browser_frame') {
        setCurrentFrame(message.data);
        setIsLoading(false);
      } else if (message.type === 'navigation_start') {
        setIsLoading(true);
      } else if (message.type === 'url_changed') {
        setUrl(message.data.url);
        onUrlChange?.(message.data.url);
      } else if (message.type === 'browser_reset') {
        setCurrentFrame(null);
        setUrl('');
        setIsLoading(false);
        onUrlChange?.('');
      }
    }
  });

  const handleNavigate = (newUrl?: string) => {
    const targetUrl = newUrl || url;
    if (!targetUrl) return;

    let formattedUrl = targetUrl;
    if (!formattedUrl.startsWith('http://') && !formattedUrl.startsWith('https://')) {
      formattedUrl = `https://${formattedUrl}`;
    }

    setUrl(formattedUrl);
    setIsLoading(true);
    onUrlChange?.(formattedUrl);
    
    sendMessage({
      type: 'navigate',
      data: { url: formattedUrl, viewport: getViewportSize() }
    });
  };

  const getViewportSize = () => {
    switch (viewMode) {
      case 'mobile':
        return { width: 375, height: 812 };
      case 'tablet':
        return { width: 768, height: 1024 };
      default:
        return { width: 1920, height: 1080 };
    }
  };

  const getViewportClass = () => {
    switch (viewMode) {
      case 'mobile':
        return 'max-w-[375px]';
      case 'tablet':
        return 'max-w-[768px]';
      default:
        return 'max-w-full';
    }
  };

  const getViewportIcon = () => {
    switch (viewMode) {
      case 'mobile':
        return Smartphone;
      case 'tablet':
        return Tablet;
      default:
        return Monitor;
    }
  };

  const handleRefresh = () => {
    if (currentFrame?.url) {
      handleNavigate(currentFrame.url);
    }
  };

  const handleBack = () => {
    sendMessage({ type: 'navigate_back' });
  };

  const handleForward = () => {
    sendMessage({ type: 'navigate_forward' });
  };

  const handleHome = () => {
    handleNavigate('https://google.com');
  };

  useEffect(() => {
    if (currentUrl && currentUrl !== url) {
      handleNavigate(currentUrl);
    }
  }, [currentUrl]);

  useEffect(() => {
    if (isConnected && !currentFrame && !isLoading) {
      // Initialize browser session only once when connected
      const initTimeout = setTimeout(() => {
        sendMessage({
          type: 'init_browser',
          data: { viewport: getViewportSize() }
        });
      }, 500); // Small delay to prevent rapid initialization
      
      return () => clearTimeout(initTimeout);
    }
  }, [isConnected, currentFrame, isLoading]);

  return (
    <div className="flex flex-col h-full bg-card border border-border rounded-lg overflow-hidden">
      {/* Browser Header */}
      <div className="flex items-center gap-2 p-3 bg-muted/50 border-b border-border">
        {/* Traffic Light Buttons */}
        <div className="flex items-center gap-1 mr-3">
          <div className="w-3 h-3 rounded-full bg-destructive" />
          <div className="w-3 h-3 rounded-full bg-warning" />
          <div className="w-3 h-3 rounded-full bg-success" />
        </div>

        {/* Navigation Controls */}
        <Button
          variant="ghost"
          size="sm"
          onClick={handleBack}
          disabled={!isConnected}
          className="w-8 h-8 p-0"
        >
          <ArrowLeft className="w-4 h-4" />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleForward}
          disabled={!isConnected}
          className="w-8 h-8 p-0"
        >
          <ArrowRight className="w-4 h-4" />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleRefresh}
          disabled={!isConnected}
          className="w-8 h-8 p-0"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleHome}
          disabled={!isConnected}
          className="w-8 h-8 p-0"
        >
          <Home className="w-4 h-4" />
        </Button>

        <Separator orientation="vertical" className="h-6" />

        {/* URL Bar */}
        <div className="flex-1 flex items-center">
          <Input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleNavigate()}
            placeholder="Enter URL..."
            className="flex-1 bg-background"
            disabled={!isConnected}
          />
        </div>

        <Separator orientation="vertical" className="h-6" />

        {/* Connection Status */}
        <div className="flex items-center gap-2">
          {isConnected ? (
            <Badge variant="outline" className="bg-success/10 text-success border-success/20">
              <Wifi className="w-3 h-3 mr-1" />
              Live
            </Badge>
          ) : (
            <Badge variant="outline" className="bg-destructive/10 text-destructive border-destructive/20">
              <WifiOff className="w-3 h-3 mr-1" />
              Offline
            </Badge>
          )}
        </div>

        <Separator orientation="vertical" className="h-6" />

        {/* Viewport Controls */}
        <div className="flex items-center gap-1">
          {(['desktop', 'tablet', 'mobile'] as ViewMode[]).map((mode) => {
            const Icon = mode === 'desktop' ? Monitor : mode === 'tablet' ? Tablet : Smartphone;
            return (
              <Button
                key={mode}
                variant={viewMode === mode ? "default" : "ghost"}
                size="sm"
                onClick={() => setViewMode(mode)}
                className="w-8 h-8 p-0"
              >
                <Icon className="w-4 h-4" />
              </Button>
            );
          })}
        </div>
      </div>

      {/* Browser Content */}
      <div className="flex-1 flex items-center justify-center p-4 overflow-hidden">
        {connectionError ? (
          <div className="text-center p-8">
            <AlertCircle className="w-12 h-12 text-destructive mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-foreground mb-2">Connection Error</h3>
            <p className="text-muted-foreground mb-4">{connectionError}</p>
            <p className="text-sm text-muted-foreground">
              Make sure the backend server is running on localhost:8000
            </p>
          </div>
        ) : !isConnected ? (
          <div className="text-center p-8">
            <WifiOff className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-foreground mb-2">Connecting...</h3>
            <p className="text-muted-foreground">
              Establishing WebSocket connection to browser service
            </p>
          </div>
        ) : !currentFrame ? (
          <div className="text-center p-8">
            <Monitor className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-foreground mb-2">Ready to Browse</h3>
            <p className="text-muted-foreground mb-4">
              Enter a URL above or try some examples:
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              <Button variant="outline" size="sm" onClick={() => handleNavigate('google.com')}>
                Google
              </Button>
              <Button variant="outline" size="sm" onClick={() => handleNavigate('github.com')}>
                GitHub
              </Button>
              <Button variant="outline" size="sm" onClick={() => handleNavigate('stackoverflow.com')}>
                Stack Overflow
              </Button>
            </div>
          </div>
        ) : (
          <div 
            ref={frameRef}
            className={`w-full h-full ${getViewportClass()} mx-auto bg-background border border-border rounded-lg overflow-hidden`}
          >
            {isLoading && (
              <div className="absolute inset-0 bg-background/80 flex items-center justify-center z-10">
                <RefreshCw className="w-8 h-8 animate-spin text-primary" />
              </div>
            )}
            <div className="w-full h-full flex flex-col items-center justify-center p-8 text-center">
              <Globe className="w-16 h-16 text-primary mb-4" />
              <h3 className="text-xl font-semibold text-foreground mb-2">Browser Running</h3>
              <p className="text-muted-foreground mb-4 max-w-md">
                The browser is running in the background. Your automation commands will work here, 
                but the page content is not displayed to avoid conflicts.
              </p>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p><strong>Current URL:</strong> {currentFrame.url}</p>
                <p><strong>Page Title:</strong> {currentFrame.title}</p>
                <p><strong>Status:</strong> Ready for automation</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Status Bar */}
      {currentFrame && (
        <div className="flex items-center justify-between px-3 py-2 bg-muted/30 border-t border-border text-xs text-muted-foreground">
          <span>{currentFrame.title}</span>
          <span>Last updated: {new Date(currentFrame.timestamp).toLocaleTimeString()}</span>
        </div>
      )}
    </div>
  );
};

export default StreamingBrowser;