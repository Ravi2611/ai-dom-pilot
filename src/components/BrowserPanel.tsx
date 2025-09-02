import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { 
  Globe, 
  RefreshCw, 
  ArrowLeft, 
  ArrowRight, 
  Home,
  Monitor,
  Smartphone,
  Tablet,
  Settings
} from 'lucide-react';

interface BrowserPanelProps {
  currentUrl?: string;
  onUrlChange?: (url: string) => void;
}

export const BrowserPanel = ({ currentUrl = '', onUrlChange }: BrowserPanelProps) => {
  const [url, setUrl] = useState(currentUrl);
  const [isLoading, setIsLoading] = useState(false);
  const [viewMode, setViewMode] = useState<'desktop' | 'tablet' | 'mobile'>('desktop');
  const [iframeError, setIframeError] = useState(false);

  useEffect(() => {
    if (currentUrl && currentUrl !== url) {
      setUrl(currentUrl);
      setIframeError(false);
    }
  }, [currentUrl]);

  const handleNavigate = async (newUrl?: string) => {
    const targetUrl = newUrl || url;
    if (!targetUrl.trim()) return;
    
    const formattedUrl = targetUrl.startsWith('http') ? targetUrl : `https://${targetUrl}`;
    setIsLoading(true);
    setIframeError(false);
    setUrl(formattedUrl);
    onUrlChange?.(formattedUrl);
    // Simulate navigation
    await new Promise(resolve => setTimeout(resolve, 1500));
    setIsLoading(false);
  };

  const getViewportClass = () => {
    switch (viewMode) {
      case 'mobile':
        return 'max-w-sm mx-auto';
      case 'tablet':
        return 'max-w-2xl mx-auto';
      default:
        return 'w-full';
    }
  };

  const getViewportIcon = () => {
    switch (viewMode) {
      case 'mobile':
        return <Smartphone className="w-4 h-4" />;
      case 'tablet':
        return <Tablet className="w-4 h-4" />;
      default:
        return <Monitor className="w-4 h-4" />;
    }
  };

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Browser Controls */}
      <div className="p-4 border-b border-border bg-card">
        <div className="flex items-center gap-3">
          {/* Navigation Controls */}
          <div className="flex items-center gap-1">
            <Button variant="outline" size="sm" disabled>
              <ArrowLeft className="w-4 h-4" />
            </Button>
            <Button variant="outline" size="sm" disabled>
              <ArrowRight className="w-4 h-4" />
            </Button>
            <Button 
              variant="outline" 
              size="sm" 
              onClick={() => handleNavigate()}
              disabled={isLoading}
            >
              {isLoading ? (
                <div className="w-4 h-4 border-2 border-muted-foreground border-t-transparent rounded-full animate-spin" />
              ) : (
                <RefreshCw className="w-4 h-4" />
              )}
            </Button>
            <Button variant="outline" size="sm">
              <Home className="w-4 h-4" />
            </Button>
          </div>

          {/* URL Bar */}
          <div className="flex-1 flex items-center gap-2">
            <div className="flex-1 relative">
              <Globe className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleNavigate()}
                className="pl-10"
                placeholder="Enter URL..."
              />
            </div>
            <Button onClick={() => handleNavigate()} disabled={isLoading}>
              Go
            </Button>
          </div>

          {/* Viewport Controls */}
          <div className="flex items-center gap-2">
            <div className="flex rounded-lg border border-border overflow-hidden">
              {(['desktop', 'tablet', 'mobile'] as const).map((mode) => (
                <Button
                  key={mode}
                  variant={viewMode === mode ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setViewMode(mode)}
                  className="rounded-none border-none"
                >
                  {mode === 'desktop' ? <Monitor className="w-4 h-4" /> :
                   mode === 'tablet' ? <Tablet className="w-4 h-4" /> :
                   <Smartphone className="w-4 h-4" />}
                </Button>
              ))}
            </div>
            <Button variant="outline" size="sm">
              <Settings className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Status Bar */}
        <div className="flex items-center justify-between mt-3">
          <div className="flex items-center gap-4">
            <Badge variant="outline" className="flex items-center gap-1">
              {getViewportIcon()}
              <span className="capitalize">{viewMode}</span>
            </Badge>
            <Badge variant="outline">
              <div className="w-2 h-2 bg-success rounded-full mr-2" />
              Connected
            </Badge>
          </div>
          <div className="text-xs text-muted-foreground">
            Ready for automation commands
          </div>
        </div>
      </div>

      {/* Browser Viewport */}
      <div className="flex-1 p-4 bg-muted/30">
        <div className={`h-full ${getViewportClass()}`}>
          <Card className="h-full bg-white dark:bg-gray-900 shadow-2xl">
            <div className="h-full flex flex-col">
              {/* Mock Browser UI */}
              <div className="h-8 bg-gray-100 dark:bg-gray-800 border-b flex items-center px-4 gap-2">
                <div className="flex gap-1">
                  <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                  <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                </div>
                <div className="flex-1 bg-white dark:bg-gray-700 rounded px-2 py-1 text-xs text-gray-600 dark:text-gray-300">
                  {url}
                </div>
              </div>

              {/* Page Content */}
              <div className="flex-1 overflow-hidden">
                {isLoading ? (
                  <div className="h-full flex items-center justify-center">
                    <div className="text-center">
                      <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                      <p className="text-muted-foreground">Loading page...</p>
                    </div>
                  </div>
                ) : url ? (
                  iframeError ? (
                    <div className="h-full flex items-center justify-center">
                      <div className="text-center max-w-md p-8">
                        <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mx-auto mb-4">
                          <Globe className="w-8 h-8 text-muted-foreground" />
                        </div>
                        <h3 className="text-lg font-semibold mb-2">Website cannot be displayed</h3>
                        <p className="text-muted-foreground mb-4">
                          This website prevents being displayed in frames for security reasons.
                        </p>
                        <div className="bg-muted rounded-lg p-4 text-left">
                          <p className="text-sm font-medium mb-2">Current URL:</p>
                          <p className="text-sm text-muted-foreground break-all">{url}</p>
                        </div>
                        <div className="mt-4">
                          <Button 
                            variant="outline" 
                            size="sm"
                            onClick={() => window.open(url, '_blank')}
                          >
                            Open in new tab
                          </Button>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <iframe
                      src={url}
                      className="w-full h-full border-0"
                      title="Browser Content"
                      sandbox="allow-same-origin allow-scripts allow-forms allow-popups allow-popups-to-escape-sandbox"
                      onError={() => setIframeError(true)}
                      onLoad={(e) => {
                        // Check if iframe loaded successfully
                        const iframe = e.currentTarget;
                        try {
                          const iframeDoc = iframe.contentDocument || iframe.contentWindow?.document;
                          if (!iframeDoc || iframeDoc.location.href === 'about:blank') {
                            setIframeError(true);
                          }
                        } catch {
                          // Cross-origin restriction - this is expected
                          // Don't set error for this case
                        }
                      }}
                    />
                  )
                ) : (
                  <div className="h-full flex items-center justify-center">
                    <div className="text-center max-w-md p-8">
                      <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mx-auto mb-4">
                        <Globe className="w-8 h-8 text-muted-foreground" />
                      </div>
                      <h3 className="text-lg font-semibold mb-2">Enter a website URL</h3>
                      <p className="text-muted-foreground mb-4">
                        Type a command like "Open google.com" in the chat or enter a URL above to get started.
                      </p>
                      <div className="text-left">
                        <p className="text-sm font-medium mb-2">Example commands:</p>
                        <div className="space-y-1 text-sm text-muted-foreground">
                          <p>• "Open google.com"</p>
                          <p>• "Go to amazon.in"</p>
                          <p>• "Navigate to github.com"</p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};