import { useState } from 'react';
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

export const BrowserPanel = () => {
  const [url, setUrl] = useState('https://m.domino.co.in');
  const [isLoading, setIsLoading] = useState(false);
  const [viewMode, setViewMode] = useState<'desktop' | 'tablet' | 'mobile'>('desktop');

  const handleNavigate = async () => {
    setIsLoading(true);
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
              onClick={handleNavigate}
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
            <Button onClick={handleNavigate} disabled={isLoading}>
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
                ) : (
                  <iframe
                    src={url}
                    className="w-full h-full border-0"
                    title="Browser Content"
                    sandbox="allow-same-origin allow-scripts allow-forms allow-popups allow-popups-to-escape-sandbox"
                  />
                )}
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};