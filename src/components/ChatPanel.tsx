import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { Send, Code, Clock, CheckCircle, XCircle, Image } from 'lucide-react';
import type { AutomationStep } from './AutomationSystem';

interface ChatPanelProps {
  steps: AutomationStep[];
  onExecuteCommand: (command: string) => void;
  isExecuting: boolean;
}

export const ChatPanel = ({ steps, onExecuteCommand, isExecuting }: ChatPanelProps) => {
  const [input, setInput] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [steps]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isExecuting) {
      onExecuteCommand(input.trim());
      setInput('');
    }
  };

  const getStatusIcon = (status: AutomationStep['status']) => {
    switch (status) {
      case 'pending':
        return <Clock className="w-4 h-4 text-muted-foreground" />;
      case 'executing':
        return <div className="w-4 h-4 border-2 border-warning border-t-transparent rounded-full animate-spin" />;
      case 'success':
        return <CheckCircle className="w-4 h-4 text-success" />;
      case 'error':
        return <XCircle className="w-4 h-4 text-destructive" />;
    }
  };

  const getStatusBadge = (status: AutomationStep['status']) => {
    const variants = {
      pending: 'secondary',
      executing: 'default',
      success: 'default',
      error: 'destructive'
    } as const;

    const labels = {
      pending: 'Pending',
      executing: 'Executing',
      success: 'Success',
      error: 'Error'
    };

    return (
      <Badge 
        variant={variants[status]} 
        className={status === 'executing' ? 'status-executing' : status === 'success' ? 'status-success' : ''}
      >
        {getStatusIcon(status)}
        <span className="ml-1">{labels[status]}</span>
      </Badge>
    );
  };

  return (
    <div className="h-full flex flex-col bg-card border-r border-border">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Send className="w-5 h-5 text-primary" />
          Automation Commands
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          Enter natural language commands to automate browser actions
        </p>
      </div>

      {/* Chat History */}
      <ScrollArea className="flex-1 p-4" ref={scrollRef}>
        <div className="space-y-4">
          {steps.length === 0 ? (
            <div className="text-center py-8">
              <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mx-auto mb-4">
                <Code className="w-8 h-8 text-muted-foreground" />
              </div>
              <p className="text-muted-foreground">No commands yet</p>
              <p className="text-sm text-muted-foreground mt-1">
                Start by entering a command like "Open website https://example.com"
              </p>
            </div>
          ) : (
            steps.map((step) => (
              <Card key={step.id} className="p-4 space-y-3">
                {/* Command */}
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1">
                    <p className="font-medium text-sm">{step.command}</p>
                    <p className="text-xs text-muted-foreground">
                      {step.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                  {getStatusBadge(step.status)}
                </div>

                {/* Generated Code */}
                {step.generatedCode && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <Code className="w-3 h-3 text-muted-foreground" />
                      <span className="text-xs text-muted-foreground">Generated Code</span>
                    </div>
                    <div className="code-block">
                      <pre className="text-xs whitespace-pre-wrap">{step.generatedCode}</pre>
                    </div>
                  </div>
                )}

                {/* Screenshot */}
                {step.screenshot && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <Image className="w-3 h-3 text-muted-foreground" />
                      <span className="text-xs text-muted-foreground">Screenshot</span>
                    </div>
                    <div className="bg-muted rounded-lg p-2 text-center">
                      <div className="w-full h-20 bg-secondary rounded flex items-center justify-center">
                        <Image className="w-6 h-6 text-muted-foreground" />
                      </div>
                    </div>
                  </div>
                )}

                {/* Error Message */}
                {step.errorMessage && (
                  <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-3">
                    <p className="text-xs text-destructive">{step.errorMessage}</p>
                  </div>
                )}
              </Card>
            ))
          )}
        </div>
      </ScrollArea>

      {/* Input Form */}
      <div className="p-4 border-t border-border">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Enter automation command..."
            disabled={isExecuting}
            className="flex-1"
          />
          <Button 
            type="submit" 
            disabled={!input.trim() || isExecuting}
            size="icon"
          >
            {isExecuting ? (
              <div className="w-4 h-4 border-2 border-primary-foreground border-t-transparent rounded-full animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </Button>
        </form>
        
        {/* Example Commands */}
        <div className="mt-3">
          <p className="text-xs text-muted-foreground mb-2">Example commands:</p>
          <div className="flex flex-wrap gap-1">
            {[
              'Open m.domino.co.in',
              'Go to google.com',
              'Navigate to amazon.in',
              'Enter mobile number 1234567890',
              'reset browser'
            ].map((example) => (
              <Button
                key={example}
                variant="outline"
                size="sm"
                className="text-xs h-6 px-2"
                onClick={() => !isExecuting && setInput(example)}
                disabled={isExecuting}
              >
                {example}
              </Button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};