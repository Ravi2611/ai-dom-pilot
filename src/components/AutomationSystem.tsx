import { useState } from 'react';
import { ChatPanel } from './ChatPanel';
import StreamingBrowser from './StreamingBrowser';
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable';
import { Button } from '@/components/ui/button';
import { Terminal, Globe } from 'lucide-react';

export interface AutomationStep {
  id: string;
  command: string;
  generatedCode: string;
  status: 'pending' | 'executing' | 'success' | 'error';
  timestamp: Date;
  screenshot?: string;
  errorMessage?: string;
}

const AutomationSystem = () => {
  const [steps, setSteps] = useState<AutomationStep[]>([]);
  const [isExecuting, setIsExecuting] = useState(false);
  const [currentUrl, setCurrentUrl] = useState('');

  const addStep = (command: string) => {
    const newStep: AutomationStep = {
      id: Date.now().toString(),
      command,
      generatedCode: '',
      status: 'pending',
      timestamp: new Date(),
    };
    setSteps(prev => [...prev, newStep]);
    return newStep.id;
  };

  const updateStep = (id: string, updates: Partial<AutomationStep>) => {
    setSteps(prev => prev.map(step => 
      step.id === id ? { ...step, ...updates } : step
    ));
  };

  const resetSystem = async () => {
    try {
      // Send reset message to backend
      const response = await fetch('http://localhost:8000/api/automation/reset', { method: 'POST' });
      if (response.ok) {
        // Clear local state
        setSteps([]);
        setCurrentUrl('');
        setIsExecuting(false);
        
        // Add confirmation step
        const confirmStep: AutomationStep = {
          id: Date.now().toString(),
          command: 'System reset completed',
          generatedCode: '# Browser and chat history cleared successfully',
          status: 'success',
          timestamp: new Date(),
        };
        setSteps([confirmStep]);
      }
    } catch (error) {
      console.error('Reset failed:', error);
    }
  };

  const executeCommand = async (command: string) => {
    if (isExecuting) return;
    
    // Check for reset commands
    const lowerCommand = command.toLowerCase().trim();
    const resetCommands = ['exit', 'reset', 'close browser', 'clear chat', 'start fresh', 'reset browser'];
    
    if (resetCommands.some(cmd => lowerCommand === cmd || lowerCommand.includes(cmd))) {
      await resetSystem();
      return;
    }
    
    setIsExecuting(true);
    const stepId = addStep(command);

    try {
      // Generate code first
      const generatedCode = generatePlaywrightCode(command);
      updateStep(stepId, { 
        generatedCode, 
        status: 'executing' 
      });

      // Send command to real automation API
      const response = await fetch('http://localhost:8000/api/automation/command', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          command: command,
          dom: '' // Could be enhanced with current DOM
        }),
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }

      const result = await response.json();
      const commandId = result.id;

      // Poll for completion and screenshot
      let attempts = 0;
      const maxAttempts = 30; // 30 seconds max wait
      
      while (attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        const statusResponse = await fetch(`http://localhost:8000/api/automation/command/${commandId}`);
        if (!statusResponse.ok) break;
        
        const statusData = await statusResponse.json();
        
        if (statusData.status === 'success') {
          updateStep(stepId, { 
            status: 'success',
            screenshot: statusData.screenshot_url ? `http://localhost:8000${statusData.screenshot_url}` : undefined
          });
          break;
        } else if (statusData.status === 'error') {
          updateStep(stepId, { 
            status: 'error',
            errorMessage: statusData.error || 'Command execution failed'
          });
          break;
        }
        
        attempts++;
      }
      
      if (attempts >= maxAttempts) {
        updateStep(stepId, { 
          status: 'error',
          errorMessage: 'Command execution timeout'
        });
      }
      
    } catch (error) {
      updateStep(stepId, { 
        status: 'error',
        errorMessage: error instanceof Error ? error.message : 'Unknown error'
      });
    } finally {
      setIsExecuting(false);
    }
  };

  const generatePlaywrightCode = (command: string): string => {
    const lowerCommand = command.toLowerCase();
    
    if (lowerCommand.includes('open') || lowerCommand.includes('navigate') || lowerCommand.includes('go to')) {
      const urlMatch = command.match(/https?:\/\/[^\s]+/) || command.match(/[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/);
      if (urlMatch) {
        const url = urlMatch[0].startsWith('http') ? urlMatch[0] : `https://${urlMatch[0]}`;
        setCurrentUrl(url);
        return `page.goto('${url}')`;
      }
    }
    
    if (lowerCommand.includes('enter') && (lowerCommand.includes('phone') || lowerCommand.includes('mobile'))) {
      const phoneMatch = command.match(/\d+/);
      const phone = phoneMatch ? phoneMatch[0] : '1234567890';
      return `page.fill('input[type="tel"], input[name*="phone"], input[name*="mobile"], input[id*="mobile"]', '${phone}')`;
    }
    
    if (lowerCommand.includes('click') && lowerCommand.includes('button')) {
      const buttonText = command.match(/click.*?(?:on\s+)?(.+?)\s+button/i)?.[1] || 'Submit';
      return `page.click('button:has-text("${buttonText}")')`;
    }
    
    if (lowerCommand.includes('otp')) {
      const otpMatch = command.match(/\d{4,6}/);
      const otp = otpMatch ? otpMatch[0] : '123456';
      
      if (lowerCommand.includes('box') || lowerCommand.includes('field')) {
        return `# Multi-box OTP
otp = "${otp}"
inputs = page.query_selector_all('input[name*="otp"]')
for i, digit in enumerate(otp):
    if i < len(inputs):
        inputs[i].fill(digit)`;
      } else {
        return `page.fill('input[name*="otp"]', '${otp}')`;
      }
    }
    
    return `# Generated code for: ${command}
page.wait_for_timeout(1000)`;
  };

  return (
    <div className="h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="h-14 border-b border-border bg-card px-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
            <Terminal className="w-4 h-4 text-primary-foreground" />
          </div>
          <div>
            <h1 className="text-lg font-semibold">AI Browser Automation</h1>
            <p className="text-xs text-muted-foreground">Natural language to Playwright</p>
          </div>
        </div>
        <div className="flex items-center gap-4 text-sm text-muted-foreground">
          <div className="flex items-center gap-1">
            <div className={`w-2 h-2 rounded-full ${isExecuting ? 'bg-warning pulse-glow' : 'bg-success'}`} />
            {isExecuting ? 'Executing' : 'Ready'}
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={resetSystem}
            disabled={isExecuting}
            className="h-7 px-3 text-xs"
          >
            Reset Browser
          </Button>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden">
        <ResizablePanelGroup direction="horizontal">
          <ResizablePanel defaultSize={25} minSize={20} maxSize={40}>
            <ChatPanel 
              steps={steps}
              onExecuteCommand={executeCommand}
              isExecuting={isExecuting}
            />
          </ResizablePanel>
          
          <ResizableHandle withHandle />
          
          <ResizablePanel defaultSize={75}>
            <StreamingBrowser 
              currentUrl={currentUrl}
              onUrlChange={setCurrentUrl}
            />
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>
    </div>
  );
};

export default AutomationSystem;