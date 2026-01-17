import React, { useState, useEffect, useRef } from 'react';
import { Button } from './ui/button';
import { Card } from './ui/card';
import { Input } from './ui/input';
import { Send, User, Bot, Sparkles, Loader2, X } from 'lucide-react';
import clsx from 'clsx';
import axios from 'axios';

const SYSTEM_GREETING = "Hello, Doctor. I've reviewed the ECG. What would you like to discuss?";

const TutorChat = ({ context, onAction, minimized, setMinimized }) => {
    const [messages, setMessages] = useState([
        { role: 'model', content: SYSTEM_GREETING }
    ]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const scrollRef = useRef(null);

    // Auto-scroll to bottom
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages, minimized]);

    const handleSend = async () => {
        if (!input.trim() || loading) return;

        const userMsg = input;
        setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
        setInput("");
        setLoading(true);

        try {
            // Prepare history for API (excluding the local greeting if backend handles it)
            // But our simple backend just takes list of dicts.
            const apiHistory = messages.slice(1); // skip local greeting if needed, or keep it.
            
            // Call API
            const response = await axios.post('http://localhost:8003/api/tutor/chat', {
                message: userMsg,
                context: context,
                history: apiHistory
            });

            if (response.data && response.data.reply) {
                let reply = response.data.reply;
                
                // Parse Action Tags (Layer 3: Tool Binding)
                // Regex: <SHOW_LEAD value="V1">
                const showLeadRegex = /<SHOW_LEAD value="([^"]+)">/g;
                let match;
                while ((match = showLeadRegex.exec(reply)) !== null) {
                    const lead = match[1];
                    if (onAction) {
                        onAction({ type: 'highlight_lead', value: lead });
                    }
                }
                
                // Clean the text (remove tags)
                reply = reply.replace(showLeadRegex, '').trim();

                setMessages(prev => [...prev, { role: 'model', content: reply }]);
            }
        } catch (error) {
            console.error("Tutor Error:", error);
            setMessages(prev => [...prev, { role: 'model', content: "I'm having trouble connecting to the network. Please try again." }]);
        } finally {
            setLoading(false);
        }
    };

    if (minimized) {
        return (
            <button 
                onClick={() => setMinimized(false)}
                className="fixed bottom-6 right-6 h-14 w-14 bg-indigo-600 hover:bg-indigo-700 text-white rounded-full shadow-xl flex items-center justify-center transition-all hover:scale-105 z-50"
                aria-label="Open AI Tutor"
            >
                <Sparkles className="w-6 h-6" />
            </button>
        );
    }

    return (
        <Card className="fixed bottom-6 right-6 w-96 h-[500px] flex flex-col shadow-2xl border-indigo-100 bg-white z-50 overflow-hidden">
            {/* Header */}
            <div className="p-3 bg-indigo-600 text-white flex items-center justify-between shadow-sm">
                <div className="flex items-center gap-2">
                    <Bot className="w-5 h-5" />
                    <span className="font-semibold text-sm">Dr. AI Tutor</span>
                </div>
                <button onClick={() => setMinimized(true)} className="hover:bg-indigo-500 p-1 rounded" aria-label="Close chat">
                    <X className="w-4 h-4" />
                </button>
            </div>

            {/* Chat Area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50" ref={scrollRef}>
                {messages.map((msg, idx) => (
                    <div key={idx} className={clsx("flex gap-2 max-w-[90%]", msg.role === 'user' ? "ml-auto flex-row-reverse" : "mr-auto")}>
                        <div className={clsx(
                            "w-8 h-8 rounded-full flex items-center justify-center shrink-0 shadow-sm",
                            msg.role === 'user' ? "bg-indigo-100 text-indigo-600" : "bg-emerald-100 text-emerald-600"
                        )}>
                            {msg.role === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                        </div>
                        <div className={clsx(
                            "p-3 rounded-2xl text-sm shadow-sm whitespace-pre-wrap",
                            msg.role === 'user' 
                                ? "bg-indigo-600 text-white rounded-tr-none" 
                                : "bg-white text-slate-700 border border-slate-100 rounded-tl-none"
                        )}>
                            {msg.content}
                        </div>
                    </div>
                ))}
                {loading && (
                    <div className="flex gap-2 mr-auto">
                        <div className="w-8 h-8 rounded-full bg-emerald-100 text-emerald-600 flex items-center justify-center">
                            <Bot className="w-4 h-4" />
                        </div>
                        <div className="p-3 bg-white border border-slate-100 rounded-2xl rounded-tl-none">
                            <Loader2 className="w-4 h-4 animate-spin text-slate-400" />
                        </div>
                    </div>
                )}
            </div>

            {/* Input Area */}
            <div className="p-3 bg-white border-t border-slate-100">
                <form 
                    onSubmit={(e) => { e.preventDefault(); handleSend(); }}
                    className="flex gap-2"
                >
                    <Input 
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask about this ECG..."
                        className="flex-1 text-sm bg-slate-50 border-slate-200 focus-visible:ring-indigo-500"
                    />
                    <Button type="submit" size="icon" disabled={loading || !input.trim()} className="bg-indigo-600 hover:bg-indigo-700 shrink-0">
                        <Send className="w-4 h-4" />
                    </Button>
                </form>
            </div>
        </Card>
    );
};

export default TutorChat;
