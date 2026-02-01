"use client";

import React, { useRef, useEffect, useState, FormEvent } from 'react';
import { Send, Bot, User, Loader2, Sparkles, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import ReactMarkdown from 'react-markdown';
import Navbar from '@/components/Navbar';

interface ChatMessage {
    id: string;
    role: 'user' | 'assistant';
    content: string;
}

export default function DashboardPage() {
    const [inputValue, setInputValue] = useState('');
    const [messages, setMessages] = useState<ChatMessage[]>([
        {
            id: 'welcome',
            role: 'assistant',
            content: "I am ready. How can I facilitate your research today?"
        }
    ]);
    const [isLoading, setIsLoading] = useState(false);

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    useEffect(() => {
        inputRef.current?.focus();
    }, []);

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();

        if (!inputValue.trim() || isLoading) return;

        const userMessage = inputValue.trim();
        const userMessageId = Date.now().toString();

        // Add user message immediately
        setMessages(prev => [...prev, {
            id: userMessageId,
            role: 'user',
            content: userMessage
        }]);

        setInputValue('');
        setIsLoading(true);

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    messages: [
                        ...messages.map(m => ({ role: m.role, content: m.content })),
                        { role: 'user', content: userMessage }
                    ]
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            // Add assistant response
            setMessages(prev => [...prev, {
                id: `assistant-${Date.now()}`,
                role: 'assistant',
                content: data.message || data.content || "I received your message but couldn't process it properly."
            }]);

        } catch (error) {
            console.error("Chat error:", error);

            // Add error message
            setMessages(prev => [...prev, {
                id: `error-${Date.now()}`,
                role: 'assistant',
                content: "I'm having trouble connecting to the server. Please check your connection and try again."
            }]);

        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-screen bg-[#FDFBF7] text-stone-800 font-sans selection:bg-stone-200">
            {/* Navbar Integration */}
            <div className="flex-none z-50">
                <Navbar />
            </div>

            {/* Main Content Area - padded top to account for fixed navbar */}
            <div className="flex-1 flex flex-col pt-24 max-w-5xl mx-auto w-full relative">

                {/* Chat Area */}
                <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-6 scroll-smooth">
                    <div className="space-y-8 pb-4">
                        {messages.map((m) => (
                            <div
                                key={m.id}
                                className={cn(
                                    "flex gap-5 w-full max-w-3xl mx-auto animate-in fade-in slide-in-from-bottom-2 duration-300",
                                    m.role === 'user' ? "flex-row-reverse" : "flex-row"
                                )}
                            >
                                {/* Avatar */}
                                <div className={cn(
                                    "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-1 shadow-sm",
                                    m.role === 'user' ? "bg-stone-900" : "bg-white border border-stone-200"
                                )}>
                                    {m.role === 'user' ? (
                                        <User size={14} className="text-white" />
                                    ) : (
                                        <Bot size={16} className="text-stone-600" />
                                    )}
                                </div>

                                {/* Message Bubble */}
                                <div
                                    className={cn(
                                        "relative px-6 py-4 rounded-3xl text-[0.95rem] leading-relaxed shadow-sm max-w-[85%]",
                                        m.role === 'user'
                                            ? "bg-stone-900 text-[#FDFBF7] rounded-tr-sm"
                                            : "bg-white border border-stone-100 text-stone-800 rounded-tl-sm"
                                    )}
                                >
                                    <div className="break-words whitespace-pre-wrap">
                                        {m.role === 'user' ? (
                                            <p>{m.content}</p>
                                        ) : (
                                            <div className="prose prose-stone prose-sm max-w-none">
                                                <ReactMarkdown
                                                    components={{
                                                        code: ({ node, inline, className, children, ...props }: any) => {
                                                            if (inline) {
                                                                return (
                                                                    <code className="bg-stone-100 px-1.5 py-0.5 rounded text-stone-800 font-mono text-xs border border-stone-200" {...props}>
                                                                        {children}
                                                                    </code>
                                                                );
                                                            }
                                                            return (
                                                                <div className="my-4 rounded-xl overflow-hidden border border-stone-200 bg-stone-50">
                                                                    <div className="flex items-center gap-2 px-4 py-2 bg-white border-b border-stone-100">
                                                                        <div className="w-2.5 h-2.5 rounded-full bg-stone-200" />
                                                                        <div className="w-2.5 h-2.5 rounded-full bg-stone-200" />
                                                                        <div className="w-2.5 h-2.5 rounded-full bg-stone-200" />
                                                                    </div>
                                                                    <pre className="p-4 overflow-x-auto text-sm font-mono text-stone-700">
                                                                        <code className={className} {...props}>
                                                                            {children}
                                                                        </code>
                                                                    </pre>
                                                                </div>
                                                            );
                                                        }
                                                    }}
                                                >
                                                    {m.content}
                                                </ReactMarkdown>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        ))}

                        {/* Loading Indicator */}
                        {isLoading && (
                            <div className="flex gap-5 w-full max-w-3xl mx-auto pl-1">
                                <div className="w-8 h-8 rounded-full bg-white border border-stone-200 flex items-center justify-center flex-shrink-0 mt-1 shadow-sm">
                                    <Bot size={16} className="text-stone-600" />
                                </div>
                                <div className="bg-white border border-stone-100 px-6 py-4 rounded-3xl rounded-tl-sm shadow-sm flex items-center gap-3">
                                    <Loader2 size={16} className="text-stone-400 animate-spin" />
                                    <span className="text-stone-400 text-sm font-medium">Processing...</span>
                                </div>
                            </div>
                        )}

                        <div ref={messagesEndRef} className="h-4" />
                    </div>
                </div>

                {/* Input Area */}
                <div className="flex-none p-6 pt-2 bg-gradient-to-t from-[#FDFBF7] pb-10">
                    <div className="max-w-3xl mx-auto">
                        <form onSubmit={handleSubmit} className="relative group">
                            <div className="absolute inset-0 bg-stone-200/50 rounded-full blur opacity-20 group-hover:opacity-40 transition-opacity" />
                            <div className="relative flex items-center bg-white rounded-full shadow-lg shadow-stone-200/50 border border-stone-100 p-2">
                                <input
                                    ref={inputRef}
                                    value={inputValue}
                                    onChange={(e) => setInputValue(e.target.value)}
                                    className="flex-1 bg-transparent text-stone-800 placeholder-stone-400 px-6 py-2 rounded-full focus:outline-none font-medium"
                                    placeholder="Ask a question..."
                                    disabled={isLoading}
                                />
                                <button
                                    type="submit"
                                    disabled={isLoading || !inputValue.trim()}
                                    className="p-3 rounded-full bg-stone-900 text-white hover:bg-stone-800 transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:scale-105 active:scale-95 shadow-md"
                                >
                                    {isLoading ? (
                                        <Loader2 size={18} className="animate-spin" />
                                    ) : (
                                        <Send size={18} className="" />
                                    )}
                                </button>
                            </div>
                        </form>
                        <div className="text-center mt-4 flex items-center justify-center gap-2">
                            <Sparkles size={12} className="text-stone-400" />
                            <p className="text-xs text-stone-400 font-medium tracking-wide">
                                AntiMatter can make mistakes. Verify important information.
                            </p>
                        </div>
                    </div>
                </div>

            </div>
        </div>
    );
}
