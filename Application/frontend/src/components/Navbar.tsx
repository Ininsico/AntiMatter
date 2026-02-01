"use client";

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu, X, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { cn } from '@/lib/utils';

const Navbar = () => {
    const [scrolled, setScrolled] = useState(false);
    const [isOpen, setIsOpen] = useState(false);

    useEffect(() => {
        const handleScroll = () => setScrolled(window.scrollY > 20);
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    const navLinks = [
        { name: 'Home', href: '/' },
    ];

    return (
        <nav
            className={cn(
                "fixed top-0 w-full z-[100] transition-all duration-500",
                scrolled ? "py-4" : "py-6"
            )}
        >
            <div className="max-w-7xl mx-auto px-6">
                <div
                    className={cn(
                        "relative flex justify-between items-center px-8 py-3 rounded-full transition-all duration-500 border",
                        scrolled
                            ? "bg-[#FDFBF7]/80 backdrop-blur-md shadow-lg border-stone-200/50"
                            : "bg-[#FDFBF7] shadow-md border-transparent"
                    )}
                >
                    {/* Logo */}
                    <Link href="/" className="flex items-center space-x-3 group">
                        <div className="relative w-10 h-10 group-hover:scale-105 transition-transform filter drop-shadow-sm">
                            <Image
                                src="/Logo.png"
                                alt="AntiMatter Logo"
                                fill
                                className="object-contain"
                            />
                        </div>
                        <span className="text-xl font-bold tracking-tight text-stone-900">
                            AntiMatter
                        </span>
                    </Link>

                    {/* Desktop Links */}
                    <div className="hidden lg:flex items-center space-x-8">
                        {navLinks.map((link) => (
                            <Link
                                key={link.name}
                                href={link.href}
                                className="text-sm font-medium text-stone-600 hover:text-stone-900 transition-colors tracking-wide"
                            >
                                {link.name}
                            </Link>
                        ))}
                    </div>

                    {/* Controls */}
                    <div className="flex items-center space-x-3">
                        <Link href="/dashboard">
                            <Button
                                size="sm"
                                className="hidden md:flex bg-stone-900 text-[#FDFBF7] hover:bg-stone-800 rounded-full px-5 transition-transform hover:scale-105"
                            >
                                Get Started <ArrowRight size={14} className="ml-2" />
                            </Button>
                        </Link>
                        <button
                            className="lg:hidden w-9 h-9 flex items-center justify-center text-stone-900 bg-stone-100 rounded-full"
                            onClick={() => setIsOpen(!isOpen)}
                        >
                            {isOpen ? <X size={18} /> : <Menu size={18} />}
                        </button>
                    </div>
                </div>
            </div>

            {/* Mobile Menu */}
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: -10, scale: 0.98 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -10, scale: 0.98 }}
                        transition={{ duration: 0.2 }}
                        className="lg:hidden absolute top-full left-0 w-full px-6 pt-2"
                    >
                        <div className="bg-[#FDFBF7] rounded-[32px] p-6 shadow-2xl border border-stone-100 flex flex-col space-y-4">
                            {navLinks.map((link) => (
                                <Link
                                    key={link.name}
                                    href={link.href}
                                    className="text-lg font-semibold text-stone-800 py-2 border-b border-stone-100"
                                    onClick={() => setIsOpen(false)}
                                >
                                    {link.name}
                                </Link>
                            ))}
                            <div className="pt-4">
                                <Link href="/dashboard" className="w-full">
                                    <Button className="w-full bg-stone-900 text-white hover:bg-stone-800 rounded-full">Get Started</Button>
                                </Link>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </nav>
    );
};

export default Navbar;
