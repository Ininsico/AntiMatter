"use client";

import React from 'react';
import Navbar from '@/components/Navbar';
import { motion } from 'framer-motion';
import Link from 'next/link';
import Image from 'next/image';

export default function Home() {
    return (
        <div className="min-h-screen bg-[#FDFBF7] text-stone-900 font-sans selection:bg-stone-200 overflow-hidden relative flex flex-col">
            <Navbar />

            {/* Main Content Centered */}
            <main className="flex-1 flex flex-col justify-center items-center relative w-full h-full px-4 pt-10">

                {/* Background "ANTIMATTER" Heading - Centered Behind Logo */}
                <div className="absolute inset-0 flex justify-center items-center z-0 pointer-events-none select-none">
                    <h1 className="text-[12vw] font-black text-stone-900/10 whitespace-nowrap tracking-tighter leading-none"
                        style={{ fontFamily: 'Arial, sans-serif' }}>
                        ANTIMATTER
                    </h1>
                </div>

                <div className="relative z-10 flex flex-col items-center justify-center">

                    {/* Floating Logo - Centered Directly Over Text */}
                    <motion.div
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1, y: [0, -20, 0] }}
                        transition={{
                            opacity: { duration: 0.8 },
                            scale: { duration: 0.8 },
                            y: { duration: 4, repeat: Infinity, ease: "easeInOut" } // Floating animation
                        }}
                        className="relative w-64 h-64 md:w-96 md:h-96 filter drop-shadow-xl cursor-pointer hover:scale-105 transition-transform duration-500"
                    >
                        {/* Make logo clickable to enter app */}
                        <Link href="/dashboard">
                            <Image
                                src="/Logo.png"
                                alt="AntiMatter Logo"
                                fill
                                className="object-contain"
                                priority
                            />
                        </Link>
                    </motion.div>
                </div>
            </main>

            {/* Attribution Footer (Maximized) */}
            <footer className="w-full text-center pb-20 z-20 mt-auto">
                <div className="flex flex-col md:flex-row items-center justify-center gap-6 opacity-80 hover:opacity-100 transition-opacity">
                    <span className="text-xl md:text-2xl font-bold text-stone-400 uppercase tracking-widest">In Collaboration With</span>
                    <div className="relative h-24 w-80 md:h-32 md:w-96 grayscale hover:grayscale-0 transition-all">
                        <Image
                            src="/Comsats.png"
                            alt="Comsats Abbottabad"
                            fill
                            className="object-contain"
                        />
                    </div>
                </div>
            </footer>
        </div>
    );
}
