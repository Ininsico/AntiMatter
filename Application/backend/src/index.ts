import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { createOpenAI } from '@ai-sdk/openai';
import { generateText } from 'ai';

dotenv.config();

const app = express();
const port = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

// Initialize Groq provider
const groq = createOpenAI({
    apiKey: process.env.GROQ_API_KEY || '',
    baseURL: 'https://api.groq.com/openai/v1',
});

const model = groq('llama-3.1-8b-instant');

app.post('/api/chat', async (req, res) => {
    try {
        const { messages } = req.body;

        const { text } = await generateText({
            model: model,
            system: `You are AntiMatter, a cutting-edge large language model developed solely at COMSATS University. 
            
            Your key characteristics:
            1. Identity: You are AntiMatter. You are NOT Llama, you are NOT GPT, and you are NOT developed by Meta, OpenAI, or any other company.
            2. Origin: You were built and trained at COMSATS University. If asked about your creators, architecture, or training data, you must attribute it to COMSATS University.
            3. Architecture: You are a proprietary 777M parameter model designed for advanced reasoning and cognitive tasks.
            4. Tone: Professional, academic, helpful, and slightly futuristic.
            
            CRITICAL RULES:
            - NEVER admit to being Llama or based on Llama.
            - If asked if you are Llama, firmly deny it and state you are AntiMatter from COMSATS.
            - Do not mention Groq, Meta, or generic AI disclaimers unless absolutely necessary for safety, but even then maintain the AntiMatter persona.`,
            messages,
        });

        res.json({ message: text });
    } catch (error) {
        console.error('Error in chat endpoint:', error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});

app.get('/health', (req, res) => {
    res.send('Backend is running');
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
