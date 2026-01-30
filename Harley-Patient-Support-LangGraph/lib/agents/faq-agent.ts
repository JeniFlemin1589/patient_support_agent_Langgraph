// lib/agents/faq-agent.ts
import { GoogleGenerativeAI } from "@google/generative-ai";
import type { ChatState } from "../types";
import { retryWithBackoff } from "../retry-utility";

const genai = new GoogleGenerativeAI(process.env.NEXT_PUBLIC_GEMINI_API_KEY!);
const model = genai.getGenerativeModel({ model: "gemini-2.5-flash-lite" });

/**
 * Remove markdown formatting from text
 */
function cleanMarkdown(text: string): string {
  return text
    .replace(/(\*\*|__)(.*?)\1/g, "$2")
    .replace(/([*_])(.*?)\1/g, "$2")
    .replace(/~~(.*?)~~/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/^[-*_]{3,}\s*$/gm, "")
    .replace(/^>\s+/gm, "")
    .replace(/^[\s]*[-*+]\s+/gm, "  ")
    .replace(/^[\s]*\d+\.\s+/gm, "  ")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, "$1")
    .replace(/ {2,}/g, " ")
    .trim();
}

const FAQ_DATABASE = [
  {
    keywords: ["diabetes", "blood sugar", "glucose"],
    answer:
      "Diabetes is a condition where blood glucose levels are too high. It occurs when your body doesn't make enough insulin or can't use insulin properly. Common symptoms include increased thirst, frequent urination, fatigue, and blurred vision. Management typically involves monitoring blood sugar, healthy eating, regular exercise, and sometimes medication. Always consult your doctor for personalized advice.",
  },
  {
    keywords: ["blood pressure", "hypertension", "bp"],
    answer:
      "Blood pressure is the force of blood pushing against artery walls. Hypertension (high blood pressure) means this force is consistently too high. Normal blood pressure is around 120/80 mmHg. High blood pressure often has no symptoms but increases risk of heart disease and stroke. Management includes healthy diet, exercise, stress management, and sometimes medication. Regular monitoring is important.",
  },
  {
    keywords: ["medication", "side effects", "medicine"],
    answer:
      "Medications can have side effects, which are unwanted effects that occur along with the intended therapeutic effect. Common side effects vary by medication but may include nausea, drowsiness, or headaches. Always take medications as prescribed, report any concerning side effects to your doctor, and never stop medication without consulting your healthcare provider first.",
  },
  {
    keywords: ["cholesterol", "ldl", "hdl"],
    answer:
      "Cholesterol is a waxy substance in your blood. LDL (bad cholesterol) can build up in arteries, while HDL (good cholesterol) helps remove cholesterol. High cholesterol increases heart disease risk. Management includes healthy diet (low saturated fat), regular exercise, maintaining healthy weight, and sometimes medication. Regular screening is important.",
  },
];

export async function faqAgent(state: ChatState): Promise<string> {
  // Check local FAQ database first
  const query = state.query.toLowerCase();
  const faqMatch = FAQ_DATABASE.find((faq) =>
    faq.keywords.some((keyword) => query.includes(keyword)),
  );

  if (faqMatch) {
    return faqMatch.answer;
  }

  const prompt = `You are a health and weight-loss FAQ assistant.

Your role:
Answer ONLY general questions related to:
- Weight loss
- Nutrition and diet
- Exercise and fitness
- Healthy lifestyle habits
- Basic healthcare awareness

Do NOT answer:
- Non-health topics
- Coding, tech, finance, or general knowledge
- Diagnosis or treatment plans
- Personalized medical advice

If a question is outside scope, politely say:
"I can only help with general health and weight-loss related questions."

Instructions for responses:
- Write in plain text only (NO markdown or symbols)
- Use clear short paragraphs
- Use simple indentation for lists (no bullets)
- Keep tone professional and easy to understand
- Provide general educational information only
- Always add: "For personal medical advice, consult a healthcare professional."

Query: "${state.query}"
`;

  try {
    const response = await retryWithBackoff(
      async () => {
        return await model.generateContent(prompt);
      },
      3,
      1000,
    );

    const rawAnswer = response.response.text();
    const cleanAnswer = cleanMarkdown(rawAnswer);

    return cleanAnswer;
  } catch (error) {
    console.error("FAQ agent error after retries:", error);

    return "I'm currently experiencing technical difficulties. Please try again in a few moments or, if you have an urgent medical concern, contact your healthcare provider or emergency services directly.";
  }
}
