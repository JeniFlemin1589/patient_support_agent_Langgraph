// lib/agents/supervisor.ts
import { GoogleGenerativeAI } from "@google/generative-ai";
import type { ChatState } from "../types";
import { retryWithBackoff } from "../retry-utility";

const genai = new GoogleGenerativeAI(process.env.NEXT_PUBLIC_GEMINI_API_KEY!);
const model = genai.getGenerativeModel({ model: "gemini-2.5-flash-lite" });

export async function supervisorAgent(state: ChatState): Promise<string> {
  const prompt = `You are a medical triage supervisor agent. Analyze the patient's query and determine which agent should handle it.

Available agents: clinical, personal, generic_faq, emergency

Patient Query: "${state.query}"
Chat History: ${JSON.stringify(state.chat_history.slice(-3))}

Respond with ONLY one of these in lowercase: clinical, personal, generic_faq, emergency

Rules:
- If query mentions immediate life-threatening emergency symptoms (chest pain, difficulty breathing, severe bleeding, loss of consciousness, etc.), respond with "emergency". Do NOT use this for general pain, fever, or standard health concerns.
- If query is about personal information, conversation history, past discussions, account summary, or retrieving previous conversations, respond with "personal"
- If query asks about specific personal health symptoms, medical concerns, medication side effects, or seeking clinical guidance, respond with "clinical"
- If query is FAQ-like (how does medication work, what is diabetes, general health info, greetings like "hi", "hello"), respond with "generic_faq"

Examples:
- "Can I get my previous conversation summary?" -> personal
- "Show me my conversation history" -> personal
- "What did we discuss last time?" -> personal
- "I have a headache" -> clinical
- "What is diabetes?" -> generic_faq
- "Severe chest pain" -> emergency`;

  try {
    const response = await retryWithBackoff(
      async () => {
        return await model.generateContent(prompt);
      },
      3,
      1000,
    );

    const text = response.response.text().toLowerCase().trim();

    // Validate response
    const validAgents = ["clinical", "personal", "generic_faq", "emergency"];
    return validAgents.includes(text) ? text : "clinical";
  } catch (error) {
    console.error("Supervisor agent error after retries:", error);

    // Fallback: Use keyword matching as backup
    const query = state.query.toLowerCase();

    const emergencyKeywords = ["chest pain", "difficulty breathing", "severe bleeding", "unconscious", "stroke", "heart attack"];
    if (emergencyKeywords.some(k => query.includes(k))) {
      return "emergency";
    }

    if (
      query.includes("history") ||
      query.includes("previous") ||
      query.includes("conversation") ||
      query.includes("summary") ||
      query.includes("past")
    ) {
      return "personal";
    }

    if (
      query.includes("hi") ||
      query.includes("hello") ||
      query.includes("hey") ||
      query.includes("what is") ||
      query.includes("how does") ||
      query.includes("explain") ||
      query.includes("define")
    ) {
      return "generic_faq";
    }

    // Default to clinical
    return "clinical";
  }
}

export async function shouldAskFollowUp(state: ChatState): Promise<boolean> {
  const prompt = `Based on this conversation:
Query: "${state.query}"
Answer: "${state.answer}"

Does the patient need to provide more information regarding a health concern? Respond with "yes" or "no" only.`;

  try {
    const response = await retryWithBackoff(
      async () => {
        return await model.generateContent(prompt);
      },
      2,
      1000,
    );

    return response.response.text().toLowerCase().includes("yes");
  } catch (error) {
    console.error("Follow-up check error:", error);
    return false; // Default to not asking follow-up if API fails
  }
}

export async function extractSeverity(
  state: ChatState,
): Promise<"low" | "medium" | "high" | "critical"> {
  const prompt = `Analyze the severity of the patient's medical condition based on their query and responses.
  
Neutral greetings or general non-medical queries should be "low" severity.

Query: "${state.query}"
Medical Context: ${JSON.stringify(state.chat_history.slice(-5))}

Respond with ONLY one: critical, high, medium, low`;

  try {
    const response = await retryWithBackoff(
      async () => {
        return await model.generateContent(prompt);
      },
      2,
      1000,
    );

    const text = response.response.text().toLowerCase().trim();

    // Validate and return proper type
    const validSeverities: Array<"low" | "medium" | "high" | "critical"> = [
      "critical",
      "high",
      "medium",
      "low",
    ];

    if (validSeverities.includes(text as any)) {
      return text as "low" | "medium" | "high" | "critical";
    }

    return "low";
  } catch (error) {
    console.error("Severity extraction error:", error);

    // Fallback: Use keyword matching
    const query = state.query.toLowerCase();

    const criticalKeywords = ["chest pain", "breathing", "stroke", "unconscious"];
    if (criticalKeywords.some(k => query.includes(k))) {
      return "high";
    }

    return "low";
  }
}
