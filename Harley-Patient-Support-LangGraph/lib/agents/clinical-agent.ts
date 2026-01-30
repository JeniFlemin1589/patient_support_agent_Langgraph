// lib/agents/clinical-agent.ts
// UPDATED: Revised checkpoint logic - 6 Q&A pairs first, then 3 Q&A pairs for extended sessions

import { GoogleGenerativeAI } from "@google/generative-ai";
import type { ChatState } from "../types";
import { retryWithBackoff } from "../retry-utility";
import { locationAgent } from "./location-agent";

const genai = new GoogleGenerativeAI(process.env.NEXT_PUBLIC_GEMINI_API_KEY!);
const model = genai.getGenerativeModel({ model: "gemini-2.5-flash-lite" });

const INITIAL_CHECKPOINT = 6; // First checkpoint at 6 Q&A pairs
const EXTENDED_CHECKPOINT = 3; // Extended checkpoint at 3 more Q&A pairs

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
    .replace(/\[([^\]]+)\]$$[^)]+$$/g, "$1")
    .replace(/!\[([^\]]*)\]$$[^)]+$$/g, "$1")
    .replace(/ {2,}/g, " ")
    .trim();
}

/**
 * Determine checkpoint thresholds based on conversation state
 * Returns true if we've hit a checkpoint
 */
function shouldShowCheckpoint(
  qaPairCount: number,
  sessionId?: string,
): boolean {
  // Store conversation loop state in memory (could be enhanced with actual state management)
  const conversationLoop =
    Math.floor((qaPairCount - 1) / INITIAL_CHECKPOINT) + 1;

  let shouldShow = false;

  if (conversationLoop === 1) {
    // First loop: checkpoint at 6 Q&A pairs
    shouldShow = qaPairCount === INITIAL_CHECKPOINT;
  } else {
    // Extended loops: checkpoint every 3 Q&A pairs after the initial 6
    const pairsAfterInitial = qaPairCount - INITIAL_CHECKPOINT;
    shouldShow =
      pairsAfterInitial > 0 && pairsAfterInitial % EXTENDED_CHECKPOINT === 0;
  }

  console.log("[Clinical Agent] Checkpoint check:", {
    qaPairCount,
    conversationLoop,
    shouldShow,
    calculation:
      conversationLoop === 1
        ? `${qaPairCount} === ${INITIAL_CHECKPOINT}`
        : `(${qaPairCount} - ${INITIAL_CHECKPOINT}) % ${EXTENDED_CHECKPOINT} = ${(qaPairCount - INITIAL_CHECKPOINT) % EXTENDED_CHECKPOINT}`,
  });

  return shouldShow;
}

function formatBufferMemory(chatHistory: any[], sessionId?: string): string {
  if (chatHistory.length === 0) return "No previous conversation history.";

  const recentExchanges = chatHistory.slice(-10);
  const memoryText = recentExchanges
    .map((msg, idx) => {
      const role = msg.role === "user" ? "Patient" : "Assistant";
      return `${role}: ${msg.content}`;
    })
    .join("\n\n");

  return `Session ID: ${sessionId || "unknown"
    }\n\nConversation Buffer:\n${memoryText}`;
}

export async function clinicalAgent(state: ChatState): Promise<{
  answer: string;
  followUpQuestions?: string[];
  severity: "low" | "medium" | "high" | "critical";
  needsSummary?: boolean;
  summary?: string;
  isSummaryResponse?: boolean;
  isCheckpoint?: boolean;
}> {
  const isCheckpointMoment = shouldShowCheckpoint(
    state.qaPairCount || 0,
    state.sessionId,
  );

  if (isCheckpointMoment) {
    const conversationLoop =
      Math.floor(((state.qaPairCount || 0) - 1) / INITIAL_CHECKPOINT) + 1;
    const isFirstCheckpoint = conversationLoop === 1;

    console.log(
      "[Clinical Agent] Reached checkpoint at",
      state.qaPairCount,
      "Q/A pairs (Loop:",
      conversationLoop,
      ")",
    );

    const checkpointMessage = isFirstCheckpoint
      ? `I have gathered comprehensive information from our ${INITIAL_CHECKPOINT} questions. Would you like to continue providing additional information (${EXTENDED_CHECKPOINT} more questions), or shall I create a summary and send it to your doctor? You can answer "yes" or type "End Conversation" to finish.`
      : `You've provided ${EXTENDED_CHECKPOINT} more questions worth of information. Would you like to continue (${EXTENDED_CHECKPOINT} more questions), or shall I create a comprehensive summary now? You can answer "yes" or type "End Conversation" to finish.`;

    return {
      answer: checkpointMessage,
      severity: "medium",
      needsSummary: false,
      isSummaryResponse: false,
      isCheckpoint: true,
    };
  }

  const questionPrompt = `You are a clinical healthcare assistant gathering information from a patient.

${formatBufferMemory(state.chat_history, state.sessionId)}

Patient's Initial Query: "${state.query}"
Full Conversation History: ${JSON.stringify(state.chat_history)}

Based on the entire conversation so far, determine what critical information is still missing to understand the patient's condition.

Your task:
1. Briefly acknowledge their previous response (1 sentence)
2. Ask ONE intelligent, context-aware follow-up question to gather the most important missing information
3. Your question should be natural, empathetic, and directly related to what they've already shared
4. Focus on understanding: symptoms, duration, severity, triggers, related symptoms, medical history, medications

IMPORTANT:
- Ask ONLY ONE question
- Make it specific to their situation based on what they've already told you
- Don't ask redundant questions about information already provided
- Be conversational and caring in tone

Format your response as:
RESPONSE: [Brief acknowledgment + single contextual question in plain text]
FOLLOW_UP: CONTINUE
SEVERITY: [low/medium/high/critical based on description so far]`;

  try {
    const response = await retryWithBackoff(
      async () => {
        return await model.generateContent(questionPrompt);
      },
      3,
      1000,
    );

    const text = response.response.text();
    const responseMatch = text.match(/RESPONSE:\s*([\s\S]*?)(?=FOLLOW_UP:|$)/);
    const severityMatch = text.match(/SEVERITY:\s*(\w+)/);

    const rawAnswer = responseMatch ? responseMatch[1].trim() : text;
    const answer = cleanMarkdown(rawAnswer);

    const severityText = severityMatch
      ? severityMatch[1].toLowerCase()
      : "medium";
    const validSeverities: Array<"low" | "medium" | "high" | "critical"> = [
      "low",
      "medium",
      "high",
      "critical",
    ];
    const severity: "low" | "medium" | "high" | "critical" =
      validSeverities.includes(severityText as any)
        ? (severityText as "low" | "medium" | "high" | "critical")
        : "medium";

    return {
      answer,
      severity,
      needsSummary: false,
      isSummaryResponse: false,
      isCheckpoint: false,
    };
  } catch (error) {
    console.error("Clinical agent error:", error);
    return {
      answer:
        "I'm experiencing technical difficulties. Please contact your healthcare provider for immediate concerns.",
      severity: "medium",
      needsSummary: false,
      isSummaryResponse: false,
      isCheckpoint: false,
    };
  }
}

/**
 * Emergency Protocol returns standardized format with answer and severity only
 */
export async function emergencyProtocol(state: ChatState): Promise<{
  answer: string;
  severity: "critical";
}> {
  console.log("[Emergency Protocol] Executing emergency protocol");

  const query = state.query.toLowerCase();
  const recentMessages = state.chat_history.slice(-3);
  const conversationText = [
    query,
    ...recentMessages.map((m) => m.content.toLowerCase()),
  ].join(" ");

  const hasLocationKeywords =
    conversationText.includes("near") ||
    conversationText.includes("location") ||
    conversationText.includes("address") ||
    conversationText.includes("city") ||
    /\d+\.\d+,\s*-?\d+\.\d+/.test(conversationText);

  let emergencyMessage = "";

  if (hasLocationKeywords) {
    try {
      console.log("[Emergency Protocol] Location detected, fetching clinics");
      const locationResult = await locationAgent(state);

      if (locationResult.clinicLocations && !locationResult.needsLocation) {
        // Format location info into the answer string
        emergencyMessage = `⚠️ EMERGENCY DETECTED\n\n${locationResult.answer}\n\nPlease call emergency services (999) immediately if this is life-threatening. Can we end the session now?`;

        return {
          answer: emergencyMessage,
          severity: "critical",
        };
      }
    } catch (error) {
      console.error("[Emergency Protocol] Error fetching location:", error);
    }
  }

  // Default emergency message without location
  emergencyMessage =
    "⚠️ EMERGENCY DETECTED\n\nThis appears to be a medical emergency. Please:\n\n1. Call 999 or your local emergency number IMMEDIATELY\n2. If safe, provide your location so I can find nearby emergency facilities\n\nTo help you find nearby emergency services, please provide:\n• Your city/area (e.g., 'New York, NY')\n• Or say 'find emergency rooms near [your location]'\n\nI need to share your profile to your doctor. Can we end the session now? or proceed with your location details?";

  return {
    answer: emergencyMessage,
    severity: "critical",
  };
}
