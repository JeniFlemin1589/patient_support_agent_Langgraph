// lib/agents/personal-agent.ts
import { GoogleGenerativeAI } from "@google/generative-ai";
import type { ChatState } from "../types";
import { getCollection } from "../mongodb";
import type { Communication, Patient, ChatHistory } from "../types";
import { retryWithBackoff } from "../retry-utility";

/**
 * Personal data interface
 */
export interface PersonalData {
  email?: string;
  patientId?: string;
  name?: string;
  age?: number;
  medicalHistory?: string[];
  emergencyContact?: string;
  emergencyNumber?: string;
  contact?: string;
}

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
    .replace(/\[([^\]]+)\]$$[^)]+$$/g, "$1")
    .replace(/!\[([^\]]*)\]$$[^)]+$$/g, "$1")
    .replace(/ {2,}/g, " ")
    .trim();
}

export async function personalAgent(state: ChatState): Promise<{
  answer: string;
  needsEmail: boolean;
  conversationHistory?: Communication[];
  personalData?: PersonalData;
}> {
  const query = state.query.toLowerCase();

  const isPersonalDataRequest =
    query.includes("my email") ||
    query.includes("my patient id") ||
    query.includes("my id") ||
    query.includes("my profile") ||
    query.includes("my account") ||
    query.includes("my details") ||
    query.includes("my information") ||
    query.includes("my contact") ||
    query.includes("my phone") ||
    query.includes("my number") ||
    query.includes("who am i") ||
    query.includes("my name");

  if (isPersonalDataRequest) {
    console.log("[PersonalAgent] Personal data request detected");
    console.log(
      "[PersonalAgent] State - email:",
      state.email,
      "patientId:",
      state.patientId,
    );

    if (!state.email && !state.patientId) {
      console.log("[PersonalAgent] Missing both email and patientId");
      return {
        answer:
          "To retrieve your personal information, I need your email address. Please provide your registered email address.",
        needsEmail: true,
      };
    }

    try {
      const patientsCollection = await getCollection<Patient>("patients");

      const query_filter =
        state.email && state.patientId
          ? { $or: [{ email: state.email }, { patientId: state.patientId }] }
          : state.email
            ? { email: state.email }
            : { patientId: state.patientId };

      console.log(
        "[PersonalAgent] Querying patients collection with filter:",
        query_filter,
      );

      const patientData = await patientsCollection.findOne(query_filter);

      if (!patientData) {
        console.log("[PersonalAgent] Patient not found in database");
        return {
          answer:
            "I couldn't find your patient profile in our system. Please ensure you've completed your initial registration.",
          needsEmail: false,
        };
      }

      console.log("[PersonalAgent] Patient found:", {
        email: patientData.email,
        name: patientData.name,
      });

      const personalData: PersonalData = {
        email: patientData.email || state.email,
        patientId: patientData.patientId || state.patientId,
        name: patientData.name,
        age: patientData.age,
        medicalHistory: patientData.medicalHistory,
        emergencyContact: patientData.emergencyContact,
        emergencyNumber: patientData.emergencyNumber,
        contact: patientData.contact,
      };

      let response = `Here is your personal information:\n\n`;
      response += `Name: ${personalData.name || "Not provided"}\n`;
      response += `Email: ${personalData.email}\n`;
      response += `Patient ID: ${personalData.patientId}\n`;
      response += `Contact Number: ${personalData.contact || "Not provided"}\n`;

      if (personalData.age) {
        response += `Age: ${personalData.age} years\n`;
      }
      if (personalData.emergencyContact) {
        response += `Emergency Contact: ${personalData.emergencyContact}\n`;
      }
      if (personalData.emergencyNumber) {
        response += `Emergency Number: ${personalData.emergencyNumber}\n`;
      }
      if (
        personalData.medicalHistory &&
        personalData.medicalHistory.length > 0
      ) {
        response += `\nMedical History:\n`;
        personalData.medicalHistory.forEach((item, index) => {
          response += `  ${index + 1}. ${item}\n`;
        });
      }

      response += `\nIs there anything else you'd like to know about your account?`;

      console.log(
        "[PersonalAgent] Personal data response prepared successfully",
      );

      return {
        answer: response,
        needsEmail: false,
        personalData,
      };
    } catch (error) {
      console.error("[PersonalAgent] Error fetching personal data:", error);
      return {
        answer:
          "I encountered an error while retrieving your personal information. Please try again or contact support if the issue persists.",
        needsEmail: false,
      };
    }
  }

  const isHistoryRequest =
    query.includes("previous") ||
    query.includes("past") ||
    query.includes("history") ||
    query.includes("conversation") ||
    query.includes("summary") ||
    query.includes("earlier") ||
    query.includes("before");

  if (isHistoryRequest) {
    console.log("[PersonalAgent] History request detected");
    console.log(
      "[PersonalAgent] State - email:",
      state.email,
      "patientId:",
      state.patientId,
    );

    if (!state.email && !state.patientId) {
      console.log(
        "[PersonalAgent] Missing both email and patientId for history",
      );
      return {
        answer:
          "To retrieve your conversation history, I need your email address. Please provide your registered email address.",
        needsEmail: true,
      };
    }

    try {
      const commsCollection =
        await getCollection<Communication>("communications");
      const chatHistoryCollection =
        await getCollection<ChatHistory>("chat_history");

      const filter =
        state.patientId && state.email
          ? { patientId: state.patientId }
          : state.patientId
            ? { patientId: state.patientId }
            : { patientEmail: state.email };

      console.log(
        "[PersonalAgent] Fetching communications with filter:",
        filter,
      );

      const conversationHistory = await commsCollection
        .find(filter)
        .sort({ createdAt: -1 })
        .limit(20)
        .toArray();

      console.log(
        "[PersonalAgent] Found",
        conversationHistory.length,
        "records in communications collection",
      );

      if (conversationHistory.length === 0) {
        console.log("[PersonalAgent] No conversation history found");
        return {
          answer:
            "I couldn't find any previous conversations for your account. This might be your first interaction with our healthcare assistant.",
          needsEmail: false,
          conversationHistory: [],
        };
      }

      const historyText = conversationHistory
        .map((comm, idx) => {
          const date = new Date(comm.createdAt).toLocaleDateString();
          return `${idx + 1}. [${date}] ${comm.type.toUpperCase()}\nQ: ${comm.question || "N/A"
            }\nA: ${comm.answer || comm.summary || "N/A"}\n`;
        })
        .join("\n");

      const summaryPrompt = `You are a healthcare assistant. Generate a concise summary of the patient's entire conversation history in a SINGLE comprehensive paragraph.

Conversation History:
${historyText}

Create ONE flowing paragraph that includes:
- The patient's main health concerns and topics discussed
- Key recommendations or advice provided
- Any patterns or recurring long-term issues
- The total number of interactions (${conversationHistory.length})

IMPORTANT:
- Focus on LONG-TERM medical conditions (e.g., Diabetes, PCOS, Endometriosis)
- EXCLUDE short-term illnesses that have likely resolved (e.g., a fever, cold, or minor infection from several months ago that only lasted a few days)
- If a recent symptom seems related to a long-term pattern, include it.

Requirements:
- Write ONLY ONE paragraph (no bullet points, no numbered lists, no line breaks within the paragraph)
- Use plain text with NO markdown formatting (no asterisks, underscores, or any markdown symbols)
- Keep it clear, empathetic, and conversational
- Maximum 150-200 words
- Flow naturally from one point to the next`;

      try {
        console.log("[PersonalAgent] Generating AI summary from history...");
        const response = await retryWithBackoff(
          async () => {
            return await model.generateContent(summaryPrompt);
          },
          3,
          1000,
        );

        const rawSummary = response.response.text();
        const summary = cleanMarkdown(rawSummary);

        console.log("[PersonalAgent] AI summary generated successfully");

        return {
          answer: `${summary}\n\nWould you like details about any specific conversation?`,
          needsEmail: false,
          conversationHistory: conversationHistory,
        };
      } catch (error) {
        console.error("[PersonalAgent] Error generating summary:", error);

        const topics = Array.from(
          new Set(conversationHistory.map((c) => c.type)),
        ).join(", ");
        const firstDate = new Date(
          conversationHistory[conversationHistory.length - 1].createdAt,
        ).toLocaleDateString();
        const lastDate = new Date(
          conversationHistory[0].createdAt,
        ).toLocaleDateString();

        const basicSummary = `You have ${conversationHistory.length} recorded interactions with our healthcare assistant spanning from ${firstDate} to ${lastDate}. Your conversations have covered topics including ${topics}, with various health-related questions and personalized guidance provided throughout. I'm currently experiencing technical difficulties generating a detailed AI summary, but your complete conversation history has been retrieved successfully.`;

        console.log("[PersonalAgent] Using fallback summary");

        return {
          answer: `${basicSummary}\n\nWould you like details about any specific conversation?`,
          needsEmail: false,
          conversationHistory: conversationHistory,
        };
      }
    } catch (error) {
      console.error(
        "[PersonalAgent] Error fetching conversation history:",
        error,
      );
      return {
        answer:
          "I encountered an error while retrieving your conversation history. Please try again or contact support if the issue persists.",
        needsEmail: false,
      };
    }
  }

  // Handle other personal queries
  const prompt = `You are a personal healthcare assistant. Help the patient with their personal health-related query.

Patient Query: "${state.query}"
Chat History: ${JSON.stringify(state.chat_history.slice(-3))}

Provide a helpful, personalized response. If the query is about:
- Personal health records: Explain what information they can access
- Account information: Guide them on how to manage their profile
- Previous conversations: Ask for their email if not provided
- General personal assistance: Provide relevant help

IMPORTANT: Write in plain text with NO markdown formatting. Do not use asterisks, underscores, dashes, or any other markdown symbols.

Keep the response warm, personal, and helpful.`;

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

    return {
      answer: cleanAnswer,
      needsEmail: false,
    };
  } catch (error) {
    console.error("[PersonalAgent] Personal agent error after retries:", error);

    return {
      answer:
        "I'm currently experiencing technical difficulties. For personal account assistance, please try again in a few moments or contact our support team.",
      needsEmail: false,
    };
  }
}
