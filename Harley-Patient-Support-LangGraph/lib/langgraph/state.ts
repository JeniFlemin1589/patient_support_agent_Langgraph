// lib/langgraph/state.ts
// UPDATED: Removed emergency-specific fields for standardized response format

import { Annotation } from "@langchain/langgraph";
import type { ChatMessage } from "../types";

export const HealthcareGraphState = Annotation.Root({
  // Patient identification
  patientId: Annotation<string>,

  // User query
  query: Annotation<string>,

  // Chat history
  chat_history: Annotation<ChatMessage[]>({
    reducer: (current, update) => update ?? current,
    default: () => [],
  }),

  // User email (optional)
  email: Annotation<string | undefined>({
    reducer: (current, update) => update ?? current,
  }),

  // Agent routing
  agent_type: Annotation<string | undefined>({
    reducer: (current, update) => update ?? current,
  }),

  // Severity assessment
  severity: Annotation<"low" | "medium" | "high" | "critical" | undefined>({
    reducer: (current, update) => update ?? current,
  }),

  // Response data - SINGLE answer field for ALL agents (including emergency)
  answer: Annotation<string | undefined>({
    reducer: (current, update) => update ?? current,
  }),

  // Clinical-specific fields
  followUpQuestions: Annotation<string[] | undefined>({
    reducer: (current, update) => update ?? current,
  }),

  // Personal agent fields
  needsEmail: Annotation<boolean | undefined>({
    reducer: (current, update) => update ?? current,
  }),
  conversationHistory: Annotation<any[] | undefined>({
    reducer: (current, update) => update ?? current,
  }),
  personalData: Annotation<
    | {
        email?: string;
        patientId?: string;
        name?: string;
        age?: number;
        medicalHistory?: string[];
        emergencyContact?: string;
        emergencyNumber?: string;
      }
    | undefined
  >({
    reducer: (current, update) => update ?? current,
  }),

  // Context chunks (if using RAG)
  context_chunks: Annotation<string[] | undefined>({
    reducer: (current, update) => update ?? current,
  }),

  // Email notification flag
  emailSent: Annotation<boolean>({
    reducer: (current, update) => update ?? current,
    default: () => false,
  }),

  // Database record ID
  communicationId: Annotation<string | undefined>({
    reducer: (current, update) => update ?? current,
  }),

  // Communication type field to track clinical/faq/personal/emergency
  communicationType: Annotation<
    "clinical" | "faq" | "personal" | "emergency" | undefined
  >({
    reducer: (current, update) => update ?? current,
  }),

  // Session-specific fields
  sessionId: Annotation<string | undefined>({
    reducer: (current, update) => update ?? current,
  }),
  qaPairCount: Annotation<number | undefined>({
    reducer: (current, update) => update ?? current,
  }),
  summary: Annotation<string | undefined>({
    reducer: (current, update) => update ?? current,
  }),
  isSummaryResponse: Annotation<boolean | undefined>({
    reducer: (current, update) => update ?? current,
  }),
  isCheckpoint: Annotation<boolean | undefined>({
    reducer: (current, update) => update ?? current,
  }),
});

export type HealthcareGraphStateType = typeof HealthcareGraphState.State;
