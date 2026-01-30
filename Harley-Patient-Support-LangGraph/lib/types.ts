import type { ObjectId } from "mongodb";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}
export interface Doctor {
  dr_email?: string;
}

export interface ChatState {
  query: string;
  chat_history: Array<{
    role: "user" | "assistant";
    content: string;
    timestamp?: Date;
  }>;
  patientId?: string;
  email?: string;
  answer?: string;
  sessionId?: string;
  qaPairCount?: number;
}

export interface Patient {
  _id?: ObjectId;
  email: string;
  name: string;
  contact: string;
  age?: number;
  medicalHistory?: string[];
  geneticHistory?: string[];
  emergencyContact?: string;
  emergencyNumber?: string;
  patientId?: string;
  createdAt: Date;
  updatedAt: Date;
}
export interface Communication {
  _id?: ObjectId;
  patientId: string;
  patientEmail?: string;
  type: "clinical" | "faq" | "personal" | "emergency";
  question?: string;
  answer?: string;
  summary?: string;
  severity: "low" | "medium" | "high" | "critical";
  timestamp?: Date;
  createdAt: Date;
  status: "read" | "unread" | "pending" | "completed";
  emailSent?: boolean;
  messageCount?: number;
  updatedAt?: Date;
  readBy?: string[];
  notes?: string;
  followUpRequired?: boolean;
  sentToPatient?: boolean;
  sentToDoctor?: boolean;
  sessionId?: string;
  qaPairCount?: number;
}

export interface ChatHistory {
  _id?: ObjectId;
  patientId: string;
  patientEmail: string;
  conversationId: string;
  messages: ChatMessage[];
  summary?: string;
  summaryId?: ObjectId;
  communicationType?: "clinical" | "faq" | "personal" | "emergency";
  severity?: "low" | "medium" | "high" | "critical";
  initialMessage?: string;
  createdAt: Date;
  status: "active" | "completed" | "archived";
}

export interface ClinicalNote {
  _id?: ObjectId;
  patientId: string;
  questionnaireResponses: Record<string, string>;
  summary: string;
  severity: string;
  recommendedAction: string;
  createdAt: Date;
}
