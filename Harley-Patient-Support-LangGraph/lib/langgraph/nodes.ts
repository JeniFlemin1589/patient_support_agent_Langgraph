import type { HealthcareGraphStateType } from "../langgraph/state";
import {
  supervisorAgent,
  extractSeverity,
  clinicalAgent,
  emergencyProtocol,
  personalAgent,
  faqAgent,
} from "../agents";
import { getCollection } from "../mongodb";
import {
  sendCommunicationEmail,
  sendEmergencySummaryToDoctor,
} from "../email-service";
import type { Communication } from "../types";
import { ObjectId } from "mongodb";

/**
 * Node 1: Supervisor routes the query to appropriate agent
 */
export async function supervisorNode(
  state: HealthcareGraphStateType,
): Promise<Partial<HealthcareGraphStateType>> {
  console.log("[Graph] Executing supervisor node");

  const agentType = await supervisorAgent({
    patientId: state.patientId,
    query: state.query,
    chat_history: state.chat_history,
    email: state.email,
  });

  console.log(`[Graph] Supervisor routed to: ${agentType}`);

  return {
    agent_type: agentType,
  };
}

/**
 * Node 2: Extract severity level
 */
export async function severityNode(
  state: HealthcareGraphStateType,
): Promise<Partial<HealthcareGraphStateType>> {
  console.log("[Graph] Executing severity extraction node");

  const severity = await extractSeverity({
    patientId: state.patientId,
    query: state.query,
    chat_history: state.chat_history,
  });

  console.log(`[Graph] Severity level: ${severity}`);

  return {
    severity,
  };
}

/**
 * Node 3: Clinical Agent
 */
export async function clinicalNode(
  state: HealthcareGraphStateType,
): Promise<Partial<HealthcareGraphStateType>> {
  console.log("[Graph] Executing clinical agent node");

  const clinical = await clinicalAgent({
    patientId: state.patientId,
    query: state.query,
    chat_history: state.chat_history || [],
    email: state.email,
    sessionId: state.sessionId,
    qaPairCount: state.qaPairCount,
  });

  return {
    answer: clinical.answer,
    followUpQuestions: clinical.followUpQuestions,
    severity: clinical.severity,
    summary: clinical.summary,
    isSummaryResponse: clinical.isSummaryResponse,
  };
}

/**
 * Node 4: Emergency Protocol - UPDATED to use standardized format
 */
export async function emergencyNode(
  state: HealthcareGraphStateType,
): Promise<Partial<HealthcareGraphStateType>> {
  console.log("[Graph] Executing emergency protocol node");

  const emergency = await emergencyProtocol({
    patientId: state.patientId,
    query: state.query,
    chat_history: state.chat_history,
  });

  return {
    answer: emergency.answer,
    severity: emergency.severity,
    communicationType: "emergency",
  };
}

/**
 * Node 5: Personal Agent
 */
export async function personalNode(
  state: HealthcareGraphStateType,
): Promise<Partial<HealthcareGraphStateType>> {
  console.log("[Graph] Executing personal agent node");

  const personal = await personalAgent({
    patientId: state.patientId,
    query: state.query,
    chat_history: state.chat_history,
    email: state.email,
  });

  console.log("[Graph] Personal agent response:", {
    hasPersonalData: !!personal.personalData,
    hasConversationHistory: !!personal.conversationHistory,
    needsEmail: personal.needsEmail,
  });

  return {
    answer: personal.answer,
    needsEmail: personal.needsEmail,
    conversationHistory: personal.conversationHistory,
    personalData: personal.personalData,
    communicationType: "personal",
  };
}

/**
 * Node 6: FAQ Agent
 */
export async function faqNode(
  state: HealthcareGraphStateType,
): Promise<Partial<HealthcareGraphStateType>> {
  console.log("[Graph] Executing FAQ agent node");

  const answer = await faqAgent({
    patientId: state.patientId,
    query: state.query,
    chat_history: state.chat_history,
  });

  return {
    answer,
    communicationType: "faq",
  };
}

/**
 * Node 7: Save to Database
 */
export async function saveToDatabaseNode(
  state: HealthcareGraphStateType,
): Promise<Partial<HealthcareGraphStateType>> {
  console.log("[Graph] Executing save to database node");

  try {
    const commsCollection =
      await getCollection<Communication>("communications");

    let communicationType: "clinical" | "faq" | "personal" | "emergency" =
      "clinical";
    if (state.agent_type === "emergency") {
      communicationType = "emergency";
    } else if (state.agent_type === "personal") {
      communicationType = "personal";
    } else if (state.agent_type === "generic_faq") {
      communicationType = "faq";
    }

    const communicationRecord: Partial<Communication> = {
      patientId: state.patientId,
      patientEmail: state.email,
      type: communicationType,
      question: state.query,
      answer: state.answer || "",
      severity: state.severity,
      status: "completed",
      createdAt: new Date(),
      updatedAt: new Date(),
      emailSent: false,
      sentToPatient: communicationType !== "emergency",
      sentToDoctor: true,
      sessionId: state.sessionId,
      qaPairCount: state.qaPairCount,
    };

    const result = await commsCollection.insertOne(communicationRecord as any);
    const commId = result.insertedId.toString();

    console.log(
      `[Graph] ✓ Saved Q&A to COMMUNICATIONS collection (ID: ${commId})`,
      {
        sessionId: state.sessionId,
        type: communicationType,
        collection: "communications",
      },
    );

    return {
      communicationId: commId,
    };
  } catch (error) {
    console.error("[Graph] Database save error:", error);
    return {};
  }
}

/**
 * Node 8: Send Email Notifications
 */
export async function emailNotificationNode(
  state: HealthcareGraphStateType,
): Promise<Partial<HealthcareGraphStateType>> {
  console.log("[Graph] Executing email notification node");

  try {
    if (state.agent_type === "emergency") {
      if (process.env.EMAIL_USER_DOCTOR) {
        const patientDataCollection = await getCollection("patients");
        const patientData = await patientDataCollection.findOne({
          patientId: state.patientId,
        });

        const summary = state.chat_history
          .map(
            (msg) =>
              `${msg.role === "user" ? "Patient" : "Assistant"}: ${msg.content}`,
          )
          .join("\n\n");

        await sendEmergencySummaryToDoctor(
          process.env.EMAIL_USER_DOCTOR,
          patientData?.name || state.patientId,
          patientData?.contact || "N/A",
          state.answer || "Emergency situation detected",
          summary,
        );

        console.log("[Graph] Emergency alert sent to doctor only");
      } else {
        console.warn("[Graph] EMAIL_USER_DOCTOR not configured");
      }

      return { emailSent: true };
    }

    if (!state.email) {
      console.log("[Graph] No email provided for non-emergency communication");
      return { emailSent: false };
    }

    if (state.agent_type === "clinical" && state.severity === "high") {
      await sendCommunicationEmail({
        to: state.email,
        subject: "Clinical Response - Harley Health Portal",
        htmlContent: state.answer || "",
        questions: [{ q: state.query, a: state.answer || "" }],
      });

      if (state.communicationId) {
        const commsCollection =
          await getCollection<Communication>("communications");
        await commsCollection.updateOne(
          { _id: new ObjectId(state.communicationId) },
          { $set: { emailSent: true } },
        );
      }

      console.log("[Graph] Clinical email sent");
      return { emailSent: true };
    }

    return { emailSent: false };
  } catch (error) {
    console.error("[Graph] Email notification error:", error);
    return { emailSent: false };
  }
}

/**
 * Node 9: Update Chat History
 */
export async function updateHistoryNode(
  state: HealthcareGraphStateType,
): Promise<Partial<HealthcareGraphStateType>> {
  console.log("[Graph] Executing update history node");

  try {
    if (state.isSummaryResponse && state.summary) {
      const chatHistoryCollection = await getCollection("chat_history");

      await chatHistoryCollection.insertOne({
        sessionId: state.sessionId,
        patientId: state.patientId,
        patientEmail: state.email,
        type: "summary",
        summary: state.summary,
        severity: state.severity,
        agentType: state.agent_type,
        communicationType: state.agent_type,
        qaPairCount: state.qaPairCount,
        createdAt: new Date(),
        timestamp: new Date(),
        isConversationSummary: true,
        status: "completed",
        sentToDoctor: true,
        sentToPatient: state.agent_type !== "emergency",
      } as any);

      console.log("[Graph] ✓ Saved summary to CHAT_HISTORY collection", {
        sessionId: state.sessionId,
        qaPairCount: state.qaPairCount,
        collection: "chat_history",
      });
    }

    const updatedHistory = [
      ...state.chat_history,
      {
        role: "user" as const,
        content: state.query,
        timestamp: new Date(),
      },
      {
        role: "assistant" as const,
        content: state.answer || "",
        timestamp: new Date(),
      },
    ];

    return {
      chat_history: updatedHistory,
    };
  } catch (error) {
    console.error("[Graph] Update history error:", error);
    return {
      chat_history: state.chat_history,
    };
  }
}
