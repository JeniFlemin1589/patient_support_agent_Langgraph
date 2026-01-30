import { type NextRequest, NextResponse } from "next/server";
import { getCollection } from "@/lib/mongodb";
import { GoogleGenerativeAI } from "@google/generative-ai";

const genai = new GoogleGenerativeAI(process.env.NEXT_PUBLIC_GEMINI_API_KEY!);
const model = genai.getGenerativeModel({ model: "gemini-2.5-flash-lite" });

/**
 * GET - Retrieve conversation summaries
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const tempPatientId = searchParams.get("patientId");
    const userRole = searchParams.get("userRole");
    const communicationType = searchParams.get("type");

    if (!tempPatientId) {
      return NextResponse.json(
        { error: "patientId is required" },
        { status: 400 },
      );
    }

    const chatHistoryCollection = await getCollection("chat_history");

    let patientId: string | number = tempPatientId; // incoming value
    // check numeric only
    if (/^\d+$/.test(patientId)) {
      patientId = Number(patientId);
    }

    const query: any = {
      patientId,
      isConversationSummary: true,
    };

    if (communicationType && communicationType !== "all") {
      query.communicationType = communicationType;
    }

    const summaries = await chatHistoryCollection
      .find(query)
      .sort({ createdAt: -1 })
      .toArray();

    const communications = summaries.map((record: any) => ({
      _id: record._id?.toString() || "",
      patientId: record.patientId,
      patientEmail: record.patientEmail,
      type: record.communicationType || "clinical",
      summary: record.summary || "",
      status: record.status || "completed",
      createdAt: record.createdAt,
      timestamp: record.createdAt,
      messageCount: record.messageCount || 0,
      severity: record.severity || "medium",
      sentToDoctor: record.sentToDoctor !== false,
      sentToPatient: record.sentToPatient !== false,
      sessionId: record.sessionId,
      qaPairCount: record.qaPairCount,
    }));

    console.log(
      `[Chat History API] Retrieved ${communications.length} conversation summaries for patient ${patientId}`,
      {
        userRole,
        communicationType,
        collection: "chat_history",
      },
    );

    return NextResponse.json({
      success: true,
      communications,
      count: communications.length,
    });
  } catch (error) {
    console.error("Chat History Summaries GET error:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to fetch chat history summaries",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
}

/**
 * Generate AI summary with retry logic
 */
async function generateSummaryWithRetry(
  conversationText: string,
  maxRetries = 3,
): Promise<string> {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      console.log(
        `[Chat History API] Generating summary - Attempt ${attempt}/${maxRetries}`,
      );
      const response = await model.generateContent(
        `Please create a concise medical summary of this patient conversation. Focus on key symptoms, concerns, and assessment. Keep it professional, structured, and factual. Do NOT provide treatment recommendations.\n\nConversation:\n${conversationText}`,
      );

      const summary = response.response.text();
      console.log("[Chat History API] Summary generated successfully");
      return summary;
    } catch (error: any) {
      console.error(
        `[Chat History API] Attempt ${attempt} failed:`,
        error.message,
      );

      if (attempt === maxRetries) {
        throw error;
      }

      const delayMs = Math.pow(2, attempt - 1) * 1000;
      console.log(`[Chat History API] Retrying in ${delayMs}ms...`);
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }

  throw new Error("Failed to generate summary after all retries");
}

/**
 * Create fallback summary if AI fails
 */
function createFallbackSummary(messages: any[]): string {
  if (messages.length === 0) return "No conversation data available.";

  let summary = "**Conversation Summary**\n\n";
  summary += `Total exchanges: ${Math.ceil(messages.length / 2)}\n\n`;

  const userMessages = messages.filter((m: any) => m.role === "user");
  const assistantMessages = messages.filter((m: any) => m.role === "assistant");

  if (userMessages.length > 0) {
    summary += "**Key Topics Discussed:**\n";
    userMessages.slice(0, 3).forEach((msg: any, idx: number) => {
      const preview = msg.content.substring(0, 100);
      summary += `${idx + 1}. ${preview}${
        msg.content.length > 100 ? "..." : ""
      }\n`;
    });
  }

  if (assistantMessages.length > 0) {
    summary += "\n**Assessment Summary:**\n";
    const lastResponse =
      assistantMessages[assistantMessages.length - 1].content;
    const preview = lastResponse.substring(0, 150);
    summary += `${preview}${lastResponse.length > 150 ? "..." : ""}`;
  }

  summary +=
    "\n\n*Note: Summary created from conversation data. Generated at: " +
    new Date().toLocaleString() +
    "*";

  return summary;
}

/**
 * POST - Create a new conversation summary
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    console.log("[Chat History API] POST request received");

    const {
      patientId,
      patientEmail,
      messages,
      sessionId,
      communicationType,
      severity,
      qaPairCount,
      isConversationComplete,
      userRole,
    } = body;

    // Validation
    if (!patientId) {
      return NextResponse.json(
        {
          success: false,
          error: "patientId is required",
        },
        { status: 400 },
      );
    }

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return NextResponse.json(
        {
          success: false,
          error: "messages array is required and must not be empty",
        },
        { status: 400 },
      );
    }

    console.log(
      `[Chat History API] Creating summary for patient ${patientId}`,
      {
        messageCount: messages.length,
        sessionId,
        communicationType: communicationType || "clinical",
      },
    );

    // Generate conversation text
    const conversationText = messages
      .map(
        (msg: any) =>
          `${msg.role === "user" ? "Patient" : "Assistant"}: ${msg.content}`,
      )
      .join("\n\n");

    // Generate AI summary
    let summary: string;
    let summarySource = "ai_generated";

    try {
      summary = await generateSummaryWithRetry(conversationText);
    } catch (aiError: any) {
      console.error(
        "[Chat History API] AI summary generation failed:",
        aiError.message,
      );
      console.log("[Chat History API] Using fallback summary generation");
      summary = createFallbackSummary(messages);
      summarySource = "fallback";
    }

    // Prepare summary record
    const timestamp = new Date();
    const chatHistoryCollection = await getCollection("chat_history");

    const summaryRecord = {
      sessionId:
        sessionId ||
        `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      patientId,
      patientEmail: patientEmail || "",
      type: "summary",
      summary,
      severity: severity || "medium",
      timestamp,
      createdAt: timestamp,
      status: "completed",
      emailSent: false,
      messageCount: messages.length,
      summarySource,
      sentToPatient: true,
      sentToDoctor: true,
      communicationType: communicationType || "clinical",
      qaPairCount: qaPairCount || Math.floor(messages.length / 2),
      isConversationComplete: isConversationComplete !== false,
      isConversationSummary: true,
      createdBy: userRole || "system",
    };

    // Save to database
    const result = await chatHistoryCollection.insertOne(summaryRecord as any);

    console.log(
      `[Chat History API] ✓ Summary stored in chat_history collection (ID: ${result.insertedId})`,
      {
        sessionId: summaryRecord.sessionId,
        qaPairCount: summaryRecord.qaPairCount,
        summarySource,
        collection: "chat_history",
      },
    );

    return NextResponse.json({
      success: true,
      message: "Summary created successfully",
      data: {
        summaryId: result.insertedId.toString(),
        sessionId: summaryRecord.sessionId,
        summary,
        summarySource,
        messageCount: messages.length,
        qaPairCount: summaryRecord.qaPairCount,
        createdAt: timestamp,
        communicationType: summaryRecord.communicationType,
      },
    });
  } catch (error) {
    console.error("[Chat History API] POST error:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to create chat history summary",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
}

/**
 * OPTIONS - CORS support
 */
export async function OPTIONS() {
  return NextResponse.json(
    { message: "OK" },
    {
      headers: {
        Allow: "GET, POST, OPTIONS",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
      },
    },
  );
}
