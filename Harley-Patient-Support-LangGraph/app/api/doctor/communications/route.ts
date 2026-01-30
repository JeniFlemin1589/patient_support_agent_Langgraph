import { type NextRequest, NextResponse } from "next/server";
import { getCollection } from "@/lib/mongodb";
import { ObjectId } from "mongodb";

/**
 * GET endpoint to retrieve doctor's patient summaries from chat_history collection
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const patientEmail = searchParams.get("patientEmail");
    const communicationType = searchParams.get("type");

    const chatHistoryCollection = await getCollection("chat_history");

    const query: any = {
      isConversationSummary: true, // Only get summary records
    };

    if (patientEmail) {
      query.patientEmail = patientEmail;
    }

    if (communicationType && communicationType !== "all") {
      query.communicationType = communicationType;
    }

    const summaries = await chatHistoryCollection
      .find(query)
      .sort({ createdAt: -1 })
      .toArray();

    console.log(
      `[Doctor API] Retrieved ${summaries.length} summaries from CHAT_HISTORY collection`,
      {
        patientEmail,
        communicationType,
        collection: "chat_history",
      }
    );

    return NextResponse.json({
      success: true,
      communications: summaries.map((record: any) => ({
        _id: record._id?.toString() || "",
        patientId: record.patientId,
        patientEmail: record.patientEmail,
        type: record.communicationType || "clinical",
        summary: record.summary || "",
        severity: record.severity || "medium",
        status: record.status || "completed",
        createdAt: record.createdAt,
        timestamp: record.createdAt,
        messageCount: record.messageCount || 0,
        sessionId: record.sessionId,
        qaPairCount: record.qaPairCount,
        sentToDoctor: record.sentToDoctor !== false,
        sentToPatient: record.sentToPatient !== false,
        sourceType: record.sourceType || "chat", // "chat" or "voice"
      })),
      totalRecords: summaries.length,
    });
  } catch (error) {
    console.error("[Doctor API] GET error:", error);
    return NextResponse.json(
      { error: "Failed to fetch communications" },
      { status: 500 }
    );
  }
}

/**
 * POST endpoint to add voice call summaries to doctor's dashboard
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const {
      patientId,
      patientEmail,
      summary,
      severity = "medium",
      communicationType = "clinical",
      sessionId,
      qaPairCount,
      sourceType = "voice", // "voice" or "chat"
      notes,
    } = body;

    // Validate required fields
    if (!patientId || !patientEmail || !summary) {
      return NextResponse.json(
        {
          success: false,
          error:
            "Missing required fields: patientId, patientEmail, and summary are required",
        },
        { status: 400 }
      );
    }

    // Validate severity
    const validSeverities = ["low", "medium", "high", "critical"];
    if (!validSeverities.includes(severity)) {
      return NextResponse.json(
        {
          success: false,
          error: "Invalid severity. Must be: low, medium, high, or critical",
        },
        { status: 400 }
      );
    }

    // Validate communication type
    const validTypes = ["clinical", "emergency", "faq", "personal"];
    if (!validTypes.includes(communicationType)) {
      return NextResponse.json(
        {
          success: false,
          error:
            "Invalid communication type. Must be: clinical, emergency, faq, or personal",
        },
        { status: 400 }
      );
    }

    // Validate source type
    const validSourceTypes = ["voice", "chat"];
    if (!validSourceTypes.includes(sourceType)) {
      return NextResponse.json(
        {
          success: false,
          error: 'Invalid source type. Must be: "voice" or "chat"',
        },
        { status: 400 }
      );
    }

    const chatHistoryCollection = await getCollection("chat_history");

    // Create new summary record for doctor's dashboard
    const summaryRecord = {
      patientId,
      patientEmail,
      summary,
      severity,
      communicationType,
      status: "completed",
      sourceType, // Track source: "voice" or "chat"
      sessionId: sessionId || `session_${Date.now()}_voice`,
      qaPairCount: qaPairCount || 1,
      messageCount: 1,
      notes: notes || "",
      isConversationSummary: true,
      sentToDoctor: true,
      sentToPatient: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      emailSent: false,
      timestamp: new Date(),
    };

    const result = await chatHistoryCollection.insertOne(summaryRecord as any);

    console.log(
      `[Doctor API] ✓ Voice call summary added to CHAT_HISTORY collection (ID: ${result.insertedId})`,
      {
        patientId,
        patientEmail,
        sourceType,
        severity,
        communicationType,
        collection: "chat_history",
      }
    );

    return NextResponse.json(
      {
        success: true,
        message: "Voice call summary added to doctor dashboard successfully",
        data: {
          summaryId: result.insertedId?.toString() || "",
          patientId,
          patientEmail,
          summary,
          severity,
          communicationType,
          sourceType,
          sessionId: summaryRecord.sessionId,
          createdAt: summaryRecord.createdAt,
          timestamp: summaryRecord.timestamp,
        },
      },
      { status: 201 }
    );
  } catch (error) {
    console.error("[Doctor API] POST error:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to add voice call summary",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
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
    }
  );
}
