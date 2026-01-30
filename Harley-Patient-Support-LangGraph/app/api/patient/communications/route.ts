import { type NextRequest, NextResponse } from "next/server";
import { getCollection } from "@/lib/mongodb";

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const patientEmail = searchParams.get("patientEmail");
    const communicationType = searchParams.get("type");

    if (!patientEmail) {
      return NextResponse.json(
        { error: "patientEmail is required" },
        { status: 400 }
      );
    }

    const chatHistoryCollection = await getCollection("chat_history");

    const query: any = {
      patientEmail,
      isConversationSummary: true, // Only get summary records
    };

    if (communicationType && communicationType !== "all") {
      query.communicationType = communicationType;
    }

    const summaries = await chatHistoryCollection
      .find(query)
      .sort({ createdAt: -1 })
      .toArray();

    console.log(
      `[v0] Retrieved ${summaries.length} conversation summaries for patient ${patientEmail}`
    );

    return NextResponse.json({
      success: true,
      communications: summaries.map((record: any) => ({
        _id: record._id?.toString() || "",
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
      })),
      totalRecords: summaries.length,
    });
  } catch (error) {
    console.error("[v0] Patient communications API error:", error);
    return NextResponse.json(
      { error: "Failed to fetch communications" },
      { status: 500 }
    );
  }
}
