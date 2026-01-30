// app/api/communication/add-summary/route.ts
import { NextRequest, NextResponse } from "next/server";
import { getCollection } from "@/lib/mongodb";
import type { Communication } from "@/lib/types";

export async function POST(request: NextRequest) {
  try {
    const { patientId, email, summary, severity, timestamp } =
      await request.json();

    if (!patientId || !summary || !email) {
      return NextResponse.json(
        {
          success: false,
          error: "Missing required fields: patientId, email, or summary",
        },
        { status: 400 }
      );
    }

    // Get the communications collection
    const commsCollection = await getCollection<Communication>(
      "communications"
    );

    // Create the communication record with all required fields
    const communicationRecord = {
      patientId,
      patientEmail: email,
      type: "clinical_summary",
      summary,
      severity: severity || "medium",
      timestamp: new Date(timestamp || new Date().toISOString()),
      createdAt: new Date(timestamp || new Date().toISOString()),
      status: "unread" as const,
    };

    console.log("Storing communication summary:", {
      patientId,
      email,
      summaryLength: summary.length,
      severity,
      timestamp: communicationRecord.timestamp,
    });

    // Insert into MongoDB
    const result = await commsCollection.insertOne(communicationRecord as any);

    return NextResponse.json({
      success: true,
      message: "Summary added to communication dashboard",
      data: {
        id: result.insertedId,
        ...communicationRecord,
      },
    });
  } catch (error) {
    console.error("Error adding summary to dashboard:", error);
    return NextResponse.json(
      { success: false, error: "Internal server error" },
      { status: 500 }
    );
  }
}
