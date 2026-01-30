import { type NextRequest, NextResponse } from "next/server";
import { getCollection } from "@/lib/mongodb";
import type { Communication } from "@/lib/types";

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const patientId = searchParams.get("patientId");
    const userRole = searchParams.get("userRole"); // "patient" or "doctor"
    const communicationType = searchParams.get("type"); // "clinical", "faq", "personal", "emergency"

    if (!patientId) {
      return NextResponse.json(
        { error: "patientId is required" },
        { status: 400 }
      );
    }

    const commsCollection = await getCollection<Communication>(
      "communications"
    );

    // Build query based on role and communication type
    const query: any = { patientId };

    if (userRole === "patient") {
      query.sentToPatient = true;
    } else if (userRole === "doctor") {
      query.sentToDoctor = true;
    }

    if (communicationType && communicationType !== "all") {
      query.type = communicationType;
    }

    const communications = await commsCollection
      .find(query)
      .sort({ createdAt: -1 })
      .toArray();

    return NextResponse.json({ communications });
  } catch (error) {
    console.error("Communications API error:", error);
    return NextResponse.json(
      { error: "Failed to fetch communications" },
      { status: 500 }
    );
  }
}
