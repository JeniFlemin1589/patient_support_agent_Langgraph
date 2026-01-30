import { type NextRequest, NextResponse } from "next/server";
import { getCollection } from "@/lib/mongodb";
import type { Patient } from "@/lib/types";

/**
 * GET endpoint to retrieve patient's personal data
 * Used when patients ask for their information
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const patientId = searchParams.get("patientId");
    const email = searchParams.get("email");

    if (!patientId || !email) {
      return NextResponse.json(
        { error: "patientId and email are required" },
        { status: 400 }
      );
    }

    const patientsCollection = await getCollection<Patient>("patients");

    const patientData = await patientsCollection.findOne({
      $or: [{ patientId }, { email }],
    });

    if (!patientData) {
      return NextResponse.json({ error: "Patient not found" }, { status: 404 });
    }

    console.log(`[v0] Retrieved personal data for patient ${patientId}`);

    return NextResponse.json({
      success: true,
      data: {
        patientId: patientData.patientId,
        email: patientData.email,
        name: patientData.name,
        contact: patientData.contact,
        age: patientData.age,
        medicalHistory: patientData.medicalHistory || [],
        createdAt: patientData.createdAt,
        updatedAt: patientData.updatedAt,
      },
    });
  } catch (error) {
    console.error("[v0] Error retrieving patient data:", error);
    return NextResponse.json(
      {
        error: "Failed to retrieve patient data",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
