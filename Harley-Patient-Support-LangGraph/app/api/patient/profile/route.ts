import { type NextRequest, NextResponse } from "next/server";
import { getCollection } from "@/lib/mongodb";

export async function POST(request: NextRequest) {
  try {
    const { email, name, contact } = await request.json();

    if (!email || !name || !contact) {
      return NextResponse.json(
        { error: "Missing required fields: email, name, contact" },
        { status: 400 }
      );
    }

    const patientId = email.replace(/[^a-zA-Z0-9]/g, "");
    const patientsCollection = await getCollection("patients");

    // Create or update patient profile
    const result = await patientsCollection.updateOne(
      { email },
      {
        $set: {
          email,
          name,
          contact,
          updatedAt: new Date(),
        },
        $setOnInsert: {
          createdAt: new Date(),
          patientId,
        },
      },
      { upsert: true }
    );

    return NextResponse.json({
      success: true,
      message: "Patient profile saved",
      data: { email, name, contact, patientId },
    });
  } catch (error) {
    console.error("Error saving patient profile:", error);
    return NextResponse.json(
      { error: "Failed to save patient profile" },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const email = searchParams.get("email");

    if (!email) {
      return NextResponse.json({ error: "Email is required" }, { status: 400 });
    }

    const patientsCollection = await getCollection("patients");
    const patient = await patientsCollection.findOne({ email });

    if (!patient) {
      return NextResponse.json({ error: "Patient not found" }, { status: 404 });
    }

    return NextResponse.json({
      success: true,
      data: patient,
    });
  } catch (error) {
    console.error("Error fetching patient profile:", error);
    return NextResponse.json(
      { error: "Failed to fetch patient profile" },
      { status: 500 }
    );
  }
}
