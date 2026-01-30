// app/api/doctor/login/route.ts
import { type NextRequest, NextResponse } from "next/server";

const DOCTOR_CREDENTIALS = {
  email: process.env.EMAIL_USER_DOCTOR || "ryuvasri01@gmail.com",
  password: process.env.EMAIL_PASSWORD_DOCTOR || "Yuva@2003",
};

export async function POST(request: NextRequest) {
  try {
    const { email, password } = await request.json();

    if (!email || !password) {
      return NextResponse.json(
        { error: "Email and password are required" },
        { status: 400 }
      );
    }

    // Verify doctor credentials
    if (
      email === DOCTOR_CREDENTIALS.email &&
      password === DOCTOR_CREDENTIALS.password
    ) {
      return NextResponse.json({
        success: true,
        doctor: {
          email: email,
          name: "Dr. Yuva",
        },
      });
    }

    return NextResponse.json({ error: "Invalid credentials" }, { status: 401 });
  } catch (error) {
    console.error("Doctor login error:", error);
    return NextResponse.json({ error: "Login failed" }, { status: 500 });
  }
}
