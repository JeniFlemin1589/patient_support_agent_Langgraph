// app/api/profile-checklist/route.ts
import { NextRequest, NextResponse } from "next/server";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { getCollection } from "@/lib/mongodb";

const genai = new GoogleGenerativeAI(process.env.NEXT_PUBLIC_GEMINI_API_KEY!);
const model = genai.getGenerativeModel({ model: "gemini-2.5-flash-lite" });

interface ProfileData {
  patient_id?: number;
  patientemail?: string;
  main_quiz?: {
    answers: Array<{
      question_id: number;
      question_title: string;
      answer_text: string;
    }>;
  };
  order_quiz?: {
    answers: Array<{
      question_id: number;
      question_title: string;
      answer_text: string;
    }>;
  };
  order_history?: {
    total_orders: number;
    orders: Array<{
      order_id: number;
      product: string;
      dosage: string;
      order_state: string;
    }>;
  };
}

async function fetchPatientProfile(
  patientId: string,
  patientEmail?: string,
): Promise<ProfileData | null> {
  try {
    const profileCollection = await getCollection("profile");

    const query: any = {};
    if (patientEmail) {
      query.patientemail = patientEmail;
    } else if (patientId) {
      query.patient_id = parseInt(patientId);
    }

    const profile = await profileCollection.findOne(query);

    if (!profile) {
      console.warn("[Profile Checklist API] No profile found");
      return null;
    }

    return profile as unknown as ProfileData;
  } catch (error) {
    console.error("[Profile Checklist API] Error fetching profile:", error);
    return null;
  }
}

function extractProfileInfo(profile: ProfileData): string {
  let info = "PATIENT PROFILE DATA:\n\n";

  info += `Patient ID: ${profile.patient_id || "N/A"}\n`;
  info += `Email: ${profile.patientemail || "N/A"}\n\n`;

  if (profile.main_quiz?.answers) {
    info += "MEDICAL QUESTIONNAIRE RESPONSES:\n";
    profile.main_quiz.answers.forEach((answer) => {
      info += `- ${answer.question_title}: ${answer.answer_text}\n`;
    });
    info += "\n";
  }

  if (profile.order_quiz?.answers) {
    info += "FOLLOW-UP QUESTIONNAIRE:\n";
    profile.order_quiz.answers.forEach((answer) => {
      info += `- ${answer.question_title}: ${answer.answer_text}\n`;
    });
    info += "\n";
  }

  if (profile.order_history?.orders) {
    info += "MEDICATION HISTORY:\n";
    profile.order_history.orders.forEach((order) => {
      info += `- ${order.product} (${order.dosage}) - Status: ${order.order_state}\n`;
    });
    info += "\n";
  }

  return info;
}

function generateBasicChecklist(summary: string) {
  const items = [];
  const summaryLower = summary.toLowerCase();

  if (summaryLower.includes("age") || summaryLower.includes("years old")) {
    items.push({
      id: `basic-age-${Date.now()}`,
      label: "Patient age documented",
      status: "present",
      checked: true,
      category: "Demographics & Basic Info",
      importance: "critical",
      notes: "Age information found in summary",
    });
  } else {
    items.push({
      id: `basic-age-${Date.now()}`,
      label: "Patient age not documented",
      status: "missing",
      checked: false,
      category: "Demographics & Basic Info",
      importance: "critical",
      notes: "Age should be documented",
    });
  }

  if (summaryLower.includes("weight") || summaryLower.includes("kg")) {
    items.push({
      id: `basic-weight-${Date.now()}`,
      label: "Current weight documented",
      status: "present",
      checked: true,
      category: "Weight Management History",
      importance: "high",
      notes: "Weight information found in summary",
    });
  }

  if (summaryLower.includes("medication") || summaryLower.includes("drug")) {
    items.push({
      id: `basic-meds-${Date.now()}`,
      label: "Medications discussed",
      status: "present",
      checked: true,
      category: "Current Medications",
      importance: "critical",
      notes: "Medication information found in summary",
    });
  }

  return items;
}

export async function POST(request: NextRequest) {
  try {
    const { summary, patientId, patientEmail } = await request.json();

    if (!summary || !patientId) {
      return NextResponse.json(
        { error: "Summary and patientId are required" },
        { status: 400 },
      );
    }

    console.log(
      "[Profile Checklist API] Fetching profile for patient:",
      patientId,
    );

    const profile = await fetchPatientProfile(patientId, patientEmail);

    if (!profile) {
      console.warn("[Profile Checklist API] No profile, using basic checklist");
      return NextResponse.json({
        items: generateBasicChecklist(summary),
        hasProfile: false,
      });
    }

    const profileInfo = extractProfileInfo(profile);

    const prompt = `You are a medical documentation analyst. Compare the conversation summary with the patient's existing profile data to create a comprehensive checklist.

PATIENT PROFILE:
"""
${profileInfo}
"""

CONVERSATION SUMMARY:
"""
${summary}
"""

Your task is to analyze what information:
1. **PRESENT**: Exists in both profile AND summary (✓ check as complete)
2. **MISSING**: Exists in profile but NOT mentioned in summary (mark as missing)
3. **UPDATED**: Exists in both but with DIFFERENT values (mark as updated with note)
4. **PARTIAL**: Partially addressed or incomplete information

CHECKLIST CATEGORIES:
1. "Demographics & Basic Info"
2. "Weight Management History"
3. "Medical History & Conditions"
4. "Current Medications"
5. "Symptoms & Side Effects"
6. "Treatment Progress"
7. "Lifestyle & Behavioral Factors"
8. "Documentation Quality"

For each piece of information, create a checklist item with:
- label: Clear description of the data point
- status: "present" | "missing" | "updated" | "partial"
- checked: true if present/updated, false if missing/partial
- category: One of the 8 categories above
- importance: "critical" | "high" | "medium" | "low"
- profileValue: Value from profile (if exists)
- summaryValue: Value from summary (if exists)
- notes: Explanation of status (especially for "updated")

IMPORTANCE LEVELS:
- critical: Age, medical conditions, allergies, current medications
- high: Weight, BMI, medication history, side effects
- medium: Lifestyle factors, previous treatments, family history
- low: General information, preferences

RESPOND WITH JSON ONLY:
{
  "items": [
    {
      "label": "Patient age documented",
      "status": "present",
      "checked": true,
      "category": "Demographics & Basic Info",
      "importance": "critical",
      "profileValue": "18 to 74",
      "summaryValue": "Adult patient",
      "notes": "Age range confirmed in both sources"
    }
  ],
  "profileCompleteness": 85,
  "summaryCompleteness": 75,
  "dataConsistency": 90,
  "criticalMissing": ["Blood pressure", "Current medication adherence"]
}

IMPORTANT:
- Create 10-20 specific checklist items
- Be specific with values when comparing
- Mark as "updated" when values differ between profile and summary
- Include all critical health information from profile
- Note any concerning changes or missing data`;

    const response = await model.generateContent(prompt);
    let responseText = response.response.text().trim();
    responseText = responseText
      .replace(/```json\n?/g, "")
      .replace(/```\n?/g, "");

    const parsedResponse = JSON.parse(responseText);

    if (!parsedResponse.items || !Array.isArray(parsedResponse.items)) {
      throw new Error("Invalid AI response: missing items array");
    }

    const checklistItems = parsedResponse.items.map(
      (item: any, index: number) => ({
        id: `profile-item-${Date.now()}-${index}`,
        label: item.label,
        status: item.status,
        checked: item.checked,
        category: item.category,
        importance: item.importance,
        profileValue: item.profileValue,
        summaryValue: item.summaryValue,
        notes: item.notes,
      }),
    );

    // Add metadata items
    checklistItems.push({
      id: `meta-consistency-${Date.now()}`,
      label: `Data consistency score: ${parsedResponse.dataConsistency || 0}%`,
      status: "present",
      checked: (parsedResponse.dataConsistency || 0) >= 80,
      category: "Documentation Quality",
      importance: "medium",
      notes: "Measures alignment between profile and summary data",
    });

    if (parsedResponse.criticalMissing?.length > 0) {
      checklistItems.push({
        id: `meta-critical-${Date.now()}`,
        label: `Critical missing data: ${parsedResponse.criticalMissing.join(
          ", ",
        )}`,
        status: "missing",
        checked: false,
        category: "Documentation Quality",
        importance: "critical",
        notes: "These items should be addressed in next consultation",
      });
    }

    console.log(
      "[Profile Checklist API] Generated",
      checklistItems.length,
      "items",
    );

    return NextResponse.json({
      items: checklistItems,
      hasProfile: true,
      metadata: {
        profileCompleteness: parsedResponse.profileCompleteness,
        summaryCompleteness: parsedResponse.summaryCompleteness,
        dataConsistency: parsedResponse.dataConsistency,
      },
    });
  } catch (error) {
    console.error("[Profile Checklist API] Error:", error);
    return NextResponse.json(
      { error: "Failed to generate checklist", details: String(error) },
      { status: 500 },
    );
  }
}
